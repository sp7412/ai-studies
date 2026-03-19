#!/usr/bin/env python3
"""
TC-VAE RF Pulse — CI test script
Runs full pipeline (data gen, model forward/backward, loss) on CPU with
synthetic data only. No dataset download. Completes in ~60 s on a modern CPU.

Usage:
    python3 test_notebook.py          # fast smoke test (2 epochs)
    python3 test_notebook.py --full   # 20 epochs, more assertions
"""
import sys, time, argparse
import numpy as np
from scipy.signal import windows

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--full', action='store_true', help='Run longer training test')
args = parser.parse_args()

N_EPOCHS_TEST = 20 if args.full else 2
N_TRAIN       = 512 if args.full else 128

print(f'[TC-VAE test] mode={"full" if args.full else "smoke"} '
      f'epochs={N_EPOCHS_TEST} train_n={N_TRAIN}')

# ── torch ─────────────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    from dataclasses import dataclass, field
    from typing import Optional
    print(f'[OK] torch {torch.__version__} (CPU only for CI)')
except ImportError as e:
    print(f'[FAIL] torch not available: {e}')
    sys.exit(1)

DEVICE = 'cpu'
torch.manual_seed(42)
np.random.seed(42)

N_SAMPLES     = 128
N_MOD_CLASSES = 4
DIM_PULSE_DUR = 1;  DIM_MOD_TYPE  = N_MOD_CLASSES; DIM_MOD_CONT  = 8
DIM_FILTER    = 8;  DIM_RISE      = 1;              DIM_FALL      = 1
DIM_AMPLITUDE = 1;  DIM_RESIDUALS = 4
DIM_TOTAL = DIM_PULSE_DUR + DIM_MOD_TYPE + DIM_MOD_CONT + DIM_FILTER + \
            DIM_RISE + DIM_FALL + DIM_AMPLITUDE + DIM_RESIDUALS

# ─────────────────────────────────────────────────────────────────────────────
# 1. Data generation
# ─────────────────────────────────────────────────────────────────────────────
def sinc_filter(n, bw):
    t = np.arange(-(n//2), n//2 + (n % 2))
    h = np.sinc(2 * bw * t) * windows.hann(len(t))
    return h / (np.abs(h).max() + 1e-8)

def make_baseband(mod_type, pulse_len, mod_index):
    t = np.linspace(0, 1, pulse_len)
    if mod_type == 0:
        return np.exp(1j * 2 * np.pi * 0.05 * np.arange(pulse_len))
    elif mod_type == 1:
        return np.exp(1j * np.pi * (mod_index * 0.4) * t**2 * pulse_len)
    elif mod_type == 2:
        n_syms = max(2, int(pulse_len * mod_index / 4))
        sps    = max(1, pulse_len // n_syms)
        syms   = np.random.choice([-1, 1], n_syms)
        chip   = np.repeat(syms, sps)
        if len(chip) < pulse_len:
            chip = np.pad(chip, (0, pulse_len - len(chip)), mode='edge')
        return chip[:pulse_len].astype(complex)
    else:
        n_syms = max(2, int(pulse_len * mod_index / 4))
        sps    = max(1, pulse_len // n_syms)
        consts = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        syms   = consts[np.random.randint(0, 4, n_syms)]
        chip   = np.repeat(syms, sps)
        if len(chip) < pulse_len:
            chip = np.pad(chip, (0, pulse_len - len(chip)), mode='edge')
        return chip[:pulse_len]

def generate_pulse(mod_type, pulse_duration=0.8, amplitude=0.8,
                   rise_time=0.05, fall_time=0.05, filter_bw=0.3,
                   mod_index=0.5, n=N_SAMPLES):
    pulse_len = max(8, int(pulse_duration * n))
    rise_len  = max(1, int(rise_time * pulse_len))
    fall_len  = max(1, int(fall_time * pulse_len))
    baseband  = make_baseband(mod_type, pulse_len, mod_index)
    env       = np.ones(pulse_len)
    env[:rise_len]  = 0.5*(1 - np.cos(np.pi*np.arange(rise_len)/rise_len))
    env[-fall_len:] = 0.5*(1 - np.cos(np.pi*np.arange(fall_len,0,-1)/fall_len))
    baseband  = baseband * env * amplitude
    flen = min(31, pulse_len - 2)
    if flen % 2 == 0: flen -= 1
    flen = max(3, flen)
    baseband = np.convolve(baseband, sinc_filter(flen, filter_bw), mode='same')
    baseband += 0.02*(np.random.randn(pulse_len)+1j*np.random.randn(pulse_len))
    full  = np.zeros(n, dtype=complex)
    start = (n - pulse_len) // 2
    full[start:start+pulse_len] = baseband
    iq    = np.empty(n, dtype=np.float32)
    iq[0::2] = full.real[:n//2]; iq[1::2] = full.imag[:n//2]
    return iq

# Stress test data generation
print('[1/5] Data generation stress test …')
t0 = time.time()
fails = 0
for _ in range(200):
    mod = np.random.randint(0, 4)
    iq  = generate_pulse(mod,
              pulse_duration=np.random.uniform(0.5, 0.95),
              amplitude=np.random.uniform(0.3, 1.0),
              rise_time=np.random.uniform(0.02, 0.12),
              fall_time=np.random.uniform(0.02, 0.12),
              filter_bw=np.random.uniform(0.15, 0.45),
              mod_index=np.random.uniform(0.2, 0.8))
    if iq.shape != (N_SAMPLES,) or not np.isfinite(iq).all():
        fails += 1
assert fails == 0, f'Data generation: {fails}/200 failures'
print(f'[OK] 200 random pulses generated in {time.time()-t0:.2f}s')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────────────────────────────────────
print('[2/5] Building dataset …')

class PulseDataset(Dataset):
    def __init__(self, n):
        iq_list, mod_list, lab_list = [], [], []
        for _ in range(n):
            mod = np.random.randint(0, 4)
            pd  = float(np.random.uniform(0.5, 0.95))
            amp = float(np.random.uniform(0.3, 1.0))
            rt  = float(np.random.uniform(0.02, 0.12))
            ft  = float(np.random.uniform(0.02, 0.12))
            fbw = float(np.random.uniform(0.15, 0.45))
            mi  = float(np.random.uniform(0.2, 0.8))
            iq_list.append(generate_pulse(mod, pd, amp, rt, ft, fbw, mi))
            mod_list.append(mod)
            lab_list.append([pd, amp, rt, ft, fbw, mi])
        self.iq  = torch.from_numpy(np.stack(iq_list).astype(np.float32))
        self.mod = torch.tensor(mod_list, dtype=torch.long)
        self.lab = torch.from_numpy(np.array(lab_list, dtype=np.float32))
    def __len__(self): return len(self.iq)
    def __getitem__(self, i): return self.iq[i], self.mod[i], self.lab[i]

ds      = PulseDataset(N_TRAIN + 64)
tr_ds, va_ds = random_split(ds, [N_TRAIN, 64], generator=torch.Generator().manual_seed(0))
tr_dl   = DataLoader(tr_ds, batch_size=32, shuffle=True,  num_workers=0)
va_dl   = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0)
print(f'[OK] train={N_TRAIN} val=64')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────────────────────────────────────
print('[3/5] Building model …')

class ComplexLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        s = 1/np.sqrt(2*i)
        self.Wr = nn.Linear(i, o, bias=True)
        self.Wi = nn.Linear(i, o, bias=False)
        nn.init.normal_(self.Wr.weight, 0, s); nn.init.normal_(self.Wi.weight, 0, s)
    def forward(self, r, i): return self.Wr(r)-self.Wi(i), self.Wr(i)+self.Wi(r)

class CVNNDecoder(nn.Module):
    def __init__(self, d=DIM_FILTER, fl=32, h=32):
        super().__init__()
        self.embed = nn.Linear(d, h*2)
        self.cv1 = ComplexLinear(h, h); self.cv2 = ComplexLinear(h, fl)
        self.act = nn.LeakyReLU(0.1)
    def forward(self, z):
        h = self.embed(z)
        r,i = self.act(h[...,:h.shape[-1]//2]), self.act(h[...,h.shape[-1]//2:])
        r,i = self.cv1(r,i); r,i = self.act(r),self.act(i); r,i = self.cv2(r,i)
        return torch.stack([r,i],dim=-1).flatten(-2)

def gumbel_softmax(logits, tau=1.0):
    g = -torch.empty_like(logits).exponential_().log()
    return F.softmax((logits+g)/max(tau,0.05), dim=-1)

@dataclass
class NamedLatent:
    mu_pulse_dur: torch.Tensor; lv_pulse_dur: torch.Tensor
    mu_mod_type:  torch.Tensor; lv_mod_type:  torch.Tensor
    mu_mod_cont:  torch.Tensor; lv_mod_cont:  torch.Tensor
    mu_filter:    torch.Tensor; lv_filter:    torch.Tensor
    mu_rise:      torch.Tensor; lv_rise:      torch.Tensor
    mu_fall:      torch.Tensor; lv_fall:      torch.Tensor
    mu_amplitude: torch.Tensor; lv_amplitude: torch.Tensor
    mu_residuals: torch.Tensor; lv_residuals: torch.Tensor
    z_pulse_dur:  Optional[torch.Tensor] = field(default=None, repr=False)
    z_mod_type:   Optional[torch.Tensor] = field(default=None, repr=False)
    z_mod_cont:   Optional[torch.Tensor] = field(default=None, repr=False)
    z_filter:     Optional[torch.Tensor] = field(default=None, repr=False)
    z_rise:       Optional[torch.Tensor] = field(default=None, repr=False)
    z_fall:       Optional[torch.Tensor] = field(default=None, repr=False)
    z_amplitude:  Optional[torch.Tensor] = field(default=None, repr=False)
    z_residuals:  Optional[torch.Tensor] = field(default=None, repr=False)

    def reparameterize(self, tau=1.0):
        def g(mu,lv): return mu + torch.exp(0.5*lv)*torch.randn_like(lv)
        self.z_pulse_dur = g(self.mu_pulse_dur, self.lv_pulse_dur)
        self.z_mod_type  = gumbel_softmax(self.mu_mod_type, tau)
        self.z_mod_cont  = g(self.mu_mod_cont,  self.lv_mod_cont)
        self.z_filter    = g(self.mu_filter,    self.lv_filter)
        self.z_rise      = g(self.mu_rise,      self.lv_rise)
        self.z_fall      = g(self.mu_fall,      self.lv_fall)
        self.z_amplitude = g(self.mu_amplitude, self.lv_amplitude)
        self.z_residuals = g(self.mu_residuals, self.lv_residuals)

    def z_concat(self):
        return torch.cat([self.z_pulse_dur, self.z_mod_type, self.z_mod_cont,
                          self.z_filter, self.z_rise, self.z_fall,
                          self.z_amplitude, self.z_residuals], dim=-1)

    def kl_loss(self):
        def kl(mu,lv): return -0.5*(1+lv-mu.pow(2)-lv.exp()).sum(-1).mean()
        return sum(kl(getattr(self,f'mu_{s}'),getattr(self,f'lv_{s}'))
                   for s in ['pulse_dur','mod_type','mod_cont','filter',
                              'rise','fall','amplitude','residuals'])

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.te   = nn.Embedding(N_MOD_CLASSES, 16)
        self.cnn  = nn.Sequential(
            nn.Conv1d(1,32,7,stride=2,padding=3),nn.GELU(),
            nn.Conv1d(32,64,5,stride=2,padding=2),nn.GELU(),
            nn.Conv1d(64,128,3,stride=2,padding=1),nn.GELU(),
            nn.Conv1d(128,128,3,stride=2,padding=1),nn.GELU(),
            nn.AdaptiveAvgPool1d(1),nn.Flatten())
        self.mlp  = nn.Sequential(nn.Linear(144,256),nn.LayerNorm(256),nn.GELU(),
                                  nn.Linear(256,256),nn.GELU())
        def h(d): return nn.Linear(256,d*2)
        self.hpd=h(DIM_PULSE_DUR); self.hmt=h(DIM_MOD_TYPE); self.hmc=h(DIM_MOD_CONT)
        self.hfi=h(DIM_FILTER);    self.hri=h(DIM_RISE);      self.hfa=h(DIM_FALL)
        self.ham=h(DIM_AMPLITUDE); self.hre=h(DIM_RESIDUALS)
    def forward(self,iq,task):
        h = self.mlp(torch.cat([self.cnn(iq.unsqueeze(1)),self.te(task)],-1))
        def sp(x): m=x.shape[-1]//2; return x[...,:m],x[...,m:]
        return NamedLatent(*sp(self.hpd(h)),*sp(self.hmt(h)),*sp(self.hmc(h)),*sp(self.hfi(h)),
                           *sp(self.hri(h)),*sp(self.hfa(h)),*sp(self.ham(h)),*sp(self.hre(h)))

class IQDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(DIM_TOTAL,256),nn.GELU(),nn.Linear(256,128*4))
        self.dec  = nn.Sequential(
            nn.ConvTranspose1d(128,128,4,2,1),nn.GELU(),
            nn.ConvTranspose1d(128,64, 4,2,1),nn.GELU(),
            nn.ConvTranspose1d(64, 32, 4,2,1),nn.GELU(),
            nn.ConvTranspose1d(32, 16, 4,2,1),nn.GELU(),
            nn.ConvTranspose1d(16,  1, 4,2,1),nn.Tanh())
    def forward(self,z):
        h = self.dec(self.proj(z).view(-1,128,4)).squeeze(1)
        if h.shape[-1] != N_SAMPLES:
            h = F.interpolate(h.unsqueeze(1),N_SAMPLES,mode='linear',align_corners=False).squeeze(1)
        return h

class TCVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc=Encoder(); self.dec_iq=IQDecoder()
        self.dec_pd=nn.Sequential(nn.Linear(DIM_PULSE_DUR,32),nn.GELU(),nn.Linear(32,1),nn.Sigmoid())
        self.dec_am=nn.Sequential(nn.Linear(DIM_AMPLITUDE,32),nn.GELU(),nn.Linear(32,1),nn.Sigmoid())
        self.dec_fi=CVNNDecoder(); self.gumbel_tau=1.0
    def forward(self,iq,task):
        lat=self.enc(iq,task); lat.reparameterize(self.gumbel_tau)
        return dict(latent=lat, recon_iq=self.dec_iq(lat.z_concat()),
                    pred_pd=self.dec_pd(lat.z_pulse_dur).squeeze(-1),
                    pred_am=self.dec_am(lat.z_amplitude).squeeze(-1),
                    pred_fi=self.dec_fi(lat.z_filter))

model = TCVAE()
n_p   = sum(p.numel() for p in model.parameters())
print(f'[OK] {n_p:,} parameters')

# Shape smoke test
iq_t, tk_t, _ = next(iter(tr_dl))
with torch.no_grad():
    out = model(iq_t, tk_t)
assert out['recon_iq'].shape == iq_t.shape,   f"recon shape {out['recon_iq'].shape}"
assert out['pred_fi'].shape  == (len(iq_t),64), f"filter shape {out['pred_fi'].shape}"
print(f'[OK] forward pass shapes correct')

# ─────────────────────────────────────────────────────────────────────────────
# 4. Training
# ─────────────────────────────────────────────────────────────────────────────
print(f'[4/5] Training {N_EPOCHS_TEST} epochs …')
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

def compute_loss(out, iq, labels, task, beta, aux_w):
    lat  = out['latent']
    Lr   = F.mse_loss(out['recon_iq'], iq)
    Lkl  = lat.kl_loss()
    Lpd  = F.mse_loss(out['pred_pd'], labels[:,0])
    Lam  = F.mse_loss(out['pred_am'], labels[:,1])
    Lmt  = F.cross_entropy(lat.mu_mod_type, task)
    with torch.no_grad():
        tf  = torch.zeros(len(iq), 64)
        for i,bw in enumerate(labels[:,4].numpy()):
            h = sinc_filter(32, float(bw)); tf[i,0::2] = torch.from_numpy(h.astype(np.float32))
    Lfi  = F.mse_loss(out['pred_fi'], tf)
    Laux = Lpd + Lam + Lmt + Lfi
    return Lr + beta*Lkl + aux_w*Laux, Lr.item(), Lkl.item(), Lmt.item()

t0  = time.time()
losses_train = []
for ep in range(1, N_EPOCHS_TEST+1):
    model.train()
    model.gumbel_tau = max(0.2, 1.0 - 0.8*ep/N_EPOCHS_TEST)
    beta   = min(0.4, 0.4*ep/15)
    aux_w  = min(1.0, max(0.0, (ep-5)/15))
    ep_loss = 0.0
    for iq_b, tk_b, lab_b in tr_dl:
        opt.zero_grad()
        out   = model(iq_b, tk_b)
        loss, *_ = compute_loss(out, iq_b, lab_b, tk_b, beta, aux_w)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ep_loss += loss.item()
    losses_train.append(ep_loss / len(tr_dl))
    if ep % max(1, N_EPOCHS_TEST//4) == 0:
        print(f'  ep {ep:>3}  loss={losses_train[-1]:.4f}  β={beta:.3f}  α={aux_w:.3f}')

elapsed = time.time() - t0
print(f'[OK] training done in {elapsed:.1f}s')

# Loss should decrease (or at least not increase massively after warmup)
if N_EPOCHS_TEST >= 4:
    first_half = np.mean(losses_train[:N_EPOCHS_TEST//2])
    last_half  = np.mean(losses_train[N_EPOCHS_TEST//2:])
    # Allow some tolerance since we start from random and aux losses kick in mid-training
    assert last_half < first_half * 2.0, \
        f'Loss diverged: first_half={first_half:.4f} last_half={last_half:.4f}'
    print(f'[OK] loss trend: {first_half:.4f} → {last_half:.4f}')

# ─────────────────────────────────────────────────────────────────────────────
# 5. Eval
# ─────────────────────────────────────────────────────────────────────────────
print('[5/5] Evaluation checks …')
model.eval()
recon_mses = []
with torch.no_grad():
    for iq_b, tk_b, lab_b in va_dl:
        out = model(iq_b, tk_b)
        mse = F.mse_loss(out['recon_iq'], iq_b).item()
        recon_mses.append(mse)
        # Check pred shapes
        assert out['pred_pd'].shape == (len(iq_b),)
        assert out['pred_am'].shape == (len(iq_b),)
        assert out['pred_fi'].shape == (len(iq_b), 64)
        # Check pred_amplitude in valid range (sigmoid output)
        assert out['pred_am'].min().item() >= 0.0
        assert out['pred_am'].max().item() <= 1.0

mean_recon = np.mean(recon_mses)
print(f'[OK] val recon MSE = {mean_recon:.5f}')

# Latent checks
iq_b, tk_b, _ = next(iter(va_dl))
with torch.no_grad():
    lat = model.enc(iq_b, tk_b)
    lat.reparameterize(tau=0.2)
    kl  = lat.kl_loss().item()
    z   = lat.z_concat()
assert z.shape == (len(iq_b), DIM_TOTAL), f'z_concat shape {z.shape}'
assert np.isfinite(kl), f'KL is non-finite: {kl}'
print(f'[OK] KL = {kl:.4f}  z_concat shape = {z.shape}')

# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{"="*50}')
print(f'ALL TESTS PASSED  ({time.time()-t0:.1f}s total)')
print(f'{"="*50}')
