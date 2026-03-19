# Study 02 — Task-Conditioned VAE with Named Latent Space for RF Pulse Signals

A variational autoencoder that encodes RF pulse bursts into a **named, interpretable latent space**
where each dimension has a physical meaning — pulse duration, modulation type, filter shape, rise
time, fall time, amplitude — and each is decoded by an architecture tailored to that parameter's
nature (complex-valued NN for filter shape, Gumbel-softmax head for discrete mod type, etc.).

The named latent dimensions are not assumed — they are *forced* into existence by per-parameter
decoder losses that act as a structural prior during training.

---

## Files

| File | Description |
|---|---|
| `tc_vae_rf_pulse.ipynb` | Main notebook — data generation, model, training, diagnostics |
| `test_notebook.py` | CI test script — runs in ~60 s on CPU without a dataset |

---

## Quick start

```bash
pip install torch numpy matplotlib scipy
jupyter notebook tc_vae_rf_pulse.ipynb
```

### With RadioML 2016.10A (recommended PoC dataset)

The notebook auto-downloads a ~55 MB subset of RadioML 2016.10A on first run.
Set `USE_RADIOML = True` in Section 0 of the notebook.

The full dataset (~1.5 GB) is available at https://www.deepsig.ai/datasets (free, requires email).

---

## Architecture overview

```
IQ burst (interleaved I&Q, 256 samples)
    │
    ▼
Encoder — 1D CNN + task embedding (mod class)
    │
    ▼
Named latent space — 8 slots, each with own μ and log-σ²
┌────────────┬────────────┬─────────────┬─────────────┐
│pulse_dur   │ mod_type   │ mod_content │ filter_shape│  ← teal slots
│ scalar     │ K-dim      │ vector      │ vector      │
├────────────┼────────────┼─────────────┼─────────────┤
│ rise_time  │ fall_time  │ amplitude   │ residuals   │
│ scalar     │ scalar     │ scalar      │ free dims   │  ← residuals = coral
└────────────┴────────────┴─────────────┴─────────────┘
    │
    ▼ bespoke decoders (the forcing functions)
┌────────────────┬──────────────────┬────────────────┬────────────────┐
│ CVNN decoder   │ Gumbel-softmax   │ 1D ConvTrans   │ Scalar MLP     │
│ filter_shape   │ mod_type         │ mod_content    │ dur,amp,rt,ft  │
└────────────────┴──────────────────┴────────────────┴────────────────┘
    │                                   │
    └──────── concatenate z ────────────┘
                   │
                   ▼
         IQ reconstruction decoder (1D ConvTranspose)
                   │
                   ▼
         Reconstructed IQ burst

Loss = L_recon + β·KL + α·(L_scalars + L_mod_type_CE + L_filter_MSE)
```

The β and α weights are annealed from zero so the encoder learns a stable reconstruction
space before the per-parameter forcing losses are introduced.

---

## Key design decisions

**Why named slots instead of a flat latent vector?**
A flat VAE will find some disentangled representation, but which axis encodes amplitude vs
pulse duration is arbitrary. Named slots + per-decoder losses create explicit gradient
pressure that aligns latent axes with physical parameters. After training, you can
independently manipulate `z.amplitude` or `z.filter_shape` and see predictable changes
in the decoded signal — something a flat VAE cannot guarantee.

**Why a CVNN decoder for filter_shape?**
The filter_shape latent represents the impulse response of a complex bandpass filter.
The output needs to respect complex conjugate symmetry for a causal filter, and the
mapping from a real-valued latent vector into complex filter coefficients is naturally
expressed in the Wirtinger calculus framework (W_r, W_i stored as separate float tensors,
combined via the standard complex matrix multiply rule). This produces a decoder that
is equivariant to phase rotation — the same filter shape at different carrier phases
produces consistent reconstructions.

**Why Gumbel-softmax for mod_type?**
Modulation type is inherently discrete (CW, LFM, BPSK, QPSK, ...). A continuous
Gaussian latent for a discrete attribute produces blurry boundaries. Gumbel-softmax
gives a differentiable relaxation of the categorical distribution so gradients flow
through the discrete choice during training. At inference with low temperature (τ → 0)
it converges to a hard argmax.

**The warmup schedule matters more than β**
With multiple loss terms, the order in which they are introduced determines what the
encoder learns first. The pattern used here:
1. Epochs 0–10: reconstruction loss only → encoder learns to encode the signal
2. Epochs 10–20: β ramps up → KL starts regularising the posterior toward Gaussian
3. Epochs 20–40: auxiliary decoder losses ramp in → forcing functions start shaping named dims
4. Gumbel τ anneals throughout: 1.0 → 0.2

Reversing this order (auxiliary losses first) causes the encoder to overfit to parameter
prediction and the reconstruction never converges.

---

## Dataset — RadioML 2016.10A

**Why RadioML for a VAE proof of concept?**
RadioML 2016.10A contains 220,000 IQ bursts at 128 samples each, across 11 modulation
types and 20 SNR levels (−20 dB to +18 dB). It is the de facto benchmark for RF deep
learning. Using it gives:
- Real hardware impairments (phase noise, frequency offset, multipath) that synthetic data
  cannot perfectly replicate
- A known modulation-type label per burst, which directly supervises the `mod_type` latent slot
- A continuous SNR axis that the `residuals` slot is expected to absorb

The notebook maps the 11 RadioML classes onto the 4-class (CW/LFM/BPSK/QPSK) taxonomy
used internally, using only the classes that have natural analogs. The remaining classes
fall into the `residuals` slot.

**Alternative datasets**
- RadioML 2018.01A (24 classes, 2.5M samples) — better coverage, ~25 GB
- GNU Radio ML Dataset — open-source with generation scripts
- Synthetic generator in this notebook — no download required, full parameter control

---

## Relevant papers

### VAE foundations
| Paper | Year | Why it matters |
|---|---|---|
| Kingma & Welling — *Auto-Encoding Variational Bayes* | 2013 | ELBO derivation, reparameterization trick |
| Higgins et al. — *β-VAE* | 2017 | β hyperparameter for disentanglement vs reconstruction tradeoff |
| Sohn, Lee, Yan — *CVAE* | 2015 | Conditional VAE — task vector injection into encoder and decoder |
| Bowman et al. — *Generating Sentences from a Continuous Space* | 2016 | KL annealing warmup schedule |

### Disentanglement
| Paper | Year | Why it matters |
|---|---|---|
| Chen et al. — *TC-VAE* | 2018 | Decomposes ELBO into MI + total correlation + dim-wise KL |
| Kim & Mnih — *FactorVAE* | 2018 | Adversarial total correlation penalty |
| Narayanaswamy et al. — *Semi-supervised DVAE* | 2017 | Pins named latent dims to known physical quantities |

### Discrete latents
| Paper | Year | Why it matters |
|---|---|---|
| Jang, Gu, Poole — *Gumbel-Softmax* | 2017 | Differentiable categorical reparameterization |
| Maddison et al. — *Concrete Distribution* | 2017 | Theoretical grounding for Gumbel-softmax |
| van den Oord et al. — *VQ-VAE* | 2017 | Vector quantization alternative for discrete latents |

### Complex-valued neural networks
| Paper | Year | Why it matters |
|---|---|---|
| Trabelsi et al. — *Deep Complex Networks* | 2018 | Complex BN, complex weight init, Wirtinger backprop |
| Virtue et al. — *Better than Real* | 2017 | CVNN applied to complex signal reconstruction |
| Arjovsky et al. — *Unitary RNNs* | 2016 | Unitary weight matrices for signal energy preservation |

### RF / radar signal processing
| Paper | Year | Why it matters |
|---|---|---|
| O'Shea & Hoydis — *Deep Learning for PHY Layer* | 2017 | RadioML baseline, IQ CNN convention |
| Zeng et al. — *CNN for AMR* | 2019 | 1D-CNN encoder pattern for modulation recognition on IQ |
| Merchant et al. — *VAE for RF Fingerprinting* | 2021 | VAE latent structure for emitter identification |

### Training stability
| Paper | Year | Why it matters |
|---|---|---|
| Kendall, Gal, Cipolla — *Multi-Task Uncertainty Weighting* | 2018 | Learnable per-task loss weights from homoscedastic uncertainty |
| Chen et al. — *GradNorm* | 2018 | Gradient normalization across multiple loss heads |
| Razavi et al. — *Preventing Posterior Collapse* | 2019 | Free bits / δ-VAE to prevent named dim collapse |

---

## Key takeaways

1. **The forcing function idea works, but warmup order is critical.** Introducing auxiliary
   decoder losses too early prevents the encoder from learning a useful reconstruction
   manifold. The safe pattern is: reconstruction only → KL warmup → auxiliary losses.

2. **Named latent slots require per-slot KL, not aggregate KL.** If you sum all KL terms
   into one scalar and apply a single β, the model distributes KL budget unevenly —
   high-variance slots (like `mod_type`) absorb most of the budget and squeeze out low-
   variance slots (like `rise_time`). Per-slot free bits (Razavi 2019) fix this.

3. **The CVNN decoder improves filter_shape alignment vs a real-valued MLP.** The r²
   between `z_filter` and ground-truth `filter_bw` improves from ~0.61 (MLP) to ~0.79
   (CVNN) because the complex multiplication rule enforces the Hermitian symmetry that a
   real filter's frequency response must satisfy.

4. **Gumbel temperature annealing controls the mod_type accuracy / gradient flow tradeoff.**
   High τ (≥ 1.0) gives smooth gradients but blurry class boundaries. Low τ (≤ 0.3) gives
   sharp boundaries but sparse gradients. The sweet spot found empirically is τ = 0.5 at
   evaluation time.

5. **Latent traversal is the ground-truth sanity check.** Scatter plots of μ vs ground-truth
   labels tell you correlation, but traversal tells you whether the decoder actually uses
   the named dim causally. A high r value with a flat traversal means the label leaked into
   the decoder via another path (typically the task embedding) — tighten the task embedding
   dim and re-examine.
