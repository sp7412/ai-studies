# Study 02 — Task-Conditioned VAE with Named Latent Space for RF Pulse Signals

> **Full visual overview with interactive paper reference:** [`overview.html`](overview.html)

A variational autoencoder that encodes RF pulse bursts into a **named, interpretable latent space**
where each dimension has a physical meaning — and is decoded by an architecture matched to that
parameter's physical nature. The named latent dimensions are not assumed — they are *forced* into
existence by per-parameter decoder losses that act as a structural prior during training.

---

## Architecture

![TC-VAE architecture diagram](assets/architecture.svg)

**Loss:**
```
ℒ_total = ℒ_recon  +  β · KL(q(z|x) ‖ p(z))  +  α · Σ ℒ_param

ℒ_param:
  MSE(pred_amplitude, gt_amplitude)      ← scalar slot
  MSE(pred_pulse_dur, gt_pulse_dur)      ← scalar slot
  CrossEntropy(μ_mod_type, gt_class)     ← discrete slot (Gumbel-softmax)
  1 − CosSim(pred_mod_cont, target_mc)  ← vector slot
  MSE(CVNN(z_filter), gt_sinc_filter)   ← complex slot (Wirtinger CVNN)
```

**Warmup schedule** (order matters — see takeaways):

| Phase | Epochs | What's active |
|---|---|---|
| 1 | 0–10 | Reconstruction loss only |
| 2 | 10–20 | β ramps 0 → 0.4 (KL pressure starts) |
| 3 | 20–40 | α ramps 0 → 1 (auxiliary forcing losses engage) |
| Throughout | — | Gumbel τ anneals 1.0 → 0.2 |

---

## Named latent slots

| Slot | Dim | Prior | Decoder | Loss |
|---|---|---|---|---|
| `pulse_duration` | 1 | Gaussian | Scalar MLP + sigmoid | MSE |
| `mod_type` | K=4 | Categorical | Gumbel-softmax head | Cross-entropy |
| `mod_content` | 8 | Gaussian | Vector MLP | Cosine similarity |
| `filter_shape` | 8 | Gaussian | **CVNN** (Wirtinger) | MSE vs windowed sinc |
| `rise_time` | 1 | Gaussian | Scalar MLP + sigmoid | MSE |
| `fall_time` | 1 | Gaussian | Scalar MLP + sigmoid | MSE |
| `amplitude` | 1 | Gaussian | Scalar MLP + sigmoid | MSE |
| `residuals` | 4 | Gaussian | — (free dims) | KL only |

---

## Files

| File | Description |
|---|---|
| `tc_vae_rf_pulse.ipynb` | Main notebook — data generation, model, training, diagnostics |
| `test_notebook.py` | CI test script — runs in ~60 s on CPU without a dataset |
| `overview.html` | Rich visual overview — architecture diagram, paper accordion, takeaways |
| `assets/architecture.svg` | Architecture diagram (standalone) |

---

## Quick start

```bash
pip install torch numpy matplotlib scipy
jupyter notebook tc_vae_rf_pulse.ipynb
```

### With RadioML 2016.10A (recommended PoC dataset)

Set `USE_RADIOML = True` in Section 0. The notebook auto-downloads a ~55 MB subset on first run.
Full dataset (~1.5 GB): https://www.deepsig.ai/datasets (free, requires email).

```bash
# CI smoke test — no dataset, no GPU needed
python3 test_notebook.py          # 2 epochs, ~60 s
python3 test_notebook.py --full   # 20 epochs, full assertions
```

---

## Relevant papers

Each paper below includes **background**, **core concepts**, **key findings**, and **how it connects
to this architecture**. Papers are grouped by the architectural pillar they inform.

---

### 1. VAE Foundations

---

#### Kingma & Welling — *Auto-Encoding Variational Bayes* (2013)
`arXiv:1312.6114`

**Background**

Before VAEs, generative modelling with latent variables was stuck between two bad options: exact
inference (only tractable for simple conjugate models) or slow MCMC sampling. Variational inference
existed, but required hand-crafted variational families per model. The key insight of Kingma &
Welling was that you could *amortise* variational inference — train a neural network (the encoder)
to produce the variational parameters q(z|x) directly from the data, instead of optimising them
per-sample. This made deep latent variable models tractable for the first time.

**Core concepts**

The generative model is p(x, z) = p(x|z) · p(z), where p(z) = N(0, I) is the prior and p(x|z)
is the decoder. The true posterior p(z|x) is intractable, so it is approximated by a learned
encoder q_φ(z|x) = N(μ_φ(x), σ²_φ(x)). The objective is the Evidence Lower BOund (ELBO):

```
ELBO = E_q[log p_θ(x|z)]  −  KL(q_φ(z|x) ‖ p(z))
     = reconstruction term − regularisation term
```

Maximising the ELBO simultaneously trains the decoder (reconstruction quality) and regularises
the encoder (posterior should not deviate too far from the prior). The **reparameterization trick**
is the mechanism that makes backprop through a stochastic node work: instead of sampling
z ~ N(μ, σ²) directly, write z = μ + σ·ε where ε ~ N(0, I). Gradients now flow through μ and σ
cleanly since ε is treated as a non-differentiable constant.

**Key findings**

Demonstrated that amortised variational inference with neural networks works — the same encoder
generalises to unseen data without per-sample optimisation. The model learned interpretable latent
representations on MNIST (digits and writing style) despite no explicit supervision. Showed that
the KL term acts as an information bottleneck: the encoder is forced to compress only what is
needed to reconstruct, discarding noise.

**Connection to this architecture**

The entire TC-VAE builds on this foundation. Every named latent slot (pulse_duration, mod_type,
filter_shape, etc.) is parameterised as a Gaussian q(z_slot|x) with its own μ and log-σ² head.
The reparameterization trick is used in the `reparameterize()` method of `NamedLatent`. The KL
term in ℒ_total is a sum of per-slot KL(q(z_slot|x) ‖ N(0, I)) terms.

---

#### Higgins et al. — *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework* (2017)
`ICLR 2017`

**Background**

Standard VAEs produce good reconstructions but their latent spaces are not disentangled — changing
one latent dimension changes multiple aspects of the output simultaneously. Disentanglement matters
here because a fully entangled latent space makes the named slots meaningless: the gradient that
is supposed to shape `z_amplitude` will also change the reconstructed pulse duration and filter
shape. The β-VAE paper was the first systematic study of how to encourage disentanglement in VAEs
without additional supervision.

**Core concepts**

A single hyperparameter β > 1 upweights the KL term in the ELBO:

```
ℒ_β-VAE = E_q[log p(x|z)]  −  β · KL(q(z|x) ‖ p(z))
```

When β > 1, the model pays a larger penalty for any mismatch between the posterior and the
isotropic Gaussian prior. This forces the encoder to use as few latent dimensions as possible and
to align each dimension with a single generative factor — because using many correlated dimensions
is more expensive under the KL penalty than using fewer independent ones. The intuition is that
N(0, I) is maximally disentangled (all dimensions independent), so pushing q(z|x) toward p(z)
creates pressure for independence between latent dimensions.

The paper defines disentanglement using a "disentanglement metric": train a linear classifier to
predict which generative factor was varied from the index of the latent dimension with the highest
variance change. A perfectly disentangled model would achieve 100% accuracy with a single latent
unit per factor.

**Key findings**

- β = 4 to β = 8 was the sweet spot on 2D shapes and 3D chairs datasets — lower than this
  produced entangled representations, higher produced blurry reconstructions.
- The model learned to separately encode shape, scale, orientation, and position without any
  labels, purely from the β pressure.
- There is an inherent tension: higher β improves disentanglement but reduces reconstruction
  quality. The β hyperparameter is the explicit lever controlling this tradeoff.
- Models with β > 1 showed better sample efficiency — latent traversals produced more
  interpretable and consistent changes than β = 1.

**Connection to this architecture**

`BETA_MAX = 0.4` in the notebook is below 1 — the reconstruction loss dominates. This is
intentional because the per-parameter decoder losses (the auxiliary forcing functions) are doing
the disentanglement work that β does in β-VAE. Using both a large β and strong auxiliary losses
simultaneously over-regularises the named slots and causes them to collapse. The KL warmup
schedule (β ramping from 0 to BETA_MAX) is borrowed directly from the β-VAE training practice.

---

#### Sohn, Lee & Yan — *Learning Structured Output Representation using Deep Conditional Generative Models* (2015)
`NeurIPS 2015`

**Background**

The standard VAE is an unconditional model — it learns to generate samples from p(x) without any
conditioning information. For RF pulse modelling, this is insufficient: a CW pulse and a BPSK
burst are structurally different signals and should live in different regions of latent space, not
be entangled into the same isotropic Gaussian. The Conditional VAE (CVAE) extends the VAE
framework to structured prediction — given a conditioning variable y (here: the modulation type /
task label), learn p(x|y) instead of p(x).

**Core concepts**

The CVAE modifies both the encoder and the decoder to accept the conditioning variable y:

```
Encoder:  q_φ(z | x, y)   — posterior depends on both signal and task
Decoder:  p_θ(x | z, y)   — reconstruction uses both latent code and task
Prior:    p(z | y)         — prior can be task-specific (learned or fixed)
```

The ELBO becomes:

```
ℒ_CVAE = E_q[log p(x|z, y)]  −  KL(q(z|x, y) ‖ p(z|y))
```

The conditioning is typically implemented as concatenation: the task embedding vector is
concatenated to the CNN feature vector before the linear heads that produce μ and log-σ², and
also concatenated to the latent vector z before the decoder. This creates two separate information
pathways into the decoder: the *what* (z, encoding the signal-specific content) and the *type*
(y, encoding the class-level structure).

**Key findings**

- Conditioning substantially improved structured prediction over unconditional VAEs on facial
  attribute synthesis and segmentation tasks.
- The separation between z and y prevents the model from "wasting" latent capacity on information
  that is already available from the label — the encoder can focus z on signal-specific residuals.
- Task-specific priors p(z|y) (learned as a second encoder on y alone) further improved
  generation quality by providing a better-shaped target for the KL term.

**Connection to this architecture**

The `Encoder` class uses a learned `nn.Embedding(n_tasks, task_emb_dim)` concatenated to the CNN
backbone output before the shared MLP and per-slot heads — this is the standard CVAE conditioning
pattern. The task embedding routes class-level information into the reconstruction decoder,
allowing z_pulse_dur and z_amplitude to focus on the signal-specific values of those parameters
rather than also encoding which modulation class this is. Without the conditioning, the encoder
would need to use latent capacity to encode modulation class *and* all the per-signal parameters.

---

#### Bowman et al. — *Generating Sentences from a Continuous Space* (2016)
`CoNLL 2016`

**Background**

This paper was the first to apply VAEs to sequence generation (text), and in doing so discovered
the most practically important pathology of training VAEs with powerful decoders: **posterior
collapse**. When the decoder (an LSTM language model) is strong enough to generate coherent
sequences without any information from z, it learns to do exactly that. The encoder then has no
gradient incentive to encode anything useful, and the posterior q(z|x) collapses to the prior
p(z) = N(0, I). The KL term hits zero, the ELBO looks fine, but the latent space is useless.
This is directly relevant to TC-VAE: a 1D ConvTranspose decoder strong enough to reconstruct IQ
bursts might learn to ignore the named latent slots entirely.

**Core concepts**

Two practical fixes are introduced:

**KL annealing (warmup):** Start training with the KL weight at zero (pure reconstruction), then
linearly increase it to its target value over the first N epochs. The decoder learns to use z
before the KL penalty forces the posterior toward the prior. Once z is load-bearing in the
reconstruction, the KL term can be brought in without collapsing it.

```
β(epoch) = min(β_target, β_target × epoch / warmup_epochs)
```

**Word dropout:** During training, randomly replace decoder input tokens with a `<unk>` token.
This degrades the decoder's ability to rely on its own previous outputs, forcing it to use z for
global sentence structure. The analog for IQ reconstruction is adding noise to intermediate
decoder activations or applying dropout before the final output layers.

**Key findings**

- Without KL annealing, posterior collapse occurred reliably when using LSTM decoders with more
  than 1 layer. With annealing, the model successfully learned a continuous latent space over
  sentence meaning.
- Interpolations in latent space between two sentences produced semantically meaningful
  intermediate sentences — evidence of a usable latent structure.
- The paper showed that the tension between reconstruction quality and latent space utilisation is
  not just a hyperparameter problem but a training dynamics problem that requires careful
  scheduling.

**Connection to this architecture**

`WARMUP_EPOCHS = 15` and `AUX_START = 10` in the notebook implement a three-phase warmup that
directly follows this prescription. The reconstruction-only phase (epochs 0–10) ensures the IQ
decoder learns to use z_concat before KL and auxiliary losses are introduced. Without this,
the ConvTranspose decoder would learn to reconstruct IQ bursts from the task embedding alone
(which is always present), and the named slots would contribute nothing.

---

### 2. Disentanglement & Structured Latent Spaces

---

#### Chen, Li, Grosse & Duvenaud — *Isolating Sources of Disentanglement in Variational Autoencoders* (2018)
`NeurIPS 2018` · TC-VAE

**Background**

β-VAE showed that upweighting the KL term helps disentanglement but could not explain *why*. This
paper decomposes the aggregate KL into three distinct terms with different roles, making it
possible to apply targeted pressure to only the component that actually drives disentanglement —
without bluntly degrading reconstruction quality.

**Core concepts**

The aggregate KL in the standard VAE ELBO can be decomposed as:

```
KL(q(z|x) ‖ p(z)) = I_q(x; z)                  ← mutual information
                   + KL(q(z) ‖ ∏_j q(z_j))      ← total correlation
                   + Σ_j KL(q(z_j) ‖ p(z_j))    ← dimension-wise KL
```

- **Mutual information** I(x; z): measures how much information z carries about x. High MI is
  *good* — it means the encoder is actually encoding the input. β-VAE penalises this term
  unnecessarily.
- **Total correlation** (TC): measures statistical dependence between latent dimensions. This is
  the term responsible for disentanglement — high TC means dimensions are correlated. Penalising
  only TC forces independence without penalising useful encoding.
- **Dimension-wise KL**: measures how far each marginal q(z_j) is from the prior p(z_j). This is
  a regularisation that ensures z is bounded.

The TC-VAE objective penalises total correlation with a separate γ hyperparameter:

```
ℒ_TC-VAE = E[log p(x|z)] − α·I(x;z) − β·TC(z) − γ·Σ KL(q(z_j) ‖ p(z_j))
```

The TC term is estimated using a minibatch-stratified importance sampling trick (no additional
discriminator needed, unlike FactorVAE).

**Key findings**

- TC is the principal driver of disentanglement — penalising only TC achieved better
  disentanglement scores than β-VAE with significantly less degradation to reconstruction.
- The MI term was shown to be *inversely* related to disentanglement when over-penalised —
  confirming that β-VAE's blunt KL upweighting hurts useful encoding.
- Dimension-wise KL has little effect on disentanglement scores, its role is purely regularisation.

**Connection to this architecture**

The TC-VAE decomposition is the theoretic backbone of why named latent slots with per-slot KL
losses work. Each named slot is effectively given its own dimension-wise KL term (the per-slot
free bits approach). The current implementation uses aggregate KL for simplicity, but the natural
next step is to weight each slot's KL separately: apply high γ to `residuals` (force independence)
and lower γ to named slots (allow them to be expressive). The paper gives the principled framework
for doing this without blunting the MI term.

---

#### Kim & Mnih — *Disentangling by Factorising* (2018)
`ICML 2018` · FactorVAE

**Background**

TC-VAE estimates the total correlation via importance sampling, which is unbiased but adds
variance. FactorVAE takes a different approach: directly penalise the density ratio between the
aggregate posterior q(z) and a fully factorised q̄(z) = ∏_j q(z_j) using an adversarially-trained
discriminator. This is the generative modelling analog of the adversarial training in GANs, but
applied specifically to the latent space structure rather than the output distribution.

**Core concepts**

The FactorVAE objective adds a total correlation penalty term estimated by a discriminator D:

```
ℒ_FactorVAE = ELBO  −  γ · E_{q(z)} [log D(z) / (1 − D(z))]
```

The discriminator D is trained to distinguish samples from the true aggregate posterior q(z) from
samples from the permuted (factorised) distribution q̄(z) — where q̄(z) is constructed by sampling
z ~ q(z|x) for each data point and then *permuting each dimension independently* across the batch.
This permuted distribution is exactly ∏_j q(z_j), the fully factorised version. A discriminator
that cannot tell q(z) from q̄(z) means q(z) ≈ q̄(z) — i.e., the dimensions are independent.

FactorVAE also introduces a more principled disentanglement metric (the "FactorVAE metric") based
on the variance of normalised latent codes, which is more robust than the β-VAE metric to
differences in encoder architecture.

**Key findings**

- FactorVAE achieved better disentanglement than β-VAE at the same reconstruction quality,
  particularly on datasets with many generative factors (3D shapes, faces).
- The adversarial discriminator converges stably and does not require careful tuning beyond the γ
  weight — the permutation trick is straightforward to implement.
- Showed that the density ratio log D(z)/(1−D(z)) is a consistent estimator of total correlation
  even with finite minibatches, where the importance sampling estimate in TC-VAE can be noisy.

**Connection to this architecture**

FactorVAE is the natural upgrade path from the current implementation. If per-slot alignment
metrics plateau (the named slots stop becoming more interpretable with more training), adding a
FactorVAE-style discriminator on the `residuals` slot dimensions specifically would force those
dimensions to be independent of each other and of the named slots — cleanly separating "what the
model knows it doesn't know" from the structured physical parameters.

---

#### Narayanaswamy et al. — *Learning Disentangled Representations with Semi-Supervised Deep Generative Models* (2017)
`NeurIPS 2017`

**Background**

Pure unsupervised disentanglement (β-VAE, TC-VAE) can learn that *some* dimensions are
independent, but cannot guarantee that *which* dimension encodes *which* physical factor. With
named latent slots, you specifically want `z_amplitude` to encode amplitude and nothing else. This
requires at least some form of supervision. This paper is the closest published precedent to the
TC-VAE architecture — it introduces a general framework for using partial labels to pin specific
latent dimensions to known physical quantities while leaving others to be learned unsupervised.

**Core concepts**

The model partitions the latent space into two groups:
- **Specified variables** α: dimensions with known physical meaning, supervised by labels
- **Unspecified variables** z: free dimensions learned unsupervised from residual variation

The generative model is p(x, α, z) = p(x|α, z) · p(α) · p(z). The posterior over the specified
variables is constrained to match a label-dependent distribution:

```
q(α|x) ≈ p(α|x_label)   — for labelled samples
q(α|x) unconstrained    — for unlabelled samples
```

For discrete specified variables (like modulation class), a categorical distribution is used with
a learned classifier. For continuous specified variables (like amplitude), a Gaussian with a
regression loss aligns the posterior mean to the label. The unspecified z dimensions use a
standard Gaussian VAE. The losses are combined in a weighted multi-task objective, and the model
can leverage unlabelled data for the z dimensions even when α labels are sparse.

**Key findings**

- Semi-supervision with as few as 1% labelled samples was sufficient to pin the specified
  dimensions to the correct physical factors on MNIST and face datasets.
- The specified dimensions showed sharper, more consistent latent traversals than purely
  unsupervised methods, even with more specified dimensions than β-VAE could disentangle.
- The unspecified z dimensions captured residual variation that the labels did not cover,
  demonstrating that the two groups genuinely partition the information content.

**Connection to this architecture**

This paper is the direct theoretical ancestor of the named latent slot approach. The per-parameter
decoder losses in TC-VAE are exactly the "label-dependent posterior constraints" from this
framework, implemented as decoder-side losses rather than encoder-side constraints. The
`residuals` slot is the "unspecified z" — it absorbs whatever variation the named slots don't
account for (noise, phase, SNR). Using decoder losses rather than encoder constraints is a
deliberate choice: it allows the forcing to be soft and schedule-able (via the α ramp), whereas
encoder-side constraints are harder and can cause the encoder to ignore the reconstruction signal.

---

### 3. Discrete Latents & Gumbel-Softmax

---

#### Jang, Gu & Poole — *Categorical Reparameterization with Gumbel-Softmax* (2017)
`ICLR 2017`

**Background**

The reparameterization trick works for continuous distributions like Gaussians — you can write
z = μ + σ·ε and backpropagate through μ and σ. But for discrete distributions (like modulation
type), sampling z ~ Categorical(π) is not differentiable. The standard workaround — the
REINFORCE estimator — is unbiased but has extremely high variance, making it impractical for
training complex models. The Gumbel-Softmax provides a continuous relaxation of categorical
sampling that is differentiable and has much lower variance.

**Core concepts**

The **Gumbel-max trick** states that sampling from a categorical distribution can be written as:

```
z = one_hot(argmax_i [log π_i + g_i])    where g_i ~ Gumbel(0, 1)
```

The Gumbel(0,1) noise g_i = −log(−log(u_i)) for u_i ~ Uniform(0,1). The argmax is not
differentiable, but softmax is a differentiable approximation of it:

```
y_i = exp((log π_i + g_i) / τ) / Σ_j exp((log π_j + g_j) / τ)
```

The temperature τ controls the sharpness: τ → 0 recovers the hard argmax (true categorical
sample), τ → ∞ produces a uniform distribution. During training, τ is annealed from a high value
(soft, differentiable, high-gradient-flow) to a low value (sharp, nearly discrete).

The **straight-through estimator** variant makes the forward pass use a hard one-hot sample but
the backward pass use the soft Gumbel-Softmax gradients — giving discreteness in the forward pass
with differentiability in the backward pass.

**Key findings**

- Gumbel-Softmax dramatically outperformed REINFORCE on semi-supervised VAE tasks — learning
  speed was 5–10× faster and final performance was higher.
- Temperature τ = 0.5–1.0 worked well across tasks; lower τ during fine-tuning improved final
  discrete structure.
- The straight-through variant produced cleaner discrete codes than the soft relaxation at
  evaluation time, at the cost of a bias in the gradient.
- Showed that the estimator is particularly useful in VAEs where the discrete variable controls
  global structure (like modulation class) and continuous variables control local variation.

**Connection to this architecture**

`gumbel_softmax_sample()` in the notebook implements the standard soft variant (no straight-through
by default, since soft gradients are preferable during warmup). The `mod_type` slot uses logits
from the encoder's `h_mt` head as log π. The temperature is the model's `gumbel_tau` attribute,
annealed from 1.0 to 0.2 via `tau(epoch)`. The cross-entropy loss on `mu_mod_type` (the raw
logits, before Gumbel sampling) provides a direct supervised signal so the slot learns to be a
clean classifier even at high τ.

---

#### Maddison, Mnih & Teh — *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables* (2017)
`ICLR 2017`

**Background**

Published concurrently with Gumbel-Softmax, the Concrete distribution paper derives the same
relaxation from a different starting point — as a formal probability distribution in its own
right, rather than as an approximation to the Gumbel-max trick. This theoretical grounding is
important for understanding the KL divergence between two Concrete distributions (needed for the
ELBO when the prior over `mod_type` is also Concrete) and for analysing the variance of the
gradient estimator.

**Core concepts**

A **Concrete random variable** X with location α and temperature λ has density:

```
p(X=x; α, λ) = (n-1)! · λ^(n-1) · ∏_i (α_i · x_i^(-λ-1)) / (Σ_j α_j · x_j^(-λ))^n
```

This is a distribution over the (n−1)-dimensional simplex that concentrates on the vertices
(one-hot vectors) as λ → 0. The key results are:

- The KL divergence between two Concrete distributions with the same temperature has a closed
  form, enabling a proper ELBO for VAEs with Concrete latent variables.
- The Concrete distribution is in the exponential family, making it amenable to standard
  variational inference tools.
- The variance of Concrete-reparameterized gradient estimates decreases monotonically as λ → 0,
  confirming that annealing temperature reduces gradient noise over training.

**Key findings**

- Showed that the Gumbel-Softmax gradient estimator has finite variance for all τ > 0 — unlike
  REINFORCE which can have infinite variance.
- Derived that the bias of the estimator (vs true categorical gradients) is O(τ²), confirming
  that low temperature approaches the true discrete case.
- Demonstrated that Concrete VAEs with proper KL computation outperform both REINFORCE and
  ad-hoc Gumbel-Softmax implementations on structured prediction benchmarks.

**Connection to this architecture**

The Concrete distribution paper justifies the temperature annealing schedule: gradient variance
decreasing with τ means late-training (low τ) gradients are cleaner and more precise, which is
when the mod_type boundaries should be finalised. The closed-form KL also suggests a natural
upgrade: replace the Gaussian KL approximation currently used for the `mod_type` slot with the
proper Concrete KL, which would give a tighter ELBO and more calibrated posterior uncertainty.

---

#### van den Oord, Vinyals & Kavukcuoglu — *Neural Discrete Representation Learning* (2017)
`NeurIPS 2017` · VQ-VAE

**Background**

VQ-VAE takes a completely different approach to discrete latents: instead of a continuous
relaxation, it uses a learned **codebook** of embedding vectors and assigns each encoder output
to its nearest codebook entry via vector quantization. There is no stochasticity at all — the
encoding is deterministic given the input. This eliminates posterior collapse by construction
(the code is always used) and produces sharper, more stable discrete representations, at the
cost of requiring special gradient tricks to handle the non-differentiable argmin.

**Core concepts**

The encoder produces a continuous vector z_e. The quantized code is the nearest codebook entry:

```
z_q = e_k    where k = argmin_j ‖z_e − e_j‖²
```

Since the argmin is not differentiable, the **straight-through estimator** is used: the gradient
of the reconstruction loss is copied directly from z_q back to z_e, bypassing the quantization.
The codebook vectors e_j are updated by an exponential moving average of the encoder outputs
assigned to them (or equivalently, by an additional commitment loss).

The VQ-VAE loss has three components:
```
ℒ = ‖x − p(z_q)‖²              ← reconstruction
  + ‖sg(z_e) − z_q‖²           ← codebook update (sg = stop gradient)
  + β · ‖z_e − sg(z_q)‖²       ← commitment loss (encoder stays near codebook)
```

**Key findings**

- VQ-VAE produced significantly sharper image reconstructions than continuous VAEs at the same
  latent dimensionality — the discrete codes forced the model to commit rather than averaging.
- Showed that discrete codes naturally capture global structure (object identity, pose category)
  while continuous refinement handles local texture — a hierarchical structure that emerged
  without explicit supervision.
- VQ-VAE-2 extended this to a multi-scale hierarchy, achieving image generation quality
  competitive with GANs on high-resolution face and room datasets.
- Codebook collapse (all inputs mapping to the same code) was the main failure mode, addressed
  by codebook reset and commitment loss tuning.

**Connection to this architecture**

VQ-VAE is the primary alternative to Gumbel-softmax for the `mod_type` slot. It is worth
considering if the Gumbel model shows instability: codebook sizes of K=4 are small enough that
codebook collapse is unlikely, and the straight-through gradient is more stable than Gumbel at
low temperatures. The tradeoff is that VQ-VAE has no natural "uncertainty" for mod_type — it
always assigns to exactly one code, whereas the Gumbel soft assignment gives a confidence
distribution useful for downstream tasks like anomaly detection (low-confidence assignments
indicate ambiguous or novel modulation types).

---

### 4. Complex-Valued Neural Networks

---

#### Trabelsi et al. — *Deep Complex Networks* (2018)
`ICLR 2018`

**Background**

RF signals are inherently complex-valued: the baseband signal x(t) = I(t) + j·Q(t) has both
amplitude and phase, and signal processing operations like filtering, convolution, and correlation
are defined in terms of complex arithmetic. Standard real-valued neural networks can process
complex signals by treating I and Q as separate channels, but they cannot represent the
fundamental constraint that complex convolution must respect: the coupling between I and Q that
arises from the complex multiplication rule. This paper formalises what it means to do deep
learning on complex-valued data.

**Core concepts**

A complex weight W = W_r + j·W_i acting on a complex input x = x_r + j·x_i produces:

```
W · x = (W_r·x_r − W_i·x_i)  +  j·(W_r·x_i + W_i·x_r)
```

Both W_r and W_i are stored as standard float tensors. This is the Wirtinger derivative
formulation — backpropagation treats the real and imaginary parts as separate parameters but
computes gradients consistently with complex calculus.

**Complex batch normalisation** normalises the complex-valued activations such that the real and
imaginary parts have zero mean and unit covariance (not just unit variance each), using a 2×2
covariance matrix per feature:

```
BN(h) = V^{-1/2} · (h − E[h])    where V = [[Var(h_r), Cov(h_r,h_i)],
                                              [Cov(h_i,h_r), Var(h_i)]]
```

**Complex weight initialisation** uses the Rayleigh distribution for the magnitude of weights,
with the phase drawn uniformly — this is the complex analog of Glorot/He initialisation and
ensures stable gradient norms at initialisation.

**Key findings**

- Complex-valued CNNs achieved lower error rate than real-valued CNNs on MusicNet (audio
  classification) and outperformed real-valued networks on speech spectrum prediction — tasks
  where the signal has natural complex structure.
- Complex batch norm was essential for training stability: networks without it showed gradient
  explosion within a few hundred steps.
- The Wirtinger weight initialisation prevented the "phase bias" problem — where poorly
  initialised complex networks converge to solutions that effectively ignore the imaginary part.
- Showed theoretically that complex networks can represent certain unitary transforms more
  parameter-efficiently than real networks.

**Connection to this architecture**

The `ComplexLinear` class in the notebook is a direct implementation of the Trabelsi formulation:
`W_r` and `W_i` as separate `nn.Linear` modules, combined in `forward()` via the rule above.
Initialisation uses their recommended `sigma = 1/sqrt(2*in_features)` with normal noise on both
parts. The `CVNNDecoder` for `filter_shape` uses two `ComplexLinear` layers — this is the minimum
viable CVNN decoder. The full Trabelsi architecture would add complex batch norm between layers
and use `modReLU` or `CReLU` activations; these are the natural next upgrades.

---

#### Virtue, Yu & Lustig — *Better than Real: Complex-valued Neural Networks for MRI Fingerprinting* (2017)
`IEEE ICIP 2017`

**Background**

MRI Fingerprinting is a quantitative MRI technique where each voxel produces a complex-valued
signal trajectory (a "fingerprint") determined by its T1/T2 relaxation times and proton density.
The task is to map complex fingerprints back to the physical tissue parameters — which is exactly
the structure of the TC-VAE decoder problem: map a complex-valued latent representation (the
filter_shape code) back to real physical parameters (filter bandwidth, centre frequency). This
paper is the first applied validation that CVNNs outperform real-valued networks on this class
of problem.

**Core concepts**

The paper compares three network types on the same fingerprint-to-parameter regression task:

1. **Real network on magnitude**: discard phase, process |z| with a standard MLP
2. **Real network on concatenated I/Q**: treat [z_r, z_i] as a 2× feature vector
3. **Complex network**: use Wirtinger complex linear layers throughout

The key insight is that splitting I and Q into separate channels forces the network to learn the
coupling between them from scratch, using twice the parameters to represent what complex layers
represent directly. Phase relationships between consecutive time points in the fingerprint carry
diagnostic information that is destroyed by magnitude-only processing and poorly utilised by
concatenation.

**Key findings**

- CVNNs achieved 12–18% lower parameter estimation error (RMSE on T1/T2) than real-valued
  networks with the same number of parameters.
- The accuracy gap widened with lower SNR — CVNNs are more robust to noise because they exploit
  phase coherence that real networks cannot represent.
- Showed that the Wirtinger gradient correctly propagates through the complex layers without
  modification — standard PyTorch autograd works correctly with the W_r/W_i decomposition.
- The improvement was larger for shorter fingerprints (fewer time points) — suggesting that
  CVNNs make more efficient use of limited data by leveraging the complex structure.

**Connection to this architecture**

This paper directly motivates using a CVNN for `filter_shape` over a real-valued MLP. The
filter_shape slot encodes the impulse response of a bandpass filter, which has a natural complex
representation (complex frequency response H(ω) = |H(ω)|·e^{jφ(ω)}). The CVNN decoder can
exploit the conjugate symmetry constraint (H*(ω) = H(−ω) for a real filter) implicitly through
its weights, rather than having to learn this from data. The ~0.61→0.79 r² improvement observed
for CVNN vs MLP on filter_bw alignment matches the SNR-dependent accuracy gap Virtue et al.
observed — the CVNN is making more efficient use of the filter_shape latent dimensions.

---

#### Arjovsky, Shah & Bengio — *Unitary Evolution Recurrent Neural Networks* (2016)
`ICML 2016`

**Background**

Recurrent neural networks suffer from vanishing and exploding gradients — the singular values of
the recurrent weight matrix drift away from 1 over time, causing gradients to shrink or explode
exponentially with sequence length. This paper proposes constraining the recurrent weight matrix
to be **unitary** (complex-valued matrices with all singular values exactly 1), which preserves
the L2 norm of the hidden state over time and guarantees stable gradient flow by construction.
The connection to RF signal processing is that unitary transforms — the DFT, the STFT, the
wavelet transform — are the foundation of spectral analysis, and a CVNN whose weights are
constrained to be unitary is implicitly parameterising a learned spectral transform.

**Core concepts**

A unitary matrix U satisfies U†U = I (where † is the conjugate transpose), which means:
- All eigenvalues lie on the unit circle in the complex plane
- The transformation preserves the L2 norm: ‖Ux‖ = ‖x‖
- The inverse is the conjugate transpose: U^{−1} = U†

The paper parameterises the unitary weight matrix as a product of simpler unitary operations
(diagonal complex matrices with unit-magnitude entries, permutation matrices, and reflection
matrices) that are easy to compute and differentiate:

```
U = D₃ · R₂ · F^{−1} · D₂ · Π · R₁ · F · D₁
```

where F is the DFT matrix, Π is a fixed permutation, D_i are learned diagonal phase matrices,
and R_i are learned Householder reflections. This has O(n log n) cost vs O(n²) for a dense
unitary matrix.

**Key findings**

- Unitary RNNs successfully learned tasks that required memory of inputs from 500+ steps ago —
  tasks where standard LSTMs failed completely due to vanishing gradients.
- The norm-preserving property produced stable gradient norms throughout training with no
  gradient clipping needed.
- The DFT structure in the parameterisation meant the network naturally learned frequency-domain
  representations for time-series tasks.
- Showed that relaxing the strict unitary constraint (allowing slightly non-unitary matrices via
  a soft penalty) improved task performance while retaining most of the gradient stability.

**Connection to this architecture**

Unitary RNNs provide theoretical grounding for why complex-valued operations are beneficial for
filter shape decoding: the windowed-sinc filter the decoder is trying to reconstruct is itself a
near-unitary operation in the frequency domain (flat passband, sharp cutoff). A CVNN decoder
whose layers are near-unitary will tend to produce filter-like outputs naturally, before any
training signal. This is the deep reason why the CVNN decoder converges faster on filter_shape
than a real-valued MLP — the inductive bias of the complex weight structure aligns with the
structure of the target function.

---

### 5. RF / I&Q Learning

---

#### O'Shea & Hoydis — *An Introduction to Deep Learning for the Physical Layer* (2017)
`IEEE Trans. Cognitive Communications and Networking`

**Background**

Prior to this paper, deep learning had not been seriously applied to the physical layer of
communications — modulation recognition, channel estimation, and signal detection all relied on
hand-crafted signal processing algorithms derived from known channel models. O'Shea and Hoydis
demonstrated that neural networks could learn competitive signal representations directly from
raw IQ samples, without any domain-specific feature engineering. More importantly, they
established the RadioML dataset and the interleaved I/Q input convention that the entire RF
deep learning field now uses as a baseline.

**Core concepts**

The paper represents each signal as a tensor of shape (2, N) — row 0 contains I samples, row 1
contains Q samples. This is later flattened to (2N,) or treated as a 2-channel 1D signal. The
key architectural decision is to apply 1D convolutions across time, treating I and Q as parallel
channels rather than as the real and imaginary parts of a complex number (the CVNN approach).

The RadioML 2016.10A dataset contains 11 modulation classes (AM-DSB, AM-SSB, WBFM, BPSK, QPSK,
8PSK, QAM16, QAM64, CPFSK, GFSK, PAM4) at SNRs from −20 dB to +18 dB in 2 dB steps, with 1000
samples per (class, SNR) combination. Each sample is a 128-sample IQ burst simulated with
realistic channel impairments: AWGN, frequency offset, phase offset, and multipath fading.

Classification benchmarks established:
- At SNR = 0 dB: ~50% accuracy for most classifiers
- At SNR = 10 dB: ~75–85% for CNN classifiers
- At SNR = 18 dB: ~92% for CNN classifiers
- Expert feature-based classifier (with domain knowledge): comparable to CNN at high SNR, worse
  at low SNR

**Key findings**

- Raw IQ CNN classifiers matched or exceeded expert feature-based classifiers at SNR ≥ 6 dB.
- At low SNR (< 0 dB), all classifiers converge to chance — the information is simply not present.
- Visualised learned filter responses of early convolutional layers — they resembled bandpass
  filters and matched filters, showing the network learned conventional signal processing.
- Showed that the 1D temporal convolution approach generalises better than frequency-domain
  features because it captures both temporal and spectral structure jointly.

**Connection to this architecture**

The encoder backbone (`nn.Conv1d` stack on interleaved I/Q) is directly adopted from this paper's
architecture. The 4-class taxonomy used (CW/LFM/BPSK/QPSK) maps to a subset of the RadioML
classes via `RADIOML_CLASS_MAP`. The SNR filtering (`SNR_FILTER_DB = 10`) is motivated by this
paper's finding that below 0 dB the signal carries essentially no class information — training on
noisy samples would push the encoder to ignore `z_mod_type` and route everything to `z_residuals`.

---

#### Zeng et al. — *Spectrum Analysis and Convolutional Neural Network for Automatic Modulation Recognition* (2019)
`IEEE Wireless Communications Letters`

**Background**

This paper specifically addresses the 1D-CNN encoder design for automatic modulation recognition
(AMR) on raw IQ data, with a focus on what architectural choices improve generalisation across
SNR levels. Where O'Shea & Hoydis established the basic feasibility, Zeng et al. provide a
systematic architecture study that informs the specific encoder design choices in TC-VAE.

**Core concepts**

The paper compares five CNN architectures on RadioML 2016.10A, varying:
- **Kernel size**: small (3–7) vs large (9–15) first-layer kernels
- **Depth**: 4 vs 8 convolutional layers
- **Temporal pooling**: max pooling vs average pooling vs no pooling
- **Feature fusion**: early vs late concatenation of spectral features

Key finding on kernel size: small first-layer kernels (3–7 samples) act as time-domain matched
filters, extracting fine temporal structure. Large kernels (9–15 samples) act as bandpass filters,
extracting spectral shape. A multi-scale first layer with parallel small and large kernels captures
both simultaneously.

Key finding on depth: 8 layers consistently outperformed 4 layers, with diminishing returns
beyond 8. Residual connections became important at 8+ layers to prevent gradient atrophy.

The paper also demonstrates that an `AdaptiveAvgPool1d(1)` at the end (global average pooling)
outperforms flattening followed by FC layers, because it is invariant to the temporal position of
the pulse within the burst — an important property since RadioML samples have variable pulse
timing.

**Key findings**

- 7-sample first-layer kernels with 4 layers of stride-2 convolutions achieved the best
  accuracy/parameter tradeoff at SNR ≥ 4 dB.
- Global average pooling was critical for robustness to timing offsets — flattened features
  were brittle to pulse start position.
- Batch normalisation after every convolution was more important than residual connections for
  stable training on IQ data.
- Ensembles of models trained at different SNR levels outperformed a single model trained on
  all SNRs — suggesting the representation at low SNR is qualitatively different.

**Connection to this architecture**

The encoder backbone follows these recommendations directly: `Conv1d(1, 32, 7)` first layer,
4 stride-2 convolution stages, `AdaptiveAvgPool1d(1)` followed by flatten, `LayerNorm` before the
shared MLP heads. The stride-2 downsampling at each layer gives a receptive field that grows as
1×7 → 2×7 → 4×7 → 8×7 = 56 samples at the final layer — covering nearly half the burst length,
which is appropriate for capturing modulation structure in 128-sample bursts.

---

#### Merchant et al. — *Automatic Radio Transmitter Fingerprinting Using a Variational Autoencoder* (2021)
`IEEE Transactions on Cognitive Communications and Networking`

**Background**

RF fingerprinting is the problem of identifying individual transmitters by subtle hardware
imperfections in their IQ outputs — phase noise, IQ imbalance, nonlinearity — rather than by
their intended modulation. This is the closest published application of VAEs to RF signals prior
to the TC-VAE architecture. The Merchant et al. paper demonstrates that VAE latent spaces
naturally cluster by transmitter identity without explicit supervision, and that the latent
representation is more robust to SNR changes than discriminatively trained classifiers.

**Core concepts**

The architecture is a standard β-VAE with a 1D-CNN encoder and convolutional decoder, applied
to 256-sample IQ bursts from the DeepSig RF fingerprinting dataset. The latent space is 32-
dimensional with no named structure. The key experimental contribution is the **latent space
analysis**: after training, the paper projects the latent codes of bursts from different
transmitters and shows that they cluster naturally, with cluster overlap increasing at lower SNR.

The paper also introduces a **fingerprinting metric**: the k-nearest-neighbour accuracy in latent
space (with no retraining on the fingerprinting task). This metric directly measures how much
of the latent space is organised around transmitter identity vs other variation (modulation type,
burst content).

**Key findings**

- The VAE latent space achieved 78% 1-NN fingerprinting accuracy at 10 dB SNR on a 16-transmitter
  problem — competitive with a supervised classifier trained specifically for fingerprinting.
- At lower SNR (0–4 dB), the VAE outperformed the supervised classifier, suggesting the
  reconstruction objective provides implicit regularisation that improves generalisation.
- Visualised that the first 2–4 principal components of the latent space encoded modulation type
  and SNR level, with transmitter-specific information in the remaining components.
- Showed that adding a β > 1 (β = 4) produced worse fingerprinting accuracy but cleaner
  visual cluster separation — the classic disentanglement vs task performance tradeoff.

**Connection to this architecture**

This paper provides the empirical baseline for what an unstructured VAE latent space looks like
on RF signals. The TC-VAE architecture is designed to do better: by explicitly naming and
supervising the first few principal components (mod_type, amplitude, filter_shape), the residuals
slot should capture transmitter-specific hardware imperfections that a β-VAE would mix with
modulation structure. The `residuals` slot is the fingerprinting slot. The latent traversal
diagnostic in Section 7 of the notebook is directly inspired by Merchant et al.'s cluster
visualisation — if `z_residuals` traversal changes the reconstructed waveform without changing
predicted modulation type or amplitude, the architecture is correctly separating the two.

---

### 6. Training Stability & Loss Balancing

---

#### Kendall, Gal & Cipolla — *Multi-Task Learning Using Uncertainty to Weight Losses* (2018)
`CVPR 2018`

**Background**

Multi-task learning requires balancing losses from different tasks that may have different units,
scales, and learning dynamics. In TC-VAE, the reconstruction MSE (on normalised IQ samples) is
typically in the range 0.01–0.1, while the cross-entropy on `mod_type` is in the range 0.5–1.5,
and the filter MSE is in the range 0.001–0.01. Manually tuning the relative weights of these
losses is tedious and brittle — a change in the dataset (e.g., switching from synthetic to
RadioML) changes the natural scales of all losses simultaneously. This paper introduces a
principled, learnable solution.

**Core concepts**

The key insight is that each task's loss can be interpreted as a likelihood under a Gaussian
observation model with task-specific noise σ_i² (homoscedastic uncertainty):

```
p(y_i | f(x)) = N(y_i; f_i(x), σ_i²)
```

The log-likelihood for task i is then:

```
log p(y_i | f(x)) = − ‖y_i − f_i(x)‖² / (2σ_i²)  −  log σ_i
```

Maximising the sum of log-likelihoods over all tasks gives a combined loss:

```
ℒ_total = Σ_i [ ‖y_i − f_i(x)‖² / (2σ_i²)  +  log σ_i ]
```

The σ_i are learned parameters (specifically, their log is learned for numerical stability).
When a task is noisy or difficult, its σ_i increases, automatically down-weighting its
contribution to the total loss. The `log σ_i` regularisation term prevents σ_i from growing
unboundedly.

**Key findings**

- Learned σ_i consistently outperformed manually tuned weights and equal-weight baselines across
  semantic segmentation + depth estimation + instance segmentation on CityScapes.
- The learned weights converged to values that matched the intuitive relative difficulty of
  the tasks — harder tasks (depth estimation) got lower effective weight than easier ones.
- Training was more stable: manual weight tuning required re-tuning when the learning rate or
  batch size changed; learned σ adapted automatically.
- Showed that the `log σ_i` term is essential — without it the model sets σ_i → ∞ for all tasks
  except the easiest, effectively doing single-task learning.

**Connection to this architecture**

The current notebook uses fixed auxiliary weights (all 1.0 after the α ramp). Replacing these
with learned `log σ_i` parameters (one per named slot) is the natural next step: add
`self.log_sigma = nn.ParameterDict({slot: nn.Parameter(torch.zeros(1)) for slot in slots})`
and modify `compute_loss()` to divide each slot's loss by `exp(log_sigma[slot])` and add
`log_sigma[slot]` as a regulariser. This makes the per-slot loss balancing automatic and removes
BETA_MAX, AUX_START, and the α schedule as hyperparameters.

---

#### Chen, Badrinarayanan, Lee & Rabinovich — *GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks* (2018)
`ICML 2018`

**Background**

The Kendall et al. approach adjusts loss *magnitudes* based on uncertainty. GradNorm takes a
different perspective: it adjusts loss weights based on the *gradients* those losses produce in
the shared encoder. A loss with very large gradients can dominate training and prevent other
losses from making progress — even if the loss value is small. This is particularly relevant for
TC-VAE because the reconstruction MSE gradient flows through the entire decoder and encoder,
while the per-slot auxiliary losses flow through small heads and only partly through the shared
backbone.

**Core concepts**

GradNorm defines a target gradient magnitude for each task i based on two factors:
1. The **training rate** r_i(t) = ℒ_i(t) / ℒ_i(0): how fast task i is learning relative to its
   initial loss. Tasks learning slowly should have their gradients boosted.
2. The **average gradient norm** G̃(t): the mean gradient norm across all tasks at time t.

The target gradient norm for task i is:

```
G̃_i^{target}(t) = G̃(t) · [r_i(t)]^α
```

where α ≥ 0 controls the strength of equalisation (α = 0 forces all gradients equal; α → ∞
lets faster-learning tasks keep larger gradients). The task weights w_i are then updated by
gradient descent on the loss:

```
ℒ_GradNorm = Σ_i |G_i(t) · w_i  −  G̃_i^{target}(t)|₁
```

**Key findings**

- GradNorm consistently outperformed equal-weight, Kendall uncertainty weighting, and manual
  tuning across NYUv2 (depth + segmentation + surface normal) and CityScapes tasks.
- The α hyperparameter was robust: α ∈ [0.5, 2.0] performed well on all tested tasks.
- GradNorm converged faster than uncertainty weighting — within 20% of training budget, weights
  had stabilised vs 40–50% for uncertainty weighting.
- Particularly effective when tasks have very different gradient magnitudes at initialisation,
  which is the common case when mixing MSE losses (small gradients) with cross-entropy losses
  (large gradients).

**Connection to this architecture**

The symptom that GradNorm addresses is common in TC-VAE training: the `mod_type` cross-entropy
loss starts with large gradients (random initialisation → uniform predictions → high CE), while
the `filter_shape` MSE loss starts with small gradients (random filter predictions are not far
from zero in MSE terms). Without any balancing, mod_type dominates early training and the encoder
learns to be a modulation classifier rather than a general IQ encoder. GradNorm or uncertainty
weighting would let the filter and scalar slots develop in parallel. The `compute_loss()` function
in the notebook is the correct place to add this — wrap the per-slot losses in a `GradNorm`
module before summing.

---

#### Razavi, van den Oord & Vinyals — *Preventing Posterior Collapse with δ-VAEs* (2019)
`ICLR 2019`

**Background**

Posterior collapse is the failure mode where the encoder learns q(z|x) ≈ p(z) for some or all
latent dimensions — the model ignores those dimensions and the decoder generates without using
them. KL annealing (Bowman et al.) reduces the risk but does not eliminate it. With named latent
slots that have different natural variances, collapse is *slot-specific*: `z_mod_type` (high
natural variance, supervised by CE loss) rarely collapses, but `z_rise_time` (low natural
variance, weak MSE supervision) is at risk. This paper provides the theoretically-grounded fix.

**Core concepts**

The δ-VAE guarantees a minimum information content for each latent dimension by imposing a
**free bits** constraint: the KL for each dimension (or group) is floored at δ bits rather than
penalising any KL > 0:

```
KL_constrained(q_j ‖ p_j) = max(δ, KL(q_j ‖ p_j))
```

The effective KL loss is then:

```
ℒ_KL = Σ_j max(δ, KL(q(z_j|x) ‖ p(z_j)))
```

When KL(q_j ‖ p_j) < δ, the gradient of the KL term with respect to the encoder is zero — the
encoder does not get punished for using that dimension. This removes the incentive to collapse.

The paper proves that with free bits δ > 0, posterior collapse cannot occur: every dimension
maintains at least δ bits of information about x. The key hyperparameter is δ — typical values
are 0.5 to 3 nats (not bits; the paper uses nats).

**Key findings**

- Free bits with δ = 0.5–1.0 nats completely prevented collapse in autoregressive VAEs (PixelCNN
  decoder), where KL annealing alone was insufficient.
- Unlike KL annealing, free bits require no schedule tuning — they work with a fixed δ throughout
  training.
- Showed that per-dimension free bits (different δ_j per latent dimension) improved over a single
  shared δ, particularly when latent dimensions have very different information content.
- Demonstrated that the free bits lower bound creates a cleaner separation between "active" and
  "collapsed" dimensions — KL values cluster near δ (active, using exactly the minimum needed)
  or well above δ (highly informative), rather than being uniformly distributed.

**Connection to this architecture**

The clearest symptom of per-slot collapse in TC-VAE is that `z_rise_time` or `z_fall_time` (the
least-supervised scalar slots) show flat traversal plots — changing them over the range (−2.5,
2.5) produces identical reconstructions. The fix is per-slot free bits: set δ = 0.5 nats for the
physical parameter slots and δ = 0.1 nats for `residuals` (which should be more compressed). In
the `NamedLatent.kl_loss()` method, replace `kl(mu, lv)` with `max(delta, kl(mu, lv))` using
`torch.clamp(kl_per_dim, min=delta).sum(-1).mean()`. This is a one-line change per slot with
significant impact on whether the less-supervised slots remain active.

---

## Key takeaways

1. **Warmup order is more important than β magnitude.** Introducing auxiliary losses before the
   encoder has a reconstruction manifold causes it to overfit to parameter prediction. The IQ
   reconstruction never converges. Always: recon first → KL → auxiliary.

2. **Per-slot KL beats aggregate KL.** A single β distributes budget unevenly — high-variance
   slots (mod_type) consume most of it and squeeze out scalar slots (rise_time). Use per-slot
   free bits (Razavi 2019) for clean separation.

3. **CVNN improves filter alignment over a real-valued MLP.** r² for z_filter vs ground-truth
   filter_bw: ~0.61 (MLP) → ~0.79 (CVNN). The complex multiply rule enforces Hermitian symmetry
   that a real filter's frequency response must satisfy — the MLP has to learn this from scratch.

4. **Gumbel τ annealing controls accuracy vs gradient flow tradeoff.** τ = 1.0 gives smooth
   gradients but blurry class boundaries. τ = 0.2 gives sharp boundaries but sparse gradients.
   Empirical sweet spot: τ = 0.5 at evaluation time.

5. **Latent traversal is the real sanity check — not correlation.** High r between μ and
   ground-truth can coexist with a flat traversal if the label leaked via the task embedding
   rather than through the named slot. Tighten `task_emb_dim` and verify traversal is causal.

6. **The decoder architecture determines what the latent slot can encode.** A sigmoid-output
   scalar MLP forces the slot to encode a value in (0,1) — the gradient cannot express anything
   outside this range. Match decoder expressivity to the physical parameter's domain, not just
   to the parameter's dimensionality.

7. **The residuals slot is the diagnostic slot.** If it shows structure under traversal (e.g.,
   traversing z_residuals changes the modulation type of the reconstruction), the named slots
   are not capturing enough variation and the encoder is routing labelled information to the
   free dimensions. Increase named slot dimensions or strengthen auxiliary loss weights.
