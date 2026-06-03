# gen-FF (cheap config) — design spec

**Goal:** Test whether reframing forward-only learning as a joint `(x,y)` energy model — (a) early
layers that model the data manifold via *denoising* negatives, and (b) a one-forward joint-energy
classification head — beats SCFF's instance-discrimination-plus-probe on MNIST, **at ≤ BP cost**, and
gains generative-classifier robustness (calibration, OOD). MNIST/MLP first (forward-only ≈ BP there,
so the generative machinery is tested without the conv price-of-locality confound).

**Background:** see `docs/ideas/2026-06-04-ff-generative-ebm.md`. SCFF's measured weaknesses: weak
in-batch negatives, label-blind objective, separate probe. gen-FF puts `y` in a joint energy
(`p(x,y) ∝ e^{−E}`, `E=−goodness`), classifies by conditioning on `x` (`argmax_y −E`), and uses
denoising/model negatives. The **cheap config** (this spec) keeps inference at **1 forward** by
putting the label energy at the **top** (not in the input — label-in-input would cost `C×` forwards).

## Architecture

A plain MLP backbone `GenFFMLP` shared by all arms (so arms differ only in the *objective*, not
capacity). `forward(x)` returns the list of layer activations `[h_1, …, h_L]`. Features for the head /
probe = `concat(normalize(h_ℓ))` over layers (same as SCFF). On top, an `EnergyHead` = a linear map
`width·L → C` whose outputs are the `C` class **goodnesses**; the energy is `E(x,y) = −head(feat)[y]`.

## Components (`empirical/genff.py`)

- `GenFFMLP(d_in, width, n_layers)` — plain MLP; `forward(x) -> [h_1..h_L]` (ReLU); `features(x)` =
  `concat_ℓ normalize(h_ℓ)`.
- `train_early_inbatch(model, X, cfg)` — SCFF-style: per layer, InfoNCE goodness on `normalize(h_ℓ)`,
  positives = noise-augmented view, **negatives = in-batch**; forward-only local ascent (reuse the
  existing `gradients.local_goodness` contrastive form). The existing-SCFF early objective.
- `train_early_denoise(model, X, cfg)` — **the new gen-FF early objective**: per layer, squared-norm
  goodness `G_ℓ = mean(h_ℓ²)`; **positive = real `x`, negative = noised `x̃ = x + σξ`**; logistic loss
  `−log σ(G_ℓ(real) − θ) − log σ(θ − G_ℓ(noised))`; forward-only, layer-local update (input detached
  between layers). Features model the data manifold `p(x)`, not instance identity.
- `EnergyHead(width·L, C)` — linear; `class_energies(feat) = head(feat)` (the `C` goodnesses).
- `train_head(model, head, X, y, cfg)` — train the head only (backbone frozen / features detached):
  loss = **CE** `cross_entropy(head(feat(x)), y)` (discriminative, `p(y|x)=softmax`)
  **+ λ·EBM** `softplus(LSE(head(feat(x̃_noise))) − LSE(head(feat(x))))` where `LSE = LogSumExp_y` is the
  negative free energy: pushes real-feature free-energy up, noised-feature down (models `p(x)`).
  One-layer backward (cheap).
- `predict(model, head, X)` — `argmax_y head(feat(x))`, **one forward**.
- `free_energy(model, head, X)` — `−LSE_y head(feat(x))` (for OOD scoring).
- `probe(model, X, y, ...)` — logistic-regression probe on `features` (the SCFF baseline path).

## Experiment (`empirical/experiments/genff_mnist.py`)

Four arms, shared backbone, MNIST (reuse `experiments/task_accuracy._load_raw` / `load_data`):

1. **supervised-BP** — backbone + linear head, full cross-entropy backprop. *Ceiling, cost reference.*
2. **SCFF+probe** — `train_early_inbatch` → logistic probe. *Our current method.*
3. **SCFF-feat+head** — `train_early_inbatch` → `train_head`; classify `argmax`. *Isolates the head.*
4. **denoise-feat+head** — `train_early_denoise` → `train_head`; classify `argmax`. *The real gen-FF.*

Per arm report:
- **accuracy** (test).
- **inference cost** — assert arms 1–4 are **1 forward** (vs the rich label-in-input `C×`); state it.
- **calibration** — ECE (expected calibration error) for the head arms (2 uses probe probs).
- **OOD** — AUROC separating MNIST test from an OOD set (uniform-noise images; optionally FashionMNIST
  via the same idx loader) using `free_energy` (head arms) — the generative-classifier payoff.

## Data flow

MNIST → backbone forward → early objective (in-batch InfoNCE *or* denoising squared-norm) trains the
backbone forward-only → freeze → train head (CE + EBM) on detached features → classify `argmax`
(1 forward). OOD: `free_energy` on noise/FashionMNIST.

## Success criteria

- **Accuracy:** arm 4 (and/or 3) ≥ arm 2 (SCFF+probe) — the energy/denoising reframe does not cost
  accuracy, ideally beats it. Arm 1 (BP) is the ceiling.
- **Cost:** all arms classify in **1 forward**; early training is forward-only closed-form/local (no
  global backward), so ≤ BP cost. State the forward-count comparison explicitly.
- **Robustness (the payoff):** head arms (3,4) beat the probe arm (2) on **ECE and OOD-AUROC** — the
  generative-classifier benefit, even in the cheap config.
- Decisive read: does **denoising/manifold** early training (arm 4) beat **in-batch** (arm 3)? and does
  the **joint-energy head** (arms 3,4) beat the **probe** (arm 2)?

## Risks / mitigations

- **θ (goodness threshold) and σ (noise) tuning** — the denoising FF is sensitive; sweep small grids;
  per-layer goodness normalization (mean over units) for stability.
- **EBM term instability** — keep λ small; the EBM negatives here are *noised features* (cheap, no
  MCMC), which are far more stable than SGLD; do not add Langevin in this spec (opt-in later).
- **Head-only training may underwhelm** — if arms 3/4 ≈ arm 2, the conclusion is "the head alone isn't
  enough; the generative signal must reach features" → motivates a follow-up where the EBM term
  trains the backbone too (costs a backward; out of this cheap-config scope).
- **OOD set** — uniform noise is the easy OOD; add FashionMNIST for a harder, more honest AUROC.

## Testing (`empirical/tests/test_genff.py`)

- `train_early_denoise` one step raises `G_ℓ(real) − G_ℓ(noised)` (manifold contrast increases).
- `EnergyHead` returns `[B, C]`; `predict` is one forward and returns labels in `[0,C)`.
- locality: a denoise step on layer ℓ leaves layer `k≠ℓ` params unchanged (forward-only, stop-grad).
- `free_energy` on real < on uniform-noise after head training (OOD direction sane).
