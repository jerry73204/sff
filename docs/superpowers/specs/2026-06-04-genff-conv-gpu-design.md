# Conv gen-FF + GPU pipeline — design spec

**Goal:** Build a focused GPU training pipeline and a convolutional gen-FF model, and run the
headline test (#1): does the **denoising-energy** gen-FF objective — which sidesteps SCFF's
transport/pooling machinery — beat SCFF's conv wall (~0.35 on CIFAR-10)? On the RTX 5090.

**Background:** `docs/ideas/2026-06-04-ff-generative-ebm.md`. gen-FF beat SCFF on MNIST (+2.9pt,
better calibration/OOD) and structurally escapes the align-to-BP transport problem (each layer has a
self-contained generative objective). SCFF failed on conv because global pooling severed its
*contrastive* objective from the residual fix and blew up `κ`. gen-FF's *denoising-energy* objective
is real-vs-noised manifold modeling, **not** instance-contrast-between-locations, so it gives every
spatial position a gradient without the pooling wound — hence the conv test is the natural next step.

## Architecture

A clean, CUDA-resident pipeline: GPU CIFAR data module (batched + real augmentation) + a unified
forward-only trainer + metrics, running four arms over a shared conv backbone.

## Conv gen-FF model

- **Backbone:** the existing `gpu_arch.ConvSCFF` (residual conv blocks; stem 32→16 stride-2; `.forward`
  returns `[stem_out, block0_out, …]`; `apply_block`, `pooled`). Unchanged.
- **Per-location denoising-energy early objective (`conv_denoise_step`, new):** per block `ℓ`, the
  per-location energy `G_{h,w} = mean_C h²_{c,h,w}` (mean over channels → `[B,H,W]`); train block `ℓ`
  so `G` is **high on real, low on noised** input via the paired contrast
  `loss = −mean_{B,h,w} logsigmoid(G_real − G_noised)`. Forward-only, layer-local (block input
  detached, mirrors `scff_local_step`). Per-location (not a global scalar) for full spatial gradient.
- **Energy head:** add `ConvSCFF.features(x)` = `concat_ℓ normalize(pooled(block_ℓ_out))` → `[B, C·L]`;
  reuse `genff.EnergyHead(C·L, n_classes)`, `genff.train_head` (CE + cheap EBM real-vs-noised features),
  `genff.predict`, `genff.free_energy` (all backbone-agnostic — they call `model.features`). Classify
  in **1 forward**.

## GPU pipeline (`gpu_pipeline.py`)

- **Data module:** load CIFAR-10 once (reuse `experiments.cifar_conv.load_cifar`), keep tensors on
  GPU; `iter_batches(X, y, batch, gen)` index-sampler; `augment_batch(x)` = on-GPU random crop
  (reflect-pad 4, random 32×32 crop per sample) + random h-flip per sample. Gaussian noise is *not*
  here — it is the denoising/EBM negative, applied in the objectives.
- **Metrics:** `ece(probs, y)`, `ood_auroc(fe_in, fe_ood)` (moved/shared from `genff_mnist`).
- **Trainer helpers:** thin wrappers that run an arm end-to-end (build model → early train → head
  train → eval) and report `acc, ece, ood_auroc, peak_MB, 1-forward`.

## Experiment (`experiments/genff_cifar.py`)

Four arms, shared `ConvSCFF` backbone (residual, `C=128`, `L=6`, `α=1/√L`), CIFAR-10 (full 50k, real
aug), all forward-only early training, all classify in 1 forward:

1. **supervised-BP** — backbone + head, full cross-entropy backprop. *Ceiling.*
2. **SCFF-conv + probe** — `gpu_arch.scff_local_step` (in-batch InfoNCE via the kernel) → logistic
   probe. *Our prior conv result (~0.35).*
3. **SCFF-conv + energy-head** — in-batch early → `train_head`; classify argmax. *Isolates the head.*
4. **gen-FF-conv (denoise + energy-head)** — `conv_denoise_step` early → `train_head`. *The real test.*

Report per arm: **accuracy, ECE, OOD-AUROC** (vs uniform-noise + optionally CIFAR-100/SVHN), **peak GPU
memory**, and confirm **1-forward inference**. Headline deltas: gen-FF vs SCFF (a4−a2), head vs probe
(a3−a2), denoise vs in-batch (a4−a3).

## Data flow

CIFAR→GPU → per batch: augment (crop+flip) → early objective (denoise per-location energy *or* in-batch
InfoNCE) trains backbone forward-only → freeze → train head (CE + EBM real-vs-noised features) →
classify argmax (1 forward). OOD: `free_energy` on noise / a second dataset.

## Success criteria

- Pipeline runs all 4 arms on GPU efficiently, reporting acc/ECE/OOD/peak-mem per arm.
- **Headline:** does gen-FF-conv (arm 4) beat SCFF-conv (arm 2, ~0.35)? A clear win = the denoising-
  energy reframe dodges the conv price-of-locality. A null = the conv gap is deeper than the objective
  (price of locality bites generative-local too). Either is a real, documented finding.
- All forward-only arms classify in 1 forward, ≤ BP cost; memory reported honestly (note: the
  early-train forward holds `O(L)` activations unless streamed — same caveat as the SCFF memory finding).

## Components / files

- `empirical/gpu_arch.py` (modify) — add `ConvSCFF.features(x)` (concat normalized pooled block reps).
- `empirical/genff_conv.py` (create) — `conv_denoise_step(model, x, cfg)` (per-location denoising
  energy, forward-only); `block_energy_gap(model, x, xn)` (real−noised goodness, for tests).
- `empirical/gpu_pipeline.py` (create) — GPU CIFAR data module (`load_cifar` reuse + on-GPU batching +
  `augment_batch`), `ece`, `ood_auroc`.
- `empirical/experiments/genff_cifar.py` (create) — the 4-arm experiment.
- `empirical/tests/test_genff_conv.py` (create) — denoise step raises the real−noised gap; locality
  (block-ℓ grad only); `ConvSCFF.features` shape; pipeline `augment_batch` shape/range; tiny e2e.

## Risks / mitigations

- **Denoising on conv may still underperform** — if arm 4 ≈ arm 2, the conv gap is not the objective
  (price of locality bites generative-local too). Honest finding; not a failure of the pipeline.
- **Memory of full-forward early training** — `O(L)` activations held; the flat-memory win needs a
  streaming schedule (out of scope; note it). Report peak memory honestly.
- **GPU augmentation correctness** — unit-test `augment_batch` shape + that flips/crops stay in range.
- **σ (noise), θ-free paired contrast, lr** — sweep small grids in the smoke run; per-layer energy is
  unnormalized, so keep lr modest and watch for divergence (as on MNIST).

## Testing (`empirical/tests/test_genff_conv.py`, CUDA-gated where needed)

- `conv_denoise_step` one step raises `block_energy_gap` (real−noised) on held-out data.
- locality: a block-ℓ denoise gradient is None for block `k≠ℓ` params.
- `ConvSCFF.features(x)` returns `[B, C·L]`.
- `augment_batch(x)` returns `[B,3,32,32]`, values finite.
- tiny end-to-end: build → 2-epoch denoise → head → `predict` returns valid labels in 1 forward.
