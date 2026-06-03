# Spatial (per-location) SCFF — design spec

**Goal:** Fix the conv bottleneck by moving the SCFF local InfoNCE objective from the
global-average-pooled rep (a re-introduced bug) to **per spatial location**, restoring spatial
gradient to the conv filters and preserving the residual-skip isometry. Target: per-location-SCFF
≫ global-pool-SCFF (our ~0.35 on CIFAR-10) and `A` recovering toward 1.0.

**Background / why.** Our conv SCFF computed goodness on `z = normalize(mean_{H,W} y) ∈ R^C`. The
diagnosis (`docs/FINDINGS.md`, geometric/transport analysis): global pooling (a) gives the conv
filters no spatial gradient — the one thing convolution is for — and (b) is a low-dimensional,
non-invertible projection that destroys the residual `M≈I` isometry the alignment theory relies on
(pooling blows up the downstream condition number `κ`). The literature study (`docs/RELATED_WORK.md`)
confirmed the **published SCFF (Nat. Commun. 2025) computes conv goodness per spatial location**
`G_{h,w} = (1/C) Σ_c y²_{c,h,w}` and reaches CIFAR-10 80.75% — so our global pool is a divergence,
not the method. GIM (Löwe 2019) reaches STL-10 81.9% with per-patch/location contrast in the
gradient-isolated regime. The fix: per-location InfoNCE, keeping our self-contrastive framework.

## Architecture

`ConvSCFF` (residual conv blocks, the `scff_signal` CUDA kernel, the forward-only per-block local
step) is **unchanged**. Only the objective head changes: tokens become spatial locations, not the
pooled image. The residual blocks, `apply_block`, `pooled` (kept for the probe), and the kernel are
reused as-is.

## The per-location objective

Block output `y ∈ [B, C, H, W]` (H=W=16 after the stride-2 stem; C=64).

- **Tokens:** per location, `z[b,h,w] = y[b,:,h,w] / ‖y[b,:,h,w]‖ ∈ R^C`. Laid out per image as
  `[H·W, C]` (H·W = 256).
- **Positive:** the same `(b,h,w)` location in an **appearance-only** augmented view
  (`x⁺ = x + noise`, NO flip/crop → location `i ↔ i` corresponds trivially, no correspondence solver).
- **Negatives:** the other `H·W − 1` locations **in the same image** (in-image InfoNCE — drives each
  location's rep to be *distinct from other locations*, i.e. spatial feature selectivity).
- **Kernel reuse:** call `scff_signal(z_b, z⁺_b, τ)` per image, with the `H·W = 256` locations as the
  kernel's "batch" (fits exactly: kernel requires `B ≤ 256`, `C = 64 ≤ 512`). Returns the tangent
  signal `s⊥_b ∈ [H·W, C]` (`= +∂g/∂z` per location), no autograd graph.

**Why this fixes it:** the objective now lives in (near) feature-map space, so (a) every filter
position receives contrastive gradient and (b) the residual `M≈I` isometry applies — pooling no
longer projects the transport into a rank-deficient space, so `κ` is bounded and `A` should recover.

## Training step (`scff_local_step_spatial`)

Per block `ℓ`, forward-only and layer-local (mirrors the existing `scff_local_step`, tokens =
locations):

1. `x⁺ = augment_appearance(x, noise)` (noise only). Forward `x` and `x⁺`, detaching between blocks
   (`ys`, `ysp` are full forwards, detached).
2. `out = apply_block(ys[ℓ].detach(), ℓ)` — differentiable wrt `blocks[ℓ]` only.
3. `z = per_location_normalize(out) ∈ [B, H·W, C]`; `z⁺ = per_location_normalize(ysp[ℓ+1]).detach()`.
4. For each image `b`: `s⊥[b] = scff_signal(z[b].detach(), z⁺[b], τ)`; stack to `s⊥ ∈ [B, H·W, C]`.
5. `grads = autograd.grad(z, blocks[ℓ].parameters(), grad_outputs=s⊥)`; ascend `p += lr·grad`.

Cross-block stop-grad (step 2 detaches the input) keeps it local and forward-only. The per-image
loop (B kernel calls per block) is the simple v1; a batched kernel is a future optimization.

## Data flow

CIFAR-10 → block forward → per-location tokens → per-image kernel InfoNCE → `s⊥` → block conv-grad →
ascend. **Probe (eval only):** global-avg-pool the *frozen* trained block features → concat over
blocks → sklearn linear probe (pooling a frozen feature for the probe is fine; only the *training*
objective must stay spatial). Alignment `A` and `κ` measured per the existing GPU machinery, but on
the spatial signal.

## Components / files

- `empirical/gpu_arch.py` — add:
  - `per_location_tokens(y) -> [B, H·W, C]` (permute + reshape + L2-normalize over C).
  - `scff_local_step_spatial(model, x, xp, tau, lr)` (the step above).
  - `block_goodness_spatial(model, x, xp, tau)` (sum of per-image per-location goodness, for tests).
  - Keep the existing global-pool `scff_local_step` / `block_goodness` for the A/B comparison.
  - `augment_appearance(x, noise)` (noise only) — or reuse `cifar_conv.augment` with flip disabled.
- `empirical/experiments/cifar_spatial.py` — compare **per-location-SCFF vs global-pool-SCFF vs
  supervised-BP** on CIFAR-10 (residual conv), report accuracy + alignment `A` + κ.
- `empirical/tests/test_scff_kernel.py` — append: token-shape test; a `scff_local_step_spatial`
  ascent test (per-location goodness rises); locality test (block-ℓ grad only).

## Experiment (`cifar_spatial.py`)

Residual conv (`C=64`, depth e.g. 8, `α=1/√L`), CIFAR-10, appearance-only aug. Arms:
`supervised-BP`, `global-pool-SCFF` (baseline = our current ~0.35), `per-location-SCFF` (the fix).
Report probe accuracy, `A`, `κ`. Success = per-location-SCFF ≫ global-pool-SCFF, approaching BP / the
paper's ~0.80; `A` recovers toward 1.0.

## Risks / mitigations

- **In-image-negative collapse / too-easy** (study flag): if per-location-SCFF underperforms or
  collapses, add **cross-image negatives** (mix other images' locations) — noted extension, deferred.
- **Per-image kernel-loop cost** (B calls/block): acceptable on GPU for v1; if too slow, **downsample
  the objective to ≤16×16** (already at 16×16) or batch the kernel — future optimization.
- **Appearance-only aug is weak** (no flip/crop): if the contrastive task is too easy (positive ≈
  query), strengthen *appearance* aug (color jitter, stronger noise) — keep it spatial-transform-free
  to preserve correspondence.
- **Probe pooling**: the probe pools a frozen feature; this is eval-only and does not affect the
  (spatial) training objective.
- **H·W > 256** (deeper/larger maps): the kernel caps at 256 tokens; our maps are 16×16=256. If a
  config exceeds it, downsample before the objective or fall back to the pure-torch reference.

## Testing

`pytest` (append to `test_scff_kernel.py`, CUDA-gated): (a) `per_location_tokens` returns
`[B,H·W,C]` with unit-norm rows; (b) `scff_local_step_spatial` raises `block_goodness_spatial` over
20 steps (ascent, sign check); (c) locality — block-ℓ step leaves block-`k≠ℓ` params unchanged.

## Success criteria

- `per-location-SCFF` accuracy ≫ `global-pool-SCFF` on CIFAR-10 (target: meaningful fraction of the
  ~0.45 gap to BP closed; ideally approaching the paper's ~0.80).
- Residual `A` recovers above the global-pool ~0.25 (toward 1.0), confirming isometry restored.
- Tests pass on GPU; the global-pool baseline reproduces ~0.35 for a clean A/B.
