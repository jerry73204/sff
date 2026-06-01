# Design: Skip-Connection SCFF (residual & dense) — method revision

**Date:** 2026-06-02
**Status:** approved (brainstorming) → ready for implementation plan
**Track:** E (empirical, method revision)

## Motivation

The Track-L proof and Track-E experiments agree on a decomposition of the SCFF
gradient-alignment deficit at initialization:

```
1 - A^(ℓ)  ≤  C/√n  +  C'·δ.
```

Findings so far:

- **Width kills the isotropy term `C/√n`** — proven (`gram_subspace_isotropy_bound`) and
  confirmed empirically (E1: the `Aniso` term scales `n^{-1/2}`, slopes −0.45, −0.53).
- **The binding term is `δ`** — cross-layer kernel drift (`p^(ℓ) ≠ p^(L)`). It is a **depth**
  effect, not cured by width (E1: total `1 - A` is flat in `n`).
- **Persistence fails** under training via **downstream/cross-layer anisotropy** growth (E2
  probe), independent of learning rate, and not an instability (weight norms preserved).
- **Local K-FAC Fisher does not help** — the alignment-breaking anisotropy lives in the
  downstream/cross-layer Jacobian, which a local-layer preconditioner cannot control.

Recurring conclusion: **the bottleneck is cross-layer**, and purely local mechanisms hit a
ceiling. Skip connections attack the cross-layer structure *architecturally* while keeping the
SCFF update local and forward-only.

### Why skips should help (mechanism)

The alignment-breaker is the downstream Jacobian `M^(ℓ+1→L)` deviating from a scalar on the
contrastive subspace `V`. For a plain net `M = Π_{k>ℓ} W^(k)` — a product of random matrices,
anisotropic unless `n` is huge, and growing more anisotropic during training.

- **Residual** `y^(ℓ) = y^(ℓ-1) + α·F^(ℓ)(y^(ℓ-1))` gives
  `M^(ℓ+1→L) = Π_{k>ℓ}(I + α J_k) ≈ I + α Σ_k J_k`. With small residual scale `α`:
  `Vᵀ MᵀM V ≈ I + O(α‖ΣJ‖)` — **isotropy by construction**, independent of width; the kernel
  barely drifts so **`δ` is small**; and training increments `J_k` stay small so anisotropy
  grows slowly (**persistence**). Fixes both failure modes at once.
- **Dense** `y^(ℓ) = F^(ℓ)([y^(0),…,y^(ℓ-1)])` gives the final loss a direct, near-identity
  gradient path to every earlier layer (concatenation), forcing overlap of `∇L_con|_ℓ` with
  the local `∇g^(ℓ)`. Stronger coupling, messier Jacobian.

## Goal & success criteria

Test whether residual and dense skip connections lift init alignment and rescue persistence,
via a **depth sweep** `L ∈ {4, 8, 16}` over `arch ∈ {plain, residual, dense}`.

A positive result is: residual/dense show
1. higher init `A`, with the gap over plain **widening with depth**;
2. lower `Aniso`;
3. persistence — `A` stays clearly above plain during training.

A negative or partial result (e.g. residual helps, dense does not; or the depth-gap does not
widen) is still a finding and must be reported honestly.

## Approach (chosen: A1, generic autograd)

Extend the existing harness; use **autograd** for the local goodness gradient and the
downstream Jacobian so the code is architecture-agnostic and exact. The plain-mode closed-form
gradient stays as the verified `1e-5` anchor; residual/dense ride autograd, which that anchor
validates. No closed-form derivation for residual/dense (buys runtime, not insight).

## Components

### 1. `model.py` — architectures

Add `arch ∈ {plain, residual, dense}`. Fixed width `n` throughout (skips need matching dims).
This is **additive**: the existing `MLP` default behavior (no stem, no skips) is preserved, so
the already-run E1/E2/Fisher experiments are unaffected. The new architectures are reached only
via the `arch` argument (with `stem=True`), e.g. through a new `ArchMLP` class or an extended
`MLP`; either way the depth sweep uses a uniform construction across the three archs.

- **Stem** `W⁰ : d_in → n` (plain projection, establishes width), applied to **all** archs in
  the sweep so the comparison is fair. Then `L` blocks at width `n`; depth = number of blocks
  `L`. `forward` returns `[y⁰ = stem(x), y¹, …, y^L]` for every arch.
- **plain**: `yˡ = act(Wˡ yˡ⁻¹)` (current behavior).
- **residual**: `yˡ = yˡ⁻¹ + α·act(Wˡ yˡ⁻¹)`; `α` configurable, default `1/√L` ("stable" init
  keeping `M ≈ I`). `α = 0` must reduce to a no-op residual branch (used in tests).
- **dense**: `yˡ = act(Wˡ · concat(y⁰,…,yˡ⁻¹))`; `Wˡ` in-dim `= d_in + (ℓ-1)·n`. Keep `n`
  moderate (128) so `L=16` (concat ≈ 2048) stays tractable. μP init scales `1/√fan_in` on the
  growing concat dim; last block keeps the extra `1/√fan_in` (μP).
- `forward` returns all block outputs `[y⁰=stem(x) or x, y¹, …, y^L]` consistently across archs
  so downstream code is uniform.
- `zˡ = normalize(yˡ)` per block, unchanged.

### 2. `gradients.py` — generic autograd

- Generalize `_layer_reps(model, x, layer)` to rebuild `zˡ` as a differentiable function of
  `Wˡ` only, arch-correct, with layer inputs detached:
  - plain: `yˡ = act(ŷ_prev @ Wˡᵀ)`
  - residual: `yˡ = ŷ_prev + α·act(ŷ_prev @ Wˡᵀ)`
  - dense: `yˡ = act(concat(ŷ⁰..ŷˡ⁻¹) @ Wˡᵀ)`
- `local_grad_autograd`, `global_grad`, `alignment_cosine` already generic (operate through
  `model(x)` / the reps); confirm they work per arch.
- The plain closed-form `local_grad` / `local_grad_factors` are unchanged and remain the anchor.

### 3. `metrics.py` — generic downstream Jacobian

- New `downstream_jacobian(model, x, layer)` computing `M^(ℓ+1→L) = ∂y^(L)/∂y^(ℓ)` via
  `torch.autograd.functional.jacobian` on a closure that maps `yˡ → y^(L)` while holding the
  frozen earlier activations (needed for dense's concat paths). Per-sample (batch loop),
  batch-mean for the reported `M`. Arch-agnostic; works in linear and ReLU.
- Replaces the linear-only weight-product Jacobian in the new experiment. The old
  `downstream_jacobian_linear` is kept only as a correctness reference for tests.
- `aniso`, `delta_gram`, `contrastive_subspace` unchanged (consume the generic `M` / reps).

### 4. `experiments/e_arch_depth.py` — the sweep

Grid `L ∈ {4,8,16}` × `arch ∈ {plain, residual, dense}`; `n=128`, batch `B=4` (`≪ √n`),
`act=linear` primary (ReLU optional flag), `tau=0.5`, ≥5 seeds.

- **(a) Init vs depth**: at initialization, mean `1−A`, `Aniso`, `δ` over non-final blocks,
  per `(arch, L)`.
- **(b) Persistence**: at fixed `L=8`, train each arch with SCFF (`scff_step`), track mean `A`
  vs step.

Outputs under `runs/` (CSV + YAML config + verdict) and `plots/`:
- plot 1: `1−A` and `Aniso` vs depth `L`, one curve per arch (init).
- plot 2: `A` vs step, one curve per arch (persistence, `L=8`).
- a written verdict checking the three success criteria.

### 5. Tests (`tests/`)

- Existing 12 pass unchanged (plain closed-form anchor, μP, metrics, Fisher).
- Add:
  - residual/dense forward output shapes correct across `L`;
  - generic `downstream_jacobian` matches `downstream_jacobian_linear` for plain linear arch
    (validates the autograd Jacobian);
  - residual with `α = 0` produces the same block output as the residual branch removed
    (i.e. `yˡ = yˡ⁻¹`), and the generic anchor (`local_grad_autograd`) runs for all archs.

## Data flow

```
x → model(arch, L)  →  [y⁰=stem(x), …, y^L]  →  zˡ = normalize(yˡ)
                        │
   local: ∇g^(ℓ) = autograd of detached-input block goodness wrt Wˡ
   global: ∇g^(L) backprop'd to Wˡ  (probe only)
   A^(ℓ) = Frobenius cosine(local, global)
   M = ∂y^L/∂y^ℓ (autograd) → Aniso;  δ from K^(ℓ) vs K^(L)
```

## Non-goals / YAGNI

- No closed-form local gradient or Jacobian for residual/dense (autograd suffices).
- No new Lean formalization in this revision (a deterministic residual-isotropy proof
  `M = I + small` is noted as an attractive follow-up, out of scope here).
- No ReLU-primary study; ReLU is an optional flag.
- No new optimizer/Fisher variants here.

## Risks

- Dense concat at `L=16` → wide weight matrices; mitigated by moderate `n=128`.
- Per-sample autograd Jacobian is slow; mitigated by small `n`, `B`, and seeds; loop is fine
  at this scale (CPU).
- Residual scale `α` choice affects the result; default `1/√L`, exposed as a knob for a
  follow-up sweep if the effect is borderline.

## Acceptance

- All tests pass (existing + new).
- `e_arch_depth.py` runs end-to-end, emits CSV + 2 plots + verdict.
- Verdict reports, per the three success criteria, whether residual and/or dense raise init
  `A` (gap widening with depth), lower `Aniso`, and persist — honestly, including partial or
  negative outcomes.
