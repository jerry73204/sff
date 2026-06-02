# SCFF / NGD-FF — Findings

Consolidated synthesis of the two-track study (Lean formalization + PyTorch experiments) of
gradient alignment between Self-Contrastive Forward-Forward (SCFF) and backpropagation.

## The question

SCFF trains each layer on its own local InfoNCE "goodness" — forward-only, no backward pass,
no weight transport. Does the **local** layer gradient `∇g^(ℓ)` point the same way as the
**global** backprop gradient `∇L_con|_ℓ`? Measured by the alignment cosine
`A^(ℓ) = cos∠(∇g^(ℓ), ∇L_con|_ℓ)`. `A=1` ⇒ local learning = BP direction, no backward pass.

## The central decomposition

Both gradients are sums of outer products `Σ_i (leftᵢ)(rightᵢ)ᵀ` sharing the **same right
factor** `(y^(ℓ-1)ᵢ)ᵀ`. So alignment lives entirely in the **left factors**, and reduces to two
conditions (proven exactly, `alignment_one_of_isotropic_and_matched`):

1. **Isotropy**: downstream operator scalar on the contrastive subspace, `MᵀM|_V = c·I`.
2. **Softmax/kernel match**: `p^(ℓ) = p^(L)`.

At initialization, wide network, while `d_V = o(√n)`:

```
1 − A^(ℓ)  ≤  C/√n  +  C'·δ
```

- `C/√n` — the **isotropy** term (random-matrix concentration).
- `δ = ‖p^(ℓ) − p^(L)‖` — the **kernel-drift** term (how much the per-layer kernel changes
  across depth).

## Track L — Lean (sorry-free)

| theorem | statement |
|---|---|
| `scff_alignment_at_init` | headline: `1 − A^(ℓ) ≤ 2K/(c‖∇g‖)·(1/√n) + 2/(c‖∇g‖)·δ` |
| `scff_alignment_at_init_expectation` | random-init form: `E[1−A] ≤ (2/(c·g₀))(K/√n + δ)` |
| `gram_subspace_isotropy_bound` | random/wide isotropy: `E‖VᵀM̃V‖_F ≤ d·√(K/n)` |
| `residual_isotropy` | **residual** isotropy: `M=1+E ⇒ ‖MᵀM−1‖ ≤ 2‖E‖+‖E‖²` (deterministic) |
| `softmax_l1_le_linear` | softmax glue: logits within `ε` ⇒ `‖p(a)−p(b)‖₁ ≤ 2e^{2M}ε` |
| `ratio_perturbation` | normalization (`z=y/‖y‖`) lifting |

The deep random-matrix fact is **proven in expectation** (not axiomatized as the design
allowed). `#print axioms` on the headline → only `propext, Classical.choice, Quot.sound`.

## Track E — Experiments (gradient↔autograd verified to 1e-5)

**E1 — init scaling.** The isotropy term scales `Aniso ∝ n^{−1/2}` (fitted slopes −0.45,
−0.53) — empirically validates `gram_subspace_isotropy_bound`. But total `1−A` is **flat in
`n`**: the binding term is `δ`, a **depth** effect not cured by width.

**E2 — persistence.** Under SCFF training, alignment **degrades** (probe: genuine dynamical
anisotropy of the downstream Jacobian on `V`, not instability/lr/`d_V`). **Local K-FAC Fisher
does not rescue it** — the breaking anisotropy is cross-layer, which a local-layer
preconditioner cannot control, and small batches (required for `d_V≪√n`) make the Fisher
factors rank-deficient.

**E3 — batch/width.** `Aniso ∝ √d_V` (fitted 0.51), a **smooth crossover** at `d_V ≈ √n`
(not a sharp knee). Maps the validity boundary: width buys isotropy only while `d_V ≪ √n`.

## Method revision — residual skip connections

The recurring diagnosis: **the bottleneck is cross-layer** (init `δ`, training anisotropy);
purely-local methods (incl. local Fisher) hit a ceiling. Residual skips attack it
architecturally:

- `M = ∏(I + αJ) ≈ I` ⇒ **isotropy by construction** (no large `n` needed) and a near-frozen
  kernel (small `δ`).
- **Result** (depth sweep `L∈{4,8,16}`): residual lifts init `A` with a gap over plain that
  *widens* with depth (residual `1−A` 0.14/0.21/0.24 vs plain 0.45/0.57/0.69), lowers `Aniso`,
  and **persists** under training (`A` 0.79→0.73 vs plain 0.43→0.34). **Dense does not help**
  (its downstream Jacobian is not near-scalar).
- **Scale law**: `1−A ≈ Aniso ≈ O(α)` (linear in the residual scale). `α ≤ 0.1` gives
  near-perfect persistent alignment; the textbook `1/√L ≈ 0.35` is **too large**; want `~1/L`.
  **ReLU residual holds** (same trend, slightly better at small `α`).
- **Proven**: `residual_isotropy` (`‖MᵀM−1‖ ≤ 2‖E‖+‖E‖²`) is the deterministic `Aniso=O(α)`
  law — turns the empirical win into a theorem, no random matrix, no `o(√n)`.

## Negative results (honest, mechanism understood)

- **Local Fisher (NGD-FF)** does not rescue persistence — cross-layer problem + rank-deficient
  small-batch Fisher.
- **Forward-gradient-on-subspace** (a BP-free estimate of the global gradient via forward-mode
  JVP, tangents restricted to `span(V)⊗span(y_prev)`): the variance machinery works
  (≈4000× reduction, the project's two theorems are exactly the variance reducer), but as a
  training-signal correction it **does not beat pure SCFF**. Catch-22: where local≈global
  (residual) the estimate is redundant noise; where a correction is needed (plain/deep, low
  `A`) the *local* subspace `V` captures only ~⅓ of the global gradient. The same `δ` coupling
  defeats it both ways.

## Relation to prior work (verified survey)

- The SCFF base paper (Nature Comms 2025) contains **no** alignment/NTK/Jacobian/Fisher/
  residual/subspace theory — all of the above is novel relative to it.
- Closest prior art is **LoCo** (overlapping local blocks add effective depth + implicit
  feedback to close the local-vs-BP gap) — *conceptually parallel* to our residual result, but
  empirical/architectural with no isotropy/Jacobian quantification. **Mono-Forward** (objective,
  not locality, is the bottleneck) is adjacent to our "cross-layer is binding."
- **The SCFF authors themselves flag "top-down feedback connections" as the route to scaling**
  to ResNet-50/ViT — i.e. the cross-layer signal our findings identify as the bottleneck.
- Nearest *theoretical* neighbors (Boopathy & Fiete 2022 NTK-local-vs-BP; Ren et al. 2022
  forward-gradient + local losses) were **not verifiable** in the survey and should be read
  directly before asserting full novelty.

## The honest headline

Local SCFF aligns with BP only up to a cross-layer term `δ`. Width fixes the isotropy half but
not `δ` (a depth effect). A small-scale **residual** architecture fixes both — provably,
cheaply, forward-only — and cleverer BP-free correction schemes (Fisher, forward-gradient) do
not beat it.

## Gaps to practical training

1. **Alignment ≠ accuracy.** We measure a proxy (gradient alignment to BP), not a trained
   model. SCFF's own numbers show the real gap (CIFAR-10 80.75% vs BP >90%; Tiny-ImageNet
   35.67%).
2. **Scale + regime.** Toy widths/depths/batches, synthetic data; the theory lives in
   `B ≪ √n`, `d_V = o(√n)`, which practical batch sizes likely violate.
3. **Linear-primary, no normalization.** `δ` may be partly an artifact of plain MLPs without
   LayerNorm/BatchNorm, which already control kernel drift in real nets.
4. **The FF value-prop untouched.** Memory (no stored activations), locality, parallelism —
   none measured; only the alignment proxy.
