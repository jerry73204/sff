# LEAN_NOTES.md — Lean definition ↔ THEORY.md symbol map

Maps each Lean def/theorem to its `THEORY.md` symbol. Required by Track-L acceptance (§3.4).

## Layer 0 — `SffProof/Defs.lean`

| Lean | THEORY.md symbol | Notes |
|---|---|---|
| `outer a r` | `(left)(right)^T` outer product | builds gradient terms |
| `frob A C` | Frobenius inner `⟨·,·⟩_F` | `Σ_{ij} A_{ij}C_{ij}` |
| `frobNorm A` | `‖·‖_F` | `√(frob A A)` |
| `cosAngleM A C` | `A^(ℓ)` (alignment cosine, §4) | Frobenius cosine of two gradient matrices |
| `Reps n B` | `{z^(ℓ)(x_i)}` | batch of unit reps |
| `gram z` | `K^(ℓ)` (§2) | `K_{ij}=⟪z_i,z_j⟫` |
| `scores z τ` | `⟪z_i,z_j⟫/τ` | softmax logits |
| `softmaxWeights z τ` | `p^(ℓ)_{ij}` (§2) | in-batch InfoNCE weights |
| `signal z zpos τ i` | `s_i^(ℓ)` (§3) | `z_{i+} − Σ_j p_{ij} z_j` |
| `gradFromFactors left right` | `∇_{W^(ℓ)} = Σ_i (left_i)(right_i)^T` (§3) | shared right factor |

## Layer 1 — `SffProof/Skeleton.lean`

| Lean theorem | Obligation (§3.2) | THEORY.md / design fact |
|---|---|---|
| `inner_product_cosine_one_of_parallel` | 1 | warm-up: parallel ⇒ cosine 1 |
| `gradient_shared_right_factor` | 2 | local & global share `(y^(ℓ-1)_i)^T`; scaling left scales grad |
| `cosAngleM_eq_one_of_parallel` | (support) | Frobenius parallel ⇒ cosine 1 |
| `alignment_one_of_isotropic_and_matched` | 3 | isotropy `M^TM\|_V=c·I` + `p^(ℓ)=p^(L)` ⇒ `A^(ℓ)=1` |
| `alignment_perturbation_bound` | 4 | `G = c·g + Err` ⇒ `1 − A ≤ 2‖Err‖²/(c‖g‖)²` |
| `one_sub_cosAngle_eq`, `norm_normalize_sub_le`, `cosAngle_smul_right` | (support) | chord identity + renorm Lipschitz |
| `frob_smul_left`, `frob_smul_right`, `frobNorm_smul`, `frob_self_nonneg`, `frobNorm_mul_self` | (support) | Frobenius algebra |

Note on Obl 4: stated abstractly for any real inner product space `E` (the Lipschitz/
real-analysis content), so it applies verbatim to the Frobenius gradient space. The error
`Err` is the operator anisotropy `E` on `V` plus the softmax mismatch `δ` (THEORY.md §5).
Bound is quadratic in `‖Err‖`; in the small-error regime it gives the linear
`1 − A ≤ C·‖Err‖` of design §3.1.

## Layer 2 — `SffProof/Hypotheses.lean` (axiomatized — NOT YET FORMALIZED)

Bundled in `structure SCFFInitHypotheses E`. Constructing a term = supplying the analytic
facts; the structure carries no `sorry`.

| Lean field | THEORY.md symbol | Status |
|---|---|---|
| `isotropy_at_init` | `‖V^T(M^TM)V − c·I‖ ≤ K/√n` → `‖errIso‖ ≤ K/√n` (§5) | hypothesis (random-matrix) |
| `gram_match` | `‖p^(ℓ) − p^(L)‖ ≤ δ` → `‖errGram‖ ≤ δ` (§5) | hypothesis (Gram matching) |
| `decomp` | `gGlob = c·gLoc + (errIso+errGram)` (§3) | structural (from Obl 2) |
| `err_small` | `‖err‖ ≤ c‖gLoc‖` | small-error regime (large `n`) |

Each hypothesis field has a docstring: informal statement, why true, citation, NOT YET
FORMALIZED tag. `scripts/check_no_sorry.sh` asserts no `sorry` exists outside this file.

### Discharging the hypotheses — progress (`SffProof/Softmax.lean`)

`gram_match` splits into deterministic glue (proven) ∘ random core (Gram closeness, TODO).
The glue is **softmax stability**, proven elementarily (no calculus):

| Lean theorem | Fact |
|---|---|
| `softmax_sub_abs_le` | logits agree to `ε` ⇒ each weight moves `≤ (e^{2ε}−1)·weight` |
| `softmax_l1_sub_le` | `‖p(a) − p(b)‖₁ ≤ e^{2ε}−1` for `ε = ‖logits a − logits b‖_∞` |
| `softmax_sum_eq_one`, `softmax_nonneg`, `softmax_denom_pos` | softmax is a distribution |

This reduces `gram_match` to Gram closeness `ε = ‖scores^(ℓ) − scores^(L)‖_∞ → 0`, the
remaining random-matrix input.

| `softmax_l1_le_linear` | linearized: `ε ≤ M` ⇒ `‖p(a)−p(b)‖₁ ≤ 2e^{2M}·ε` (linear in `ε`) |

`softmax_l1_le_linear` makes the softmax→Gram reduction **linear**, so a logit-closeness
bound `ε ≤ K/√n` gives `‖p^(ℓ)−p^(L)‖₁ ≤ C/√n` directly (no nonlinear `e^{2ε}−1`).

### Gram closeness — random core (`SffProof/Probability/Subspace.lean`)

The unnormalized Gram entry `⟨y_i,y_j⟩ = x_iᵀWᵀWx_j` is the same quadratic form as the
subspace entries (directions = input vectors):

| `gram_entry_abs_le` | `E\|x_iᵀWᵀWx_j − α\| ≤ √(n(γ−α²))` (Jensen on `gram_subspace_entry_sq`) |
| `gram_entry_isotropy_bound` | with `γ−α² ≤ K/n²`: `≤ √(K/n) = O(1/√n)` — Gram entry concentration |

So the score difference `ε = O(1/√n)` in expectation; fed through `softmax_l1_le_linear`
this gives `E‖p^(ℓ)−p^(L)‖₁ = O(1/√n)` — `gram_match` discharged in expectation, modulo
the normalization (`z = y/‖y‖`) across layers.

### Probability scaffolding (`SffProof/Probability/Moments.lean`)

Foundation for `isotropy_at_init` (expectation mode). Over a probability space:

| Lean theorem | Fact |
|---|---|
| `integral_sum_eq_zero` | `E[Σᵢ Xᵢ] = 0` (linearity, mean-zero) |
| `sq_integral_sum_eq` | **workhorse**: `E[(Σᵢ Xᵢ)²] = Σᵢ E[Xᵢ²]` for pairwise-independent mean-zero L² |

| `centered_sq_sum_eq` | engine: `E[(Σᵢ(Uᵢ²−α))²] = |s|·(β−α²)` for indep `Uᵢ`, `α=E[Uᵢ²]`, `β=E[Uᵢ⁴]` |
| `integral_abs_le_sqrt_integral_sq` | Jensen: `E\|D\| ≤ √(E[D²])` (prob measure, via `variance_nonneg`) |

`sq_integral_sum_eq` = Mathlib `IndepFun.variance_sum` ∘ `variance_of_integral_eq_zero`.
`centered_sq_sum_eq` lifts it to centered squares (variance route) — the shared engine for
the diagonal Gram entry and the rank-1 subspace restriction.

### Keystone — random-matrix core (`SffProof/Probability/RandomMatrix.lean`)

Linear / single random matrix `W` (`σ² = 1/n` μP scaling). Rows modelled as i.i.d. random
vectors via `structure RandomMatrixEnsemble`; row independence is a field, mixed within-row
moments are fields (true for i.i.d. entries).

| Lean theorem | Fact |
|---|---|
| `RandomMatrixEnsemble.gram_offdiag_sq` | `E[(WᵀW)_{pq}²] = n·σ⁴` for `p ≠ q` (= `1/n` at `σ²=1/n`) |
| `RandomMatrixEnsemble.gram_diag_centered_sq` | `E[((WᵀW)_{pp} − nσ²)²] = n·(m₄ − σ⁴)` (= `O(1/n)`) |
| `RandomMatrixEnsemble.gram_rank1_centered_sq` | `E[(vᵀWᵀWv − n α_v)²] = n·(β_v − α_v²)` — `d_V=1` 2nd moment |
| `RandomMatrixEnsemble.gram_rank1_abs_le` | Jensen: `E\|vᵀWᵀWv − n α_v\| ≤ √(n(β_v−α_v²))` |
| `RandomMatrixEnsemble.gram_rank1_isotropy_bound` | **`d_V=1` `isotropy_at_init` closed (expectation):** `≤ √(K/n) = O(1/√n)` |

Both entries of `WᵀW − E[WᵀW]` have second moment `O(1/n)`. Off-diagonal: `Σ_k R_kp R_kq`,
row products independent across `k` (via `iIndepFun.comp`), mean-zero → workhorse. Diagonal:
`Σ_k (R_kp² − σ²)` via `centered_sq_sum_eq` with `U_k = R_kp`. Rank-1: `vᵀWᵀWv = ‖Wv‖² =
Σ_k ⟨row_k,v⟩²`, same engine with `U_k = ⟨row_k,v⟩` (rows independent ⇒ projections
independent). Ensemble carries `σ²`, `m₄`; projected moments `α_v,β_v` are hypotheses.

### General `d_V` subspace restriction (`SffProof/Probability/Subspace.lean`)

| Lean theorem | Fact |
|---|---|
| `centered_sum_var_eq` (Moments) | general engine: `E[(Σᵢ(Wᵢ−α))²]=\|s\|(γ−α²)`, `Wᵢ` need not be squares |
| `integral_sqrt_le_sqrt_integral` (Moments) | Jensen for `√`: `E[√F] ≤ √(E[F])`, `F≥0` |
| `gram_subspace_entry_sq` | entry `(a,b)`: `E[(Σₖ(⟨row,vₐ⟩⟨row,v_b⟩−α))²] = n(γ−α²)` |
| `gram_subspace_frob_sq` | `E‖VᵀM̃V‖_F² = Σ_{a,b} n(γ_{ab}−α_{ab}²)` |
| `gram_subspace_frob_le` | Jensen: `E‖VᵀM̃V‖_F ≤ √(Σ_{a,b} n(γ−α²))` |
| `gram_subspace_isotropy_bound` | **general `isotropy_at_init` closed (expectation):** `≤ d·√(K/n) = O(d/√n)` |

`(VᵀM̃V)_{ab} = Σ_k (⟨row_k,vₐ⟩⟨row_k,v_b⟩ − α_{ab})`, a sum over independent rows (product
form, via `centered_sum_var_eq`). Sum the `d²` entry second moments (Frobenius), then Jensen.
With `d_V = o(√n)` the bound is `o(1)`. The deep random-matrix input is now a **theorem** in
expectation, modulo the within-row joint moments. `isotropy_at_init` field itself still stated
as the conditional hypothesis; this proves it holds in expectation for this ensemble model.

## Layer 3 — `SffProof/Main.lean`

| Lean theorem | Obligation | Fact |
|---|---|---|
| `scff_alignment_at_init` | 5 | feed Layer-2 hypotheses into Obl 4 (linear) ⇒ `1 − A^(ℓ) ≤ 2K/(c‖∇g‖)·(1/√n) + 2/(c‖∇g‖)·δ` |

(Obligation 4 `alignment_perturbation_bound` + its linear corollary
`alignment_perturbation_bound_linear` are proven in Layer 1 / `Skeleton.lean`.)

**Verification:** `#print axioms scff_alignment_at_init` → `[propext, Classical.choice,
Quot.sound]` only (no `sorryAx`). The headline theorem is fully proven modulo the explicitly
bundled hypotheses.
