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

## Layer 3 — `SffProof/Main.lean`

| Lean theorem | Obligation | Fact |
|---|---|---|
| `scff_alignment_at_init` | 5 | feed Layer-2 hypotheses into Obl 4 (linear) ⇒ `1 − A^(ℓ) ≤ 2K/(c‖∇g‖)·(1/√n) + 2/(c‖∇g‖)·δ` |

(Obligation 4 `alignment_perturbation_bound` + its linear corollary
`alignment_perturbation_bound_linear` are proven in Layer 1 / `Skeleton.lean`.)

**Verification:** `#print axioms scff_alignment_at_init` → `[propext, Classical.choice,
Quot.sound]` only (no `sorryAx`). The headline theorem is fully proven modulo the explicitly
bundled hypotheses.
