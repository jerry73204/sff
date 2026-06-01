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
| `frob_smul_left`, `frobNorm_smul`, `frob_self_nonneg`, `frobNorm_mul_self` | (support) | Frobenius algebra |

## Layer 2 — `SffProof/Hypotheses.lean` (axiomatized — NOT YET FORMALIZED)

| Planned Lean | THEORY.md symbol | Status |
|---|---|---|
| `isotropy_at_init` | `‖V^T(M^TM)V − c·I‖ ≤ K/√n` (§5) | hypothesis (random-matrix) |
| `gram_match` | `‖p^(ℓ) − p^(L)‖ ≤ δ` (§5) | hypothesis (Gram matching) |

## Layer 3 — `SffProof/Main.lean`

| Planned Lean | Obligation | Fact |
|---|---|---|
| `alignment_perturbation_bound` | 4 | `1 − A ≤ C·‖E‖` when downstream `= c·I_V + E` |
| `scff_alignment_at_init` | 5 | assemble: `1 − A^(ℓ) ≤ C/√n + C'·δ` |
