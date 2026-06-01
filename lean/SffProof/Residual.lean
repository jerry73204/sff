/-
Residual-regime isotropy (method revision, Track E result formalized).

The empirical sweep found that **residual** skip connections lift the SCFF alignment ceiling:
the downstream Jacobian becomes `M = I + αΣJ ≈ I`, so `MᵀM` is near-identity by construction —
isotropy *without* the random-matrix / large-width input. The measured law was
`1 − A ≈ Aniso ≈ O(α)`.

This file proves that mechanism deterministically. In a normed `⋆`-ring `R` (matrices with the
Frobenius norm and transpose, or any C*-algebra), if `M = 1 + E` then

    ‖Mᵀ M − 1‖ ≤ 2‖E‖ + ‖E‖²,

and in the small-error regime `‖E‖ ≤ 1` this is `≤ 3‖E‖` — *linear* in the perturbation,
matching the empirical `Aniso = O(α)`. No randomness, no `o(√n)` constraint: the deviation of
the Gram operator from the identity is controlled entirely by `‖E‖` (the residual scale `α`).

This complements `gram_subspace_isotropy_bound` (the wide-random regime, `≤ d√(K/n)`): residual
gives the same isotropy conclusion deterministically, with `K` replaced by the residual scale.
-/
import Mathlib.Analysis.CStarAlgebra.Basic
import Mathlib.Tactic.NoncommRing

namespace SffProof

variable {R : Type*} [NormedRing R] [StarRing R] [NormedStarGroup R]

/-- **Residual isotropy (deterministic).** If the downstream operator is a perturbation of the
identity, `M = 1 + E`, then its Gram operator deviates from the identity by at most
`2‖E‖ + ‖E‖²`:

`‖Mᵀ M − 1‖ ≤ 2‖E‖ + ‖E‖²`.

`Mᵀ M − 1 = Eᵀ E + E + Eᵀ`, then the triangle inequality, submultiplicativity, and
`‖Eᵀ‖ = ‖E‖`. -/
theorem residual_isotropy (M E : R) (hM : M = 1 + E) :
    ‖star M * M - 1‖ ≤ 2 * ‖E‖ + ‖E‖ ^ 2 := by
  have hexp : star M * M - 1 = star E * E + E + star E := by
    subst hM; rw [star_add, star_one]; noncomm_ring
  have h1 : ‖star E * E‖ ≤ ‖E‖ * ‖E‖ := by
    have h := norm_mul_le (star E) E
    rwa [norm_star] at h
  rw [hexp]
  calc ‖star E * E + E + star E‖
      ≤ ‖star E * E‖ + ‖E‖ + ‖star E‖ := norm_add₃_le
    _ ≤ ‖E‖ * ‖E‖ + ‖E‖ + ‖E‖ := by rw [norm_star]; gcongr
    _ = 2 * ‖E‖ + ‖E‖ ^ 2 := by ring

/-- **Residual isotropy, linear form.** In the small-residual regime `‖E‖ ≤ 1`, the deviation
is *linear* in the perturbation: `‖Mᵀ M − 1‖ ≤ 3‖E‖`. This is the deterministic analogue of
the measured `Aniso = O(α)` law (with `‖E‖ = O(α)` the residual scale). -/
theorem residual_isotropy_linear (M E : R) (hM : M = 1 + E) (hE : ‖E‖ ≤ 1) :
    ‖star M * M - 1‖ ≤ 3 * ‖E‖ := by
  refine (residual_isotropy M E hM).trans ?_
  have : ‖E‖ ^ 2 ≤ ‖E‖ := by
    rw [sq]; exact mul_le_of_le_one_right (norm_nonneg _) hE
  linarith

end SffProof
