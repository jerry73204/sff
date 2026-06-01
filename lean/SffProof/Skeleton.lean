/-
Layer 1 — Algebraic skeleton (design.md §3.1, obligations §3.2).
Deterministic finite-`n` facts. Fully proven, no `sorry`.

  1. inner_product_cosine_one_of_parallel  — warm-up (abstract inner product space).
  2. gradient_shared_right_factor           — structural: scaling left factors scales the grad.
  3. alignment_one_of_isotropic_and_matched — the heart: proportional left factors ⇒ A = 1.
-/
import SffProof.Defs

open scoped RealInnerProductSpace BigOperators

namespace SffProof

/-! ### Obligation 1 — abstract warm-up -/

section Abstract
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Cosine of the angle between two vectors of a real inner product space. -/
noncomputable def cosAngle (u v : E) : ℝ := ⟪u, v⟫ / (‖u‖ * ‖v‖)

/-- **Obligation 1.** `u = c • v`, `c > 0`, `v ≠ 0` ⇒ `cosAngle u v = 1`. -/
theorem inner_product_cosine_one_of_parallel
    {u v : E} {c : ℝ} (hc : 0 < c) (hv : v ≠ 0) (h : u = c • v) :
    cosAngle u v = 1 := by
  subst h
  have hnorm : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  rw [cosAngle, real_inner_smul_left, real_inner_self_eq_norm_mul_norm, norm_smul,
    Real.norm_eq_abs, abs_of_pos hc]
  field_simp

end Abstract

/-! ### Frobenius helper lemmas -/

variable {n m B : ℕ}

@[simp] theorem frob_smul_left (c : ℝ) (A C : Matrix (Fin n) (Fin m) ℝ) :
    frob (c • A) C = c * frob A C := by
  simp only [frob, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => Finset.sum_congr rfl (fun _ _ => by ring))

@[simp] theorem frob_smul_right (c : ℝ) (A C : Matrix (Fin n) (Fin m) ℝ) :
    frob A (c • C) = c * frob A C := by
  simp only [frob, Matrix.smul_apply, smul_eq_mul, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun _ _ => Finset.sum_congr rfl (fun _ _ => by ring))

theorem frob_self_nonneg (A : Matrix (Fin n) (Fin m) ℝ) : 0 ≤ frob A A := by
  refine Finset.sum_nonneg (fun i _ => Finset.sum_nonneg (fun j _ => ?_))
  exact mul_self_nonneg (A i j)

@[simp] theorem frobNorm_mul_self (A : Matrix (Fin n) (Fin m) ℝ) :
    frobNorm A * frobNorm A = frob A A := by
  rw [frobNorm]; exact Real.mul_self_sqrt (frob_self_nonneg A)

theorem frobNorm_smul (c : ℝ) (A : Matrix (Fin n) (Fin m) ℝ) :
    frobNorm (c • A) = |c| * frobNorm A := by
  rw [frobNorm, frobNorm, frob_smul_left, frob_smul_right]
  rw [show c * (c * frob A A) = c ^ 2 * frob A A by ring,
    Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq_eq_abs]

/-- The Frobenius cosine of parallel matrices (positive scale) is 1. -/
theorem cosAngleM_eq_one_of_parallel
    {A C : Matrix (Fin n) (Fin m) ℝ} {c : ℝ}
    (hc : 0 < c) (hC : frobNorm C ≠ 0) (h : A = c • C) :
    cosAngleM A C = 1 := by
  subst h
  have hfrob : frob C C = frobNorm C * frobNorm C := (frobNorm_mul_self C).symm
  rw [cosAngleM, frob_smul_left, frobNorm_smul, abs_of_pos hc, hfrob]
  field_simp

/-! ### Obligation 2 — structural lemma -/

/-- **Obligation 2 (gradient_shared_right_factor).** Local and global gradients share the
right factor `right`; scaling the left factors pointwise scales the assembled gradient.
With `leftGlobal i = c • leftLocal i` this makes `gradGlobal = c • gradLocal`. -/
theorem gradient_shared_right_factor
    (left : Fin B → Fin n → ℝ) (right : Fin B → Fin m → ℝ) (c : ℝ) :
    gradFromFactors (fun i => c • left i) right = c • gradFromFactors left right := by
  simp only [gradFromFactors, Finset.smul_sum]
  refine Finset.sum_congr rfl (fun i _ => ?_)
  ext a b
  simp [outer, Matrix.smul_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-! ### Obligation 3 — alignment heart -/

/-- **Obligation 3 (alignment_one_of_isotropic_and_matched).** If the global left factors
are a common positive multiple of the local ones — which is exactly what downstream
isotropy `M^T M|_V = c·I` together with matched softmax weights `p^(ℓ)=p^(L)` deliver —
then the alignment cosine is exactly 1. Pure linear algebra; the mathematical heart. -/
theorem alignment_one_of_isotropic_and_matched
    (leftLocal : Fin B → Fin n → ℝ) (right : Fin B → Fin m → ℝ) {c : ℝ}
    (hc : 0 < c)
    (hgrad : frobNorm (gradFromFactors leftLocal right) ≠ 0) :
    cosAngleM (gradFromFactors (fun i => c • leftLocal i) right)
              (gradFromFactors leftLocal right) = 1 := by
  apply cosAngleM_eq_one_of_parallel hc hgrad
  exact gradient_shared_right_factor leftLocal right c

end SffProof
