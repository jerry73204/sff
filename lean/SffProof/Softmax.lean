/-
Deterministic glue for the `gram_match` hypothesis (design.md §3.1, Layer 2).

`gram_match` assumes the layer-ℓ and final-layer InfoNCE softmax weights are close
(`‖p^(ℓ) − p^(L)‖ ≤ δ`). This file proves the *deterministic* half of that reduction:
**softmax is stable under coordinatewise logit perturbations**, with an elementary
`e^{2ε}` constant and no calculus. It reduces `gram_match` to Gram-matrix closeness
(the remaining random-matrix input).

`softmax` here is the pure logit→weights map; it specializes `Defs.softmaxWeights` with
logits `= ⟨z_i, z_j⟩ / τ`.
-/
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic

open scoped BigOperators

namespace SffProof

variable {B : ℕ} [NeZero B]

/-- Softmax of a logit vector: `softmax a i = exp(a_i) / Σ_k exp(a_k)`. -/
noncomputable def softmax (a : Fin B → ℝ) (i : Fin B) : ℝ :=
  Real.exp (a i) / ∑ k, Real.exp (a k)

theorem softmax_denom_pos (a : Fin B → ℝ) : 0 < ∑ k, Real.exp (a k) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) (Finset.univ_nonempty)

theorem softmax_nonneg (a : Fin B → ℝ) (i : Fin B) : 0 ≤ softmax a i :=
  div_nonneg (Real.exp_pos _).le (softmax_denom_pos a).le

/-- Softmax weights sum to 1 (a probability distribution over the batch). -/
theorem softmax_sum_eq_one (a : Fin B → ℝ) : ∑ i, softmax a i = 1 := by
  simp only [softmax, div_eq_mul_inv, ← Finset.sum_mul]
  rw [mul_inv_cancel₀ (softmax_denom_pos a).ne']

/-- **Softmax stability.** If logits agree coordinatewise to within `ε`, then each softmax
weight changes by at most `(e^{2ε} − 1)` times the (reference) weight. Deterministic;
the reduction of `gram_match` to Gram closeness. -/
theorem softmax_sub_abs_le {a b : Fin B → ℝ} {ε : ℝ}
    (h : ∀ k, |a k - b k| ≤ ε) (i : Fin B) :
    |softmax a i - softmax b i| ≤ (Real.exp (2 * ε) - 1) * softmax b i := by
  have hZb := softmax_denom_pos b
  have hZa := softmax_denom_pos a
  -- denominator two-sided bound: e^{−ε} Z_b ≤ Z_a ≤ e^{ε} Z_b
  have hZ_ge : Real.exp (-ε) * (∑ k, Real.exp (b k)) ≤ ∑ k, Real.exp (a k) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum (fun k _ => ?_)
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by have := (abs_le.mp (h k)).1; linarith)
  have hZ_le : (∑ k, Real.exp (a k)) ≤ Real.exp ε * (∑ k, Real.exp (b k)) := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum (fun k _ => ?_)
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by have := (abs_le.mp (h k)).2; linarith)
  -- numerator two-sided bound
  have hn_le : Real.exp (a i) ≤ Real.exp ε * Real.exp (b i) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by have := (abs_le.mp (h i)).2; linarith)
  have hn_ge : Real.exp (-ε) * Real.exp (b i) ≤ Real.exp (a i) := by
    rw [← Real.exp_add]
    exact Real.exp_le_exp.mpr (by have := (abs_le.mp (h i)).1; linarith)
  -- upper: softmax a i ≤ e^{2ε} softmax b i
  have hub : softmax a i ≤ Real.exp (2 * ε) * softmax b i := by
    have e2 : Real.exp (2 * ε) = Real.exp ε * Real.exp ε := by rw [← Real.exp_add]; ring_nf
    have hZbe : (∑ k, Real.exp (b k)) ≤ Real.exp ε * (∑ k, Real.exp (a k)) := by
      have hmul := mul_le_mul_of_nonneg_left hZ_ge (Real.exp_pos ε).le
      rwa [← mul_assoc, ← Real.exp_add, add_neg_cancel, Real.exp_zero, one_mul] at hmul
    rw [softmax, softmax, ← mul_div_assoc, div_le_div_iff₀ hZa hZb, e2]
    calc Real.exp (a i) * (∑ k, Real.exp (b k))
        ≤ (Real.exp ε * Real.exp (b i)) * (∑ k, Real.exp (b k)) :=
          mul_le_mul_of_nonneg_right hn_le hZb.le
      _ ≤ (Real.exp ε * Real.exp (b i)) * (Real.exp ε * (∑ k, Real.exp (a k))) :=
          mul_le_mul_of_nonneg_left hZbe (by positivity)
      _ = Real.exp ε * Real.exp ε * Real.exp (b i) * (∑ k, Real.exp (a k)) := by ring
  -- lower: e^{−2ε} softmax b i ≤ softmax a i
  have hlb : Real.exp (-(2 * ε)) * softmax b i ≤ softmax a i := by
    have e2 : Real.exp (-(2 * ε)) = Real.exp (-ε) * Real.exp (-ε) := by
      rw [← Real.exp_add]; ring_nf
    have e3 : Real.exp (-ε) * Real.exp ε = 1 := by rw [← Real.exp_add]; simp
    rw [softmax, softmax, ← mul_div_assoc, div_le_div_iff₀ hZb hZa, e2]
    calc Real.exp (-ε) * Real.exp (-ε) * Real.exp (b i) * (∑ k, Real.exp (a k))
        ≤ Real.exp (-ε) * Real.exp (a i) * (∑ k, Real.exp (a k)) := by
          refine mul_le_mul_of_nonneg_right ?_ hZa.le
          calc Real.exp (-ε) * Real.exp (-ε) * Real.exp (b i)
              = Real.exp (-ε) * (Real.exp (-ε) * Real.exp (b i)) := by ring
            _ ≤ Real.exp (-ε) * Real.exp (a i) :=
                mul_le_mul_of_nonneg_left hn_ge (Real.exp_pos (-ε)).le
      _ ≤ Real.exp (-ε) * Real.exp (a i) * (Real.exp ε * (∑ k, Real.exp (b k))) :=
          mul_le_mul_of_nonneg_left hZ_le (by positivity)
      _ = Real.exp (a i) * (∑ k, Real.exp (b k)) := by
          rw [show Real.exp (-ε) * Real.exp (a i) * (Real.exp ε * (∑ k, Real.exp (b k)))
                = (Real.exp (-ε) * Real.exp ε) * (Real.exp (a i) * (∑ k, Real.exp (b k))) by ring,
            e3, one_mul]
  -- AM–GM: e^{2ε} + e^{−2ε} ≥ 2
  have hamgm : 0 ≤ Real.exp (2 * ε) + Real.exp (-(2 * ε)) - 2 := by
    have e1 : Real.exp (2 * ε) = Real.exp ε * Real.exp ε := by rw [← Real.exp_add]; ring_nf
    have e2 : Real.exp (-(2 * ε)) = Real.exp (-ε) * Real.exp (-ε) := by
      rw [← Real.exp_add]; ring_nf
    have e3 : Real.exp ε * Real.exp (-ε) = 1 := by rw [← Real.exp_add]; simp
    nlinarith [sq_nonneg (Real.exp ε - Real.exp (-ε)), e1, e2, e3]
  rw [abs_le]
  refine ⟨?_, ?_⟩
  · nlinarith [hlb, mul_nonneg hamgm (softmax_nonneg b i)]
  · nlinarith [hub, softmax_nonneg b i]

/-- **ℓ¹ softmax stability.** Total-variation-style bound: if logits agree coordinatewise
to within `ε`, the softmax distributions differ in ℓ¹ by at most `e^{2ε} − 1`. This is the
deterministic reduction of `gram_match`: `‖p^(ℓ) − p^(L)‖₁ ≤ e^{2ε} − 1` with
`ε = ‖scores^(ℓ) − scores^(L)‖_∞`, leaving only Gram closeness (`ε → 0`) as random input. -/
theorem softmax_l1_sub_le {a b : Fin B → ℝ} {ε : ℝ} (h : ∀ k, |a k - b k| ≤ ε) :
    ∑ i, |softmax a i - softmax b i| ≤ Real.exp (2 * ε) - 1 := by
  calc ∑ i, |softmax a i - softmax b i|
      ≤ ∑ i, (Real.exp (2 * ε) - 1) * softmax b i :=
        Finset.sum_le_sum (fun i _ => softmax_sub_abs_le h i)
    _ = (Real.exp (2 * ε) - 1) * ∑ i, softmax b i := by rw [← Finset.mul_sum]
    _ = Real.exp (2 * ε) - 1 := by rw [softmax_sum_eq_one, mul_one]

/-- **Linearized softmax stability.** For logit perturbations bounded by `ε ≤ M`, the ℓ¹
softmax difference is *linear* in `ε`: `‖p(a) − p(b)‖₁ ≤ 2 e^{2M}·ε`. This makes the
reduction of `gram_match` to logit (Gram) closeness linear: a bound `ε ≤ K/√n` on the
score difference yields `‖p^(ℓ) − p^(L)‖₁ ≤ C/√n`. The remaining input — closeness of the
unnormalized Gram entries `x_iᵀWᵀWx_j` across layers — is the same quadratic-form
concentration as `gram_subspace_entry_sq`. -/
theorem softmax_l1_le_linear {a b : Fin B → ℝ} {ε M : ℝ} (hε0 : 0 ≤ ε) (hεM : ε ≤ M)
    (h : ∀ k, |a k - b k| ≤ ε) :
    ∑ i, |softmax a i - softmax b i| ≤ 2 * Real.exp (2 * M) * ε := by
  refine (softmax_l1_sub_le h).trans ?_
  have he3 : Real.exp (2 * ε) * Real.exp (-(2 * ε)) = 1 := by rw [← Real.exp_add]; simp
  have hx : 1 - 2 * ε ≤ Real.exp (-(2 * ε)) := by
    have := Real.add_one_le_exp (-(2 * ε)); linarith
  have hstep1 : Real.exp (2 * ε) - 1 ≤ 2 * ε * Real.exp (2 * ε) := by
    nlinarith [mul_le_mul_of_nonneg_left hx (Real.exp_pos (2 * ε)).le, he3]
  calc Real.exp (2 * ε) - 1 ≤ 2 * ε * Real.exp (2 * ε) := hstep1
    _ ≤ 2 * ε * Real.exp (2 * M) :=
        mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) (by linarith)
    _ = 2 * Real.exp (2 * M) * ε := by ring

end SffProof
