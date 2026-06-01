/-
Normalization (design.md §1: `z = y/‖y‖`).

The normalized Gram entry `⟨z_i,z_j⟩ = ⟨y_i,y_j⟩ / (‖y_i‖·‖y_j‖)` is a *ratio* of quantities
that concentrate (numerator `⟨y_i,y_j⟩ = x_iᵀWᵀWx_j` and denominator `‖y_i‖‖y_j‖`, both handled
by `gram_entry_abs_le`). This file proves the **deterministic** lifting: a ratio of two
near-constant quantities is near the ratio of the constants. Combined with the numerator /
denominator concentration this gives normalized-Gram concentration (`z`-kernel preservation).

Pure real analysis, no probability.
-/
import Mathlib.Tactic

namespace SffProof

/-- **Ratio perturbation.** If the denominator `b` stays within a factor 2 of a positive
target `b̄` (`b̄/2 ≤ b`), the ratio `a/b` deviates from `ā/b̄` by at most a linear combination
of the numerator and denominator deviations:

    |a/b − ā/b̄| ≤ (2/b̄)·|a − ā| + (2|ā|/b̄²)·|b − b̄|.

This lifts numerator/denominator concentration to ratio concentration — the deterministic
heart of normalization (`z = y/‖y‖`). -/
theorem ratio_perturbation {a b abar bbar : ℝ} (hbbar : 0 < bbar) (hb : bbar / 2 ≤ b) :
    |a / b - abar / bbar| ≤ (2 / bbar) * |a - abar| + (2 * |abar| / bbar ^ 2) * |b - bbar| := by
  have hb0 : 0 < b := lt_of_lt_of_le (by positivity) hb
  have hbb : 0 < b * bbar := mul_pos hb0 hbbar
  have h1b : 1 / b ≤ 2 / bbar := by rw [div_le_div_iff₀ hb0 hbbar]; linarith
  have h2b : 1 / (b * bbar) ≤ 2 / bbar ^ 2 := by
    rw [div_le_div_iff₀ hbb (by positivity)]; nlinarith
  have hid : a / b - abar / bbar = (a - abar) / b + abar * (bbar - b) / (b * bbar) := by
    field_simp; ring
  have t1 : |a - abar| / b ≤ (2 / bbar) * |a - abar| := by
    rw [div_eq_mul_inv, mul_comm, ← one_div]
    exact mul_le_mul_of_nonneg_right h1b (abs_nonneg _)
  have t2 : |abar| * |b - bbar| / (b * bbar) ≤ (2 * |abar| / bbar ^ 2) * |b - bbar| := by
    rw [div_eq_mul_inv]
    calc |abar| * |b - bbar| * (b * bbar)⁻¹
        ≤ |abar| * |b - bbar| * (2 / bbar ^ 2) :=
          mul_le_mul_of_nonneg_left (by rw [← one_div]; exact h2b) (by positivity)
      _ = (2 * |abar| / bbar ^ 2) * |b - bbar| := by ring
  rw [hid]
  refine (abs_add_le _ _).trans ?_
  rw [abs_div, abs_of_pos hb0, abs_div, abs_of_pos hbb, abs_mul, abs_sub_comm bbar b]
  exact add_le_add t1 t2

/-- **Normalized inner product = ratio.** `⟨z_i,z_j⟩` with `z = y/‖y‖` unfolds to the ratio
`⟨y_i,y_j⟩ / (‖y_i‖·‖y_j‖)`, to which `ratio_perturbation` applies with numerator
`a = ⟨y_i,y_j⟩` and denominator `b = ‖y_i‖·‖y_j‖`. -/
theorem normalized_gram_perturbation {yij nij dbar : ℝ} {d : ℝ}
    (hdbar : 0 < dbar) (hd : dbar / 2 ≤ d) :
    |yij / d - nij / dbar|
      ≤ (2 / dbar) * |yij - nij| + (2 * |nij| / dbar ^ 2) * |d - dbar| :=
  ratio_perturbation hdbar hd

end SffProof
