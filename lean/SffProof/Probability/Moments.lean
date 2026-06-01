/-
Probability scaffolding for discharging `isotropy_at_init` (design.md §3.1, Layer 2),
expectation mode.

The keystone (entrywise 2nd moment of `VᵀWᵀWV − I`) reduces to second moments of sums of
independent mean-zero random variables. This file proves the **workhorse**:

    E[(Σ_i X_i)²] = Σ_i E[X_i²]      for pairwise-independent, mean-zero, L² families.

Built from Mathlib's `IndepFun.variance_sum` (variance is additive over independent finite
sums) and `variance_of_integral_eq_zero` (mean 0 ⇒ Var = E[X²]). No random-matrix theory.
-/
import Mathlib.Probability.Moments.Variance
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory BigOperators

namespace SffProof

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {ι : Type*}

omit [IsProbabilityMeasure μ] in
/-- Mean of a finite sum of integrable, mean-zero variables is zero. -/
theorem integral_sum_eq_zero {s : Finset ι} {X : ι → Ω → ℝ}
    (hint : ∀ i ∈ s, Integrable (X i) μ) (hmean : ∀ i ∈ s, μ[X i] = 0) :
    μ[∑ i ∈ s, X i] = 0 := by
  simp only [Finset.sum_apply]
  rw [integral_finsetSum s hint]
  exact Finset.sum_eq_zero hmean

/-- **Workhorse: second moment of an independent mean-zero sum.**
`E[(Σ_i X_i)²] = Σ_i E[X_i²]` for a pairwise-independent, mean-zero, L² family. The cross
terms vanish (independence + zero mean); this is variance additivity in disguise. -/
theorem sq_integral_sum_eq {s : Finset ι} {X : ι → Ω → ℝ}
    (hmem : ∀ i ∈ s, MemLp (X i) 2 μ) (hmean : ∀ i ∈ s, μ[X i] = 0)
    (hindep : Set.Pairwise (s : Set ι) fun i j => IndepFun (X i) (X j) μ) :
    ∫ ω, (∑ i ∈ s, X i ω) ^ 2 ∂μ = ∑ i ∈ s, ∫ ω, (X i ω) ^ 2 ∂μ := by
  have hint : ∀ i ∈ s, Integrable (X i) μ := fun i hi => (hmem i hi).integrable one_le_two
  have hsum_mean : μ[∑ i ∈ s, X i] = 0 := integral_sum_eq_zero hint hmean
  have hsum_aem : AEMeasurable (∑ i ∈ s, X i) μ :=
    (Finset.aestronglyMeasurable_sum s fun i hi =>
      (hmem i hi).aestronglyMeasurable).aemeasurable
  -- both sides are variances; use additivity over the independent family
  have lhs_eq : ∫ ω, (∑ i ∈ s, X i ω) ^ 2 ∂μ = Var[∑ i ∈ s, X i; μ] := by
    rw [variance_of_integral_eq_zero hsum_aem hsum_mean]
    refine integral_congr_ae (Filter.Eventually.of_forall fun ω => ?_)
    simp only [Finset.sum_apply]
  rw [lhs_eq, IndepFun.variance_sum hmem hindep]
  refine Finset.sum_congr rfl fun i hi => ?_
  exact variance_of_integral_eq_zero (hmem i hi).aemeasurable (hmean i hi)

end SffProof
