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

/-- **Centered square-sum engine.** For a pairwise-independent family `U i` with `U i ∈ L⁴`,
common second moment `α = E[U i²]` and fourth moment `β = E[U i⁴]`, the centered sum of
squares has second moment `|s|·(β − α²)`:

    E[(Σ_i (U i² − α))²] = |s| · (β − α²).

Each centered square `U i² − α` is mean-zero, square-integrable and independent across `i`
(functions of the independent `U i`), so the workhorse `sq_integral_sum_eq` applies; the
per-term second moment is `Var[U i²] = E[U i⁴] − (E[U i²])² = β − α²`. This is the shared
engine for the diagonal Gram entry and the rank-1 subspace restriction. -/
theorem centered_sq_sum_eq {s : Finset ι} {U : ι → Ω → ℝ} {α β : ℝ}
    (hmemSq : ∀ i ∈ s, MemLp (fun ω => (U i ω) ^ 2) 2 μ)
    (hindep : Set.Pairwise (s : Set ι) fun i j => IndepFun (U i) (U j) μ)
    (hsq : ∀ i ∈ s, μ[fun ω => (U i ω) ^ 2] = α)
    (hfour : ∀ i ∈ s, μ[fun ω => (U i ω) ^ 4] = β) :
    ∫ ω, (∑ i ∈ s, ((U i ω) ^ 2 - α)) ^ 2 ∂μ = (s.card : ℝ) * (β - α ^ 2) := by
  have hpair : Set.Pairwise (s : Set ι)
      fun i j => IndepFun (fun ω => (U i ω) ^ 2 - α) (fun ω => (U j ω) ^ 2 - α) μ := by
    intro i hi j hj hij
    exact (hindep hi hj hij).comp (φ := fun x : ℝ => x ^ 2 - α)
      (ψ := fun x : ℝ => x ^ 2 - α) (by fun_prop) (by fun_prop)
  have hmem : ∀ i ∈ s, MemLp (fun ω => (U i ω) ^ 2 - α) 2 μ :=
    fun i hi => (hmemSq i hi).sub (memLp_const α)
  have hmean : ∀ i ∈ s, μ[fun ω => (U i ω) ^ 2 - α] = 0 := by
    intro i hi
    rw [integral_sub ((hmemSq i hi).integrable one_le_two) (integrable_const α),
      integral_const, hsq i hi]
    simp
  have hXsq : ∀ i ∈ s, ∫ ω, ((U i ω) ^ 2 - α) ^ 2 ∂μ = β - α ^ 2 := by
    intro i hi
    have haem : AEStronglyMeasurable (fun ω => (U i ω) ^ 2) μ :=
      (hmemSq i hi).aestronglyMeasurable
    have hpow : (fun ω => (U i ω) ^ 2) ^ 2 = fun ω => (U i ω) ^ 4 := by
      funext ω; simp only [Pi.pow_apply]; ring
    rw [show (∫ ω, ((U i ω) ^ 2 - α) ^ 2 ∂μ) = Var[fun ω => (U i ω) ^ 2 - α; μ] from
        (variance_of_integral_eq_zero (hmem i hi).aemeasurable (hmean i hi)).symm,
      variance_sub_const haem, variance_eq_sub (hmemSq i hi), hsq i hi, hpow, hfour i hi]
  rw [sq_integral_sum_eq hmem hmean hpair, Finset.sum_congr rfl hXsq, Finset.sum_const,
    nsmul_eq_mul]

end SffProof
