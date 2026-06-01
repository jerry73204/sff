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

/-- **Centered sum, variance-additive (general engine).** For a pairwise-independent family
`W i` (square-integrable) with common mean `α = E[W i]` and second moment `γ = E[W i²]`, the
centered sum has second moment `|s|·(γ − α²)`:

    E[(Σ_i (W i − α))²] = |s| · (γ − α²).

Generalizes `centered_sq_sum_eq` (take `W i = U i²`): the summands need not be squares, only
independent across `i`. Used for the off-diagonal-across-directions Gram entries
`⟨row,v_a⟩⟨row,v_b⟩`. -/
theorem centered_sum_var_eq {s : Finset ι} {W : ι → Ω → ℝ} {α γ : ℝ}
    (hmem : ∀ i ∈ s, MemLp (W i) 2 μ)
    (hindep : Set.Pairwise (s : Set ι) fun i j => IndepFun (W i) (W j) μ)
    (hmean : ∀ i ∈ s, μ[W i] = α)
    (hsq : ∀ i ∈ s, μ[fun ω => (W i ω) ^ 2] = γ) :
    ∫ ω, (∑ i ∈ s, (W i ω - α)) ^ 2 ∂μ = (s.card : ℝ) * (γ - α ^ 2) := by
  have hpair : Set.Pairwise (s : Set ι)
      fun i j => IndepFun (fun ω => W i ω - α) (fun ω => W j ω - α) μ := by
    intro i hi j hj hij
    exact (hindep hi hj hij).comp (φ := fun x : ℝ => x - α)
      (ψ := fun x : ℝ => x - α) (by fun_prop) (by fun_prop)
  have hmemZ : ∀ i ∈ s, MemLp (fun ω => W i ω - α) 2 μ :=
    fun i hi => (hmem i hi).sub (memLp_const α)
  have hmeanZ : ∀ i ∈ s, μ[fun ω => W i ω - α] = 0 := by
    intro i hi
    rw [integral_sub ((hmem i hi).integrable one_le_two) (integrable_const α),
      integral_const, hmean i hi]
    simp
  have hZsq : ∀ i ∈ s, ∫ ω, (W i ω - α) ^ 2 ∂μ = γ - α ^ 2 := by
    intro i hi
    have haem : AEStronglyMeasurable (W i) μ := (hmem i hi).aestronglyMeasurable
    have hv : Var[W i; μ] = γ - α ^ 2 := by
      rw [variance_eq_sub (hmem i hi), hmean i hi]; congr 1; exact hsq i hi
    rw [show (∫ ω, (W i ω - α) ^ 2 ∂μ) = Var[fun ω => W i ω - α; μ] from
        (variance_of_integral_eq_zero (hmemZ i hi).aemeasurable (hmeanZ i hi)).symm,
      variance_sub_const haem, hv]
  rw [sq_integral_sum_eq hmemZ hmeanZ hpair, Finset.sum_congr rfl hZsq, Finset.sum_const,
    nsmul_eq_mul]

/-- **Jensen / L¹ ≤ L².** On a probability space, the mean absolute value is at most the
root mean square: `E|D| ≤ √(E[D²])`. Follows from `0 ≤ Var|D| = E[D²] − (E|D|)²`. Turns a
second-moment bound into a first-moment (expected-deviation) bound. -/
theorem integral_abs_le_sqrt_integral_sq {D : Ω → ℝ} (hD : MemLp D 2 μ) :
    ∫ ω, |D ω| ∂μ ≤ Real.sqrt (∫ ω, (D ω) ^ 2 ∂μ) := by
  have hg : MemLp (fun ω => |D ω|) 2 μ := by
    have h := hD.norm; simpa only [Real.norm_eq_abs] using h
  have h := variance_nonneg (fun ω => |D ω|) μ
  rw [variance_eq_sub hg] at h
  have hg2 : μ[(fun ω => |D ω|) ^ 2] = ∫ ω, (D ω) ^ 2 ∂μ := by
    simp only [Pi.pow_apply]
    refine integral_congr_ae (Filter.Eventually.of_forall fun ω => ?_)
    simp only [sq_abs]
  rw [hg2] at h
  have hnn : 0 ≤ ∫ ω, |D ω| ∂μ := integral_nonneg (fun ω => abs_nonneg _)
  have hle : (∫ ω, |D ω| ∂μ) ^ 2 ≤ ∫ ω, (D ω) ^ 2 ∂μ := by nlinarith [h]
  calc ∫ ω, |D ω| ∂μ = Real.sqrt ((∫ ω, |D ω| ∂μ) ^ 2) := (Real.sqrt_sq hnn).symm
    _ ≤ Real.sqrt (∫ ω, (D ω) ^ 2 ∂μ) := Real.sqrt_le_sqrt hle

/-- **Jensen for `√` (concavity).** For an integrable nonnegative `F`, the mean of `√F` is at
most `√` of the mean: `E[√F] ≤ √(E[F])`. Used to turn a Frobenius second-moment bound into an
expected-operator-norm bound. -/
theorem integral_sqrt_le_sqrt_integral {F : Ω → ℝ} (hF : Integrable F μ)
    (hFnn : 0 ≤ᵐ[μ] F) :
    ∫ ω, Real.sqrt (F ω) ∂μ ≤ Real.sqrt (∫ ω, F ω ∂μ) := by
  have hYaesm : AEStronglyMeasurable (fun ω => Real.sqrt (F ω)) μ :=
    Real.continuous_sqrt.comp_aestronglyMeasurable hF.aestronglyMeasurable
  have hYmem : MemLp (fun ω => Real.sqrt (F ω)) 2 μ := by
    rw [memLp_two_iff_integrable_sq hYaesm]
    refine hF.congr ?_
    filter_upwards [hFnn] with ω hω
    rw [Real.sq_sqrt hω]
  have h := integral_abs_le_sqrt_integral_sq hYmem
  have e1 : ∫ ω, |Real.sqrt (F ω)| ∂μ = ∫ ω, Real.sqrt (F ω) ∂μ :=
    integral_congr_ae (by filter_upwards with ω; exact abs_of_nonneg (Real.sqrt_nonneg _))
  have e2 : ∫ ω, (Real.sqrt (F ω)) ^ 2 ∂μ = ∫ ω, F ω ∂μ :=
    integral_congr_ae (by filter_upwards [hFnn] with ω hω; exact Real.sq_sqrt hω)
  rwa [e1, e2] at h

end SffProof
