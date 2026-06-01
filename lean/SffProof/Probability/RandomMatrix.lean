/-
Keystone for `isotropy_at_init` (design.md §3.1, Layer 2), expectation mode — linear /
one-layer / single random matrix `W`, scoped per design (linear mode is primary).

Random-matrix core: the entries of `WᵀW − I` have second moment `O(1/n)`. Here we prove the
**off-diagonal** entry, the case that uses the workhorse most directly:

    E[(WᵀW)_{pq}²] = n · σ⁴      (p ≠ q),

so with μP scaling `σ² = 1/n` this is `1/n`. `(WᵀW)_{pq} = Σ_k W_kp W_kq` is a sum over the
`n` rows; rows are independent, so the summands `W_kp W_kq` are independent and mean-zero,
and `sq_integral_sum_eq` (Probability/Moments) applies.

Rows are modelled as i.i.d. random vectors `R k : Ω → (Fin n → ℝ)`; the mixed within-row
moments (`E[R_kp R_kq] = 0`, `E[R_kp² R_kq²] = σ⁴` for `p ≠ q`) are ensemble hypotheses,
true for i.i.d. mean-zero entries. The substantive probabilistic step — independence of the
row products `W_kp W_kq` across `k` — is *derived* from row independence via `iIndepFun.comp`.
-/
import SffProof.Probability.Moments
import Mathlib.Probability.Independence.Basic

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory BigOperators

namespace SffProof

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]

/-- A random `n × n` matrix as `n` independent rows `R k : Ω → ℝⁿ`, with per-entry variance
`σ²` and the mixed within-row moments needed for the off-diagonal Gram entry. -/
structure RandomMatrixEnsemble (Ω : Type*) [MeasurableSpace Ω] (μ : Measure Ω)
    (n : ℕ) (σ2 m4 : ℝ) where
  /-- The rows; `R k ω p` is the `(k,p)` entry. -/
  R : Fin n → Ω → (Fin n → ℝ)
  /-- Rows are mutually independent. -/
  indep : iIndepFun R μ
  /-- Distinct entries in a row are uncorrelated and mean-zero ⇒ their product has mean 0. -/
  prod_mean : ∀ (k p q : Fin n), p ≠ q → μ[fun ω => R k ω p * R k ω q] = 0
  /-- Second moment of a distinct-entry product is `σ⁴`. -/
  prod_sq : ∀ (k p q : Fin n), p ≠ q → μ[fun ω => (R k ω p) ^ 2 * (R k ω q) ^ 2] = σ2 ^ 2
  /-- Distinct-entry products are square-integrable. -/
  prod_memLp : ∀ (k p q : Fin n), MemLp (fun ω => R k ω p * R k ω q) 2 μ
  /-- Per-entry second moment (variance) is `σ²`. -/
  entry_sq : ∀ (k p : Fin n), μ[fun ω => (R k ω p) ^ 2] = σ2
  /-- Per-entry fourth moment is `m₄`. -/
  entry_four : ∀ (k p : Fin n), μ[fun ω => (R k ω p) ^ 4] = m4
  /-- Squared entries are square-integrable (entries are `L⁴`). -/
  entry_sq_memLp : ∀ (k p : Fin n), MemLp (fun ω => (R k ω p) ^ 2) 2 μ

namespace RandomMatrixEnsemble

variable {n : ℕ} {σ2 m4 : ℝ} (Ens : RandomMatrixEnsemble Ω μ n σ2 m4)

/-- **Keystone (off-diagonal).** For `p ≠ q`, the second moment of the Gram entry
`(WᵀW)_{pq} = Σ_k R_kp R_kq` is `n · σ⁴`. With `σ² = 1/n` this is `1/n`. -/
theorem gram_offdiag_sq {p q : Fin n} (hpq : p ≠ q) :
    ∫ ω, (∑ k, Ens.R k ω p * Ens.R k ω q) ^ 2 ∂μ = (n : ℝ) * σ2 ^ 2 := by
  -- the summands, and their independence across rows (derived from row independence)
  have hX_indep : iIndepFun (fun k ω => Ens.R k ω p * Ens.R k ω q) μ :=
    Ens.indep.comp (fun _ (v : Fin n → ℝ) => v p * v q)
      (fun _ => (measurable_pi_apply p).mul (measurable_pi_apply q))
  have hpair : Set.Pairwise (↑(Finset.univ : Finset (Fin n)))
      fun i j => IndepFun (fun ω => Ens.R i ω p * Ens.R i ω q)
                          (fun ω => Ens.R j ω p * Ens.R j ω q) μ :=
    fun i _ j _ hij => hX_indep.indepFun hij
  have hmem : ∀ k ∈ (Finset.univ : Finset (Fin n)),
      MemLp (fun ω => Ens.R k ω p * Ens.R k ω q) 2 μ := fun k _ => Ens.prod_memLp k p q
  have hmean : ∀ k ∈ (Finset.univ : Finset (Fin n)),
      μ[fun ω => Ens.R k ω p * Ens.R k ω q] = 0 := fun k _ => Ens.prod_mean k p q hpq
  have hYsq : ∀ k ∈ (Finset.univ : Finset (Fin n)),
      ∫ ω, (Ens.R k ω p * Ens.R k ω q) ^ 2 ∂μ = σ2 ^ 2 := by
    intro k _
    simp only [mul_pow]
    exact Ens.prod_sq k p q hpq
  rw [sq_integral_sum_eq hmem hmean hpair, Finset.sum_congr rfl hYsq,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **Keystone (diagonal).** The centered diagonal Gram entry
`(WᵀW)_{pp} − n σ² = Σ_k (R_kp² − σ²)` has second moment `n·(m₄ − σ⁴)`. With `σ² = 1/n`
and bounded `m₄` this is `O(1/n)`. Needs the fourth moment and a centering step. -/
theorem gram_diag_centered_sq {p : Fin n} :
    ∫ ω, (∑ k, ((Ens.R k ω p) ^ 2 - σ2)) ^ 2 ∂μ = (n : ℝ) * (m4 - σ2 ^ 2) := by
  have hX_indep : iIndepFun (fun k ω => (Ens.R k ω p) ^ 2 - σ2) μ :=
    Ens.indep.comp (fun _ (v : Fin n → ℝ) => (v p) ^ 2 - σ2)
      (fun _ => by fun_prop)
  have hpair : Set.Pairwise (↑(Finset.univ : Finset (Fin n)))
      fun i j => IndepFun (fun ω => (Ens.R i ω p) ^ 2 - σ2)
                          (fun ω => (Ens.R j ω p) ^ 2 - σ2) μ :=
    fun i _ j _ hij => hX_indep.indepFun hij
  have hmem : ∀ k ∈ (Finset.univ : Finset (Fin n)),
      MemLp (fun ω => (Ens.R k ω p) ^ 2 - σ2) 2 μ :=
    fun k _ => (Ens.entry_sq_memLp k p).sub (memLp_const σ2)
  have hmean : ∀ k ∈ (Finset.univ : Finset (Fin n)),
      μ[fun ω => (Ens.R k ω p) ^ 2 - σ2] = 0 := by
    intro k _
    rw [integral_sub ((Ens.entry_sq_memLp k p).integrable one_le_two) (integrable_const σ2),
      integral_const, Ens.entry_sq k p]
    simp
  have hXsq : ∀ k ∈ (Finset.univ : Finset (Fin n)),
      ∫ ω, ((Ens.R k ω p) ^ 2 - σ2) ^ 2 ∂μ = m4 - σ2 ^ 2 := by
    intro k hk
    -- E[X_k²] = Var[X_k] = Var[R_kp²] = E[R_kp⁴] − (E[R_kp²])² = m₄ − σ⁴
    have haem : AEStronglyMeasurable (fun ω => (Ens.R k ω p) ^ 2) μ :=
      (Ens.entry_sq_memLp k p).aestronglyMeasurable
    have hpow : (fun ω => (Ens.R k ω p) ^ 2) ^ 2 = fun ω => (Ens.R k ω p) ^ 4 := by
      funext ω; simp only [Pi.pow_apply]; ring
    rw [show (∫ ω, ((Ens.R k ω p) ^ 2 - σ2) ^ 2 ∂μ)
          = Var[fun ω => (Ens.R k ω p) ^ 2 - σ2; μ] from
        (variance_of_integral_eq_zero (hmem k hk).aemeasurable (hmean k hk)).symm,
      variance_sub_const haem, variance_eq_sub (Ens.entry_sq_memLp k p), Ens.entry_sq k p, hpow,
      Ens.entry_four k p]
  rw [sq_integral_sum_eq hmem hmean hpair, Finset.sum_congr rfl hXsq,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

end RandomMatrixEnsemble

end SffProof
