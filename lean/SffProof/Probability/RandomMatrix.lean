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
and bounded `m₄` this is `O(1/n)`. An instance of the centered square-sum engine with
`U k = R_kp` (the `k`-th entry of column `p`), independent across rows. -/
theorem gram_diag_centered_sq {p : Fin n} :
    ∫ ω, (∑ k, ((Ens.R k ω p) ^ 2 - σ2)) ^ 2 ∂μ = (n : ℝ) * (m4 - σ2 ^ 2) := by
  have hU_indep : iIndepFun (fun k ω => Ens.R k ω p) μ :=
    Ens.indep.comp (fun _ (v : Fin n → ℝ) => v p) (fun _ => measurable_pi_apply p)
  have hpair : Set.Pairwise (↑(Finset.univ : Finset (Fin n)))
      fun i j => IndepFun (fun ω => Ens.R i ω p) (fun ω => Ens.R j ω p) μ :=
    fun i _ j _ hij => hU_indep.indepFun hij
  rw [centered_sq_sum_eq (fun k _ => Ens.entry_sq_memLp k p) hpair
      (fun k _ => Ens.entry_sq k p) (fun k _ => Ens.entry_four k p),
    Finset.card_univ, Fintype.card_fin]

/-- **Rank-1 subspace restriction.** Along a single direction `v`, the quadratic form
`vᵀWᵀWv = ‖Wv‖² = Σ_k ⟨row_k, v⟩²` is a sum over independent rows, so its centered second
moment is `n·(β_v − α_v²)`, where `α_v = E[⟨row_k,v⟩²]` and `β_v = E[⟨row_k,v⟩⁴]` are the
projected moments. With `α_v = σ²‖v‖²` this bounds `E[(vᵀ(WᵀW)v − n α_v)²]`, the isotropy
deviation along `v` — the `d_V = 1` case of `isotropy_at_init`. The projected moments are
supplied as hypotheses (they are within-row joint moments of the entries). -/
theorem gram_rank1_centered_sq (v : Fin n → ℝ) {αv βv : ℝ}
    (hmemSq : ∀ k, MemLp (fun ω => (∑ p, Ens.R k ω p * v p) ^ 2) 2 μ)
    (hsq : ∀ k, μ[fun ω => (∑ p, Ens.R k ω p * v p) ^ 2] = αv)
    (hfour : ∀ k, μ[fun ω => (∑ p, Ens.R k ω p * v p) ^ 4] = βv) :
    ∫ ω, (∑ k, ((∑ p, Ens.R k ω p * v p) ^ 2 - αv)) ^ 2 ∂μ = (n : ℝ) * (βv - αv ^ 2) := by
  have hU_indep : iIndepFun (fun k ω => ∑ p, Ens.R k ω p * v p) μ :=
    Ens.indep.comp (fun _ (r : Fin n → ℝ) => ∑ p, r p * v p) (fun _ => by fun_prop)
  have hpair : Set.Pairwise (↑(Finset.univ : Finset (Fin n)))
      fun i j => IndepFun (fun ω => ∑ p, Ens.R i ω p * v p)
                          (fun ω => ∑ p, Ens.R j ω p * v p) μ :=
    fun i _ j _ hij => hU_indep.indepFun hij
  rw [centered_sq_sum_eq (fun k _ => hmemSq k) hpair (fun k _ => hsq k) (fun k _ => hfour k),
    Finset.card_univ, Fintype.card_fin]

/-- **Rank-1 isotropy — expected deviation (Jensen capstone).** Combining the rank-1 second
moment with `integral_abs_le_sqrt_integral_sq`, the *expected absolute* isotropy deviation
along `v` is `E|vᵀWᵀWv − n α_v| ≤ √(n·(β_v − α_v²))`. -/
theorem gram_rank1_abs_le (v : Fin n → ℝ) {αv βv : ℝ}
    (hmemSq : ∀ k, MemLp (fun ω => (∑ p, Ens.R k ω p * v p) ^ 2) 2 μ)
    (hsq : ∀ k, μ[fun ω => (∑ p, Ens.R k ω p * v p) ^ 2] = αv)
    (hfour : ∀ k, μ[fun ω => (∑ p, Ens.R k ω p * v p) ^ 4] = βv) :
    ∫ ω, |∑ k, ((∑ p, Ens.R k ω p * v p) ^ 2 - αv)| ∂μ
      ≤ Real.sqrt ((n : ℝ) * (βv - αv ^ 2)) := by
  have hD : MemLp (fun ω => ∑ k, ((∑ p, Ens.R k ω p * v p) ^ 2 - αv)) 2 μ :=
    memLp_finsetSum _ fun k _ => (hmemSq k).sub (memLp_const αv)
  have h := integral_abs_le_sqrt_integral_sq hD
  rwa [gram_rank1_centered_sq Ens v hmemSq hsq hfour] at h

/-- **Rank-1 `isotropy_at_init`, expectation form.** If the projected variance gap obeys
`β_v − α_v² ≤ K/n²` (true with `α_v = σ²‖v‖²`, `σ² = 1/n` and bounded moments), the expected
isotropy deviation along `v` is `O(1/√n)`:

    E|vᵀWᵀWv − n α_v| ≤ √(K/n).

This is the `d_V = 1` case of `isotropy_at_init` (`‖errIso‖ ≤ K/√n`) discharged in
expectation, modulo the projected/4th-moment inputs. -/
theorem gram_rank1_isotropy_bound (v : Fin n → ℝ) {αv βv K : ℝ}
    (hmemSq : ∀ k, MemLp (fun ω => (∑ p, Ens.R k ω p * v p) ^ 2) 2 μ)
    (hsq : ∀ k, μ[fun ω => (∑ p, Ens.R k ω p * v p) ^ 2] = αv)
    (hfour : ∀ k, μ[fun ω => (∑ p, Ens.R k ω p * v p) ^ 4] = βv)
    (hn : 0 < n) (hK : βv - αv ^ 2 ≤ K / (n : ℝ) ^ 2) :
    ∫ ω, |∑ k, ((∑ p, Ens.R k ω p * v p) ^ 2 - αv)| ∂μ ≤ Real.sqrt (K / (n : ℝ)) := by
  refine (gram_rank1_abs_le Ens v hmemSq hsq hfour).trans (Real.sqrt_le_sqrt ?_)
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  calc (n : ℝ) * (βv - αv ^ 2) ≤ (n : ℝ) * (K / (n : ℝ) ^ 2) :=
        mul_le_mul_of_nonneg_left hK hnpos.le
    _ = K / (n : ℝ) := by field_simp

end RandomMatrixEnsemble

end SffProof
