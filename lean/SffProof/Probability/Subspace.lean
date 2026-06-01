/-
General `d_V`-dimensional subspace restriction for `isotropy_at_init` (design.md §3.1),
expectation mode, linear / single random matrix `W`.

The deviation operator on the contrastive subspace `V` (columns `V a`, `a : Fin d`) is the
`d × d` random matrix `D_{ab} = (VᵀW ᵀWV − E[·])_{ab} = Σ_k (⟨row_k,V a⟩⟨row_k,V b⟩ − α_{ab})`,
a sum over independent rows. Each entry's second moment is `n·(γ_{ab} − α_{ab}²)` via the
general engine `centered_sum_var_eq`; summing over the `d²` entries gives the Frobenius
second moment, and Jensen turns it into the expected operator deviation.

`α_{ab} = E[⟨row,V a⟩⟨row,V b⟩]` and `γ_{ab} = E[(⟨row,V a⟩⟨row,V b⟩)²]` are the within-row
joint moments (supplied as hypotheses; they hold for i.i.d. mean-zero entries).
-/
import SffProof.Probability.RandomMatrix

open MeasureTheory ProbabilityTheory
open scoped ProbabilityTheory BigOperators

namespace SffProof

namespace RandomMatrixEnsemble

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
variable {n : ℕ} {σ2 m4 : ℝ} (Ens : RandomMatrixEnsemble Ω μ n σ2 m4)

/-- The per-row product `⟨row_k, v_a⟩·⟨row_k, v_b⟩` (entry `(a,b)` summand). -/
private def prodEntry (va vb : Fin n → ℝ) (k : Fin n) (ω : Ω) : ℝ :=
  (∑ p, Ens.R k ω p * va p) * (∑ p, Ens.R k ω p * vb p)

/-- **General entry second moment.** For two directions, the `(a,b)` deviation entry
`Σ_k (⟨row_k,v_a⟩⟨row_k,v_b⟩ − α)` has second moment `n·(γ − α²)`. Same independent-row sum
as the rank-1 case, via `centered_sum_var_eq`. -/
theorem gram_subspace_entry_sq (va vb : Fin n → ℝ) {α γ : ℝ}
    (hmem : ∀ k, MemLp (Ens.prodEntry va vb k) 2 μ)
    (hmean : ∀ k, μ[Ens.prodEntry va vb k] = α)
    (hsq : ∀ k, μ[fun ω => (Ens.prodEntry va vb k ω) ^ 2] = γ) :
    ∫ ω, (∑ k, (Ens.prodEntry va vb k ω - α)) ^ 2 ∂μ = (n : ℝ) * (γ - α ^ 2) := by
  have hindep : iIndepFun (fun k => Ens.prodEntry va vb k) μ :=
    Ens.indep.comp (fun _ (r : Fin n → ℝ) => (∑ p, r p * va p) * (∑ p, r p * vb p))
      (fun _ => by fun_prop)
  have hpair : Set.Pairwise (↑(Finset.univ : Finset (Fin n)))
      fun i j => IndepFun (Ens.prodEntry va vb i) (Ens.prodEntry va vb j) μ :=
    fun i _ j _ hij => hindep.indepFun hij
  rw [centered_sum_var_eq (fun k _ => hmem k) hpair (fun k _ => hmean k) (fun k _ => hsq k),
    Finset.card_univ, Fintype.card_fin]

/-- MemLp of a deviation entry (a finite sum of `L²` centered products). -/
theorem entry_memLp (va vb : Fin n → ℝ) (α : ℝ)
    (hmem : ∀ k, MemLp (Ens.prodEntry va vb k) 2 μ) :
    MemLp (fun ω => ∑ k, (Ens.prodEntry va vb k ω - α)) 2 μ :=
  memLp_finsetSum _ fun k _ => (hmem k).sub (memLp_const α)

/-- **Gram-entry closeness (Jensen).** The expected absolute deviation of the unnormalized
Gram entry `⟨y_i,y_j⟩ = x_iᵀWᵀWx_j` (directions `va = x_i`, `vb = x_j`) from its mean is
`E|Σ_k(⟨row,va⟩⟨row,vb⟩ − α)| ≤ √(n·(γ − α²))`. -/
theorem gram_entry_abs_le (va vb : Fin n → ℝ) {α γ : ℝ}
    (hmem : ∀ k, MemLp (Ens.prodEntry va vb k) 2 μ)
    (hmean : ∀ k, μ[Ens.prodEntry va vb k] = α)
    (hsq : ∀ k, μ[fun ω => (Ens.prodEntry va vb k ω) ^ 2] = γ) :
    ∫ ω, |∑ k, (Ens.prodEntry va vb k ω - α)| ∂μ ≤ Real.sqrt ((n : ℝ) * (γ - α ^ 2)) := by
  have h := integral_abs_le_sqrt_integral_sq (Ens.entry_memLp va vb α hmem)
  rwa [Ens.gram_subspace_entry_sq va vb hmem hmean hsq] at h

/-- **Gram-entry closeness, `O(1/√n)` form.** With the entry variance gap `γ − α² ≤ K/n²`,
the unnormalized Gram entry concentrates to its mean at rate `O(1/√n)`:
`E|x_iᵀWᵀWx_j − α| ≤ √(K/n)`. This is the random core feeding the softmax linearization
(`softmax_l1_le_linear`): the score difference `ε = O(1/√n)` ⇒ `‖p^(ℓ) − p^(L)‖₁ = O(1/√n)`,
discharging `gram_match` in expectation (modulo normalization across layers). -/
theorem gram_entry_isotropy_bound (va vb : Fin n → ℝ) {α γ K : ℝ}
    (hmem : ∀ k, MemLp (Ens.prodEntry va vb k) 2 μ)
    (hmean : ∀ k, μ[Ens.prodEntry va vb k] = α)
    (hsq : ∀ k, μ[fun ω => (Ens.prodEntry va vb k ω) ^ 2] = γ)
    (hn : 0 < n) (hgap : γ - α ^ 2 ≤ K / (n : ℝ) ^ 2) :
    ∫ ω, |∑ k, (Ens.prodEntry va vb k ω - α)| ∂μ ≤ Real.sqrt (K / (n : ℝ)) := by
  refine (Ens.gram_entry_abs_le va vb hmem hmean hsq).trans (Real.sqrt_le_sqrt ?_)
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  calc (n : ℝ) * (γ - α ^ 2) ≤ (n : ℝ) * (K / (n : ℝ) ^ 2) :=
        mul_le_mul_of_nonneg_left hgap hnpos.le
    _ = K / (n : ℝ) := by field_simp

/-- **General Frobenius second moment.** `E‖VᵀM̃V‖_F² = Σ_{a,b} n·(γ_{ab} − α_{ab}²)`, the
sum of all `d²` entry second moments. -/
theorem gram_subspace_frob_sq {d : ℕ} (V : Fin d → (Fin n → ℝ)) (α γ : Fin d → Fin d → ℝ)
    (hmem : ∀ a b k, MemLp (Ens.prodEntry (V a) (V b) k) 2 μ)
    (hmean : ∀ a b k, μ[Ens.prodEntry (V a) (V b) k] = α a b)
    (hsq : ∀ a b k, μ[fun ω => (Ens.prodEntry (V a) (V b) k ω) ^ 2] = γ a b) :
    ∫ ω, ∑ a, ∑ b, (∑ k, (Ens.prodEntry (V a) (V b) k ω - α a b)) ^ 2 ∂μ
      = ∑ a, ∑ b, (n : ℝ) * (γ a b - (α a b) ^ 2) := by
  have hintEntry : ∀ a b, Integrable
      (fun ω => (∑ k, (Ens.prodEntry (V a) (V b) k ω - α a b)) ^ 2) μ :=
    fun a b => (Ens.entry_memLp (V a) (V b) (α a b) (hmem a b)).integrable_sq
  have hintRow : ∀ a, Integrable
      (fun ω => ∑ b, (∑ k, (Ens.prodEntry (V a) (V b) k ω - α a b)) ^ 2) μ :=
    fun a => integrable_finsetSum _ fun b _ => hintEntry a b
  rw [integral_finsetSum _ fun a _ => hintRow a]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [integral_finsetSum _ fun b _ => hintEntry a b]
  refine Finset.sum_congr rfl fun b _ => ?_
  exact Ens.gram_subspace_entry_sq (V a) (V b) (hmem a b) (hmean a b) (hsq a b)

/-- **General Frobenius deviation bound (Jensen).** The expected Frobenius norm of the
subspace deviation operator is `E‖VᵀM̃V‖_F ≤ √(Σ_{a,b} n·(γ_{ab} − α_{ab}²))`. -/
theorem gram_subspace_frob_le {d : ℕ} (V : Fin d → (Fin n → ℝ)) (α γ : Fin d → Fin d → ℝ)
    (hmem : ∀ a b k, MemLp (Ens.prodEntry (V a) (V b) k) 2 μ)
    (hmean : ∀ a b k, μ[Ens.prodEntry (V a) (V b) k] = α a b)
    (hsq : ∀ a b k, μ[fun ω => (Ens.prodEntry (V a) (V b) k ω) ^ 2] = γ a b) :
    ∫ ω, Real.sqrt (∑ a, ∑ b, (∑ k, (Ens.prodEntry (V a) (V b) k ω - α a b)) ^ 2) ∂μ
      ≤ Real.sqrt (∑ a, ∑ b, (n : ℝ) * (γ a b - (α a b) ^ 2)) := by
  have hintF : Integrable
      (fun ω => ∑ a, ∑ b, (∑ k, (Ens.prodEntry (V a) (V b) k ω - α a b)) ^ 2) μ :=
    integrable_finsetSum _ fun a _ => integrable_finsetSum _ fun b _ =>
      (Ens.entry_memLp (V a) (V b) (α a b) (hmem a b)).integrable_sq
  have hnn : 0 ≤ᵐ[μ]
      fun ω => ∑ a, ∑ b, (∑ k, (Ens.prodEntry (V a) (V b) k ω - α a b)) ^ 2 :=
    Filter.Eventually.of_forall fun ω =>
      Finset.sum_nonneg fun a _ => Finset.sum_nonneg fun b _ => sq_nonneg _
  have h := integral_sqrt_le_sqrt_integral hintF hnn
  rwa [Ens.gram_subspace_frob_sq V α γ hmem hmean hsq] at h

/-- **General `isotropy_at_init`, expectation form.** If every entry's variance gap obeys
`γ_{ab} − α_{ab}² ≤ K/n²`, the expected Frobenius deviation of the `d`-dimensional subspace
restriction is `O(d/√n)`:

    E‖VᵀM̃V‖_F ≤ d · √(K/n).

With `d = d_V = o(√n)` this is `o(1)` (and matches `‖errIso‖ ≤ K/√n` when `d` is bounded).
The full multi-dimensional `isotropy_at_init` bound, discharged in expectation modulo the
within-row joint-moment inputs. -/
theorem gram_subspace_isotropy_bound {d : ℕ} (V : Fin d → (Fin n → ℝ))
    (α γ : Fin d → Fin d → ℝ) {K : ℝ}
    (hmem : ∀ a b k, MemLp (Ens.prodEntry (V a) (V b) k) 2 μ)
    (hmean : ∀ a b k, μ[Ens.prodEntry (V a) (V b) k] = α a b)
    (hsq : ∀ a b k, μ[fun ω => (Ens.prodEntry (V a) (V b) k ω) ^ 2] = γ a b)
    (hn : 0 < n) (hgap : ∀ a b, γ a b - (α a b) ^ 2 ≤ K / (n : ℝ) ^ 2) :
    ∫ ω, Real.sqrt (∑ a, ∑ b, (∑ k, (Ens.prodEntry (V a) (V b) k ω - α a b)) ^ 2) ∂μ
      ≤ (d : ℝ) * Real.sqrt (K / (n : ℝ)) := by
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have hb : ∑ a, ∑ b, (n : ℝ) * (γ a b - (α a b) ^ 2) ≤ (d : ℝ) ^ 2 * (K / (n : ℝ)) := by
    calc ∑ a, ∑ b, (n : ℝ) * (γ a b - (α a b) ^ 2)
        ≤ ∑ _a : Fin d, ∑ _b : Fin d, K / (n : ℝ) := by
          refine Finset.sum_le_sum fun a _ => Finset.sum_le_sum fun b _ => ?_
          calc (n : ℝ) * (γ a b - (α a b) ^ 2) ≤ (n : ℝ) * (K / (n : ℝ) ^ 2) :=
                mul_le_mul_of_nonneg_left (hgap a b) hnpos.le
            _ = K / (n : ℝ) := by field_simp
      _ = (d : ℝ) ^ 2 * (K / (n : ℝ)) := by
          simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring
  calc ∫ ω, Real.sqrt (∑ a, ∑ b, (∑ k, (Ens.prodEntry (V a) (V b) k ω - α a b)) ^ 2) ∂μ
      ≤ Real.sqrt (∑ a, ∑ b, (n : ℝ) * (γ a b - (α a b) ^ 2)) :=
        Ens.gram_subspace_frob_le V α γ hmem hmean hsq
    _ ≤ Real.sqrt ((d : ℝ) ^ 2 * (K / (n : ℝ))) := Real.sqrt_le_sqrt hb
    _ = (d : ℝ) * Real.sqrt (K / (n : ℝ)) := by
        rw [Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (Nat.cast_nonneg d)]

end RandomMatrixEnsemble

end SffProof
