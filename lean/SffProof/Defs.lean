/-
Layer 0 — Definitions (design.md §3.1).
Well-typed defs matching THEORY.md symbols. No proofs here.

Concrete finite model:
* width `n`, input width `m`, batch `B`, all `Fin _`.
* a representation is a function `Fin B → EuclideanSpace ℝ (Fin n)`.
* gradients live in `Matrix (Fin n) (Fin m) ℝ`, built as sums of outer products
  `(leftFactor i) ⊗ (rightFactor i)`; the **right factor is shared** between the
  local and global gradients (THEORY.md §3), which is the structural lemma's content.

Frobenius inner product is defined here directly (not via a Mathlib instance) so the
alignment cosine is fully explicit.
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped RealInnerProductSpace BigOperators

namespace SffProof

/-! ### Outer products and the Frobenius cosine -/

variable {n m B : ℕ}

/-- Outer product `a ⊗ r` as an `n × m` matrix: `(a ⊗ r)_{ab} = a_a * r_b`. -/
def outer (a : Fin n → ℝ) (r : Fin m → ℝ) : Matrix (Fin n) (Fin m) ℝ :=
  fun i j => a i * r j

/-- Frobenius inner product `⟨A, Bᵀ⟩_F = Σ_{ij} A_{ij} B_{ij}`. -/
def frob (A C : Matrix (Fin n) (Fin m) ℝ) : ℝ :=
  ∑ i, ∑ j, A i j * C i j

/-- Frobenius norm. -/
noncomputable def frobNorm (A : Matrix (Fin n) (Fin m) ℝ) : ℝ :=
  Real.sqrt (frob A A)

/-- **Alignment cosine** `A^(ℓ)` (THEORY.md §4), Frobenius version on gradient matrices. -/
noncomputable def cosAngleM (A C : Matrix (Fin n) (Fin m) ℝ) : ℝ :=
  frob A C / (frobNorm A * frobNorm C)

/-! ### Representations, Gram, InfoNCE weights, contrastive signal (THEORY.md §2–3) -/

/-- A batch of representations at one layer: unit-normalized reps `z_i ∈ ℝ^n`. -/
abbrev Reps (n B : ℕ) := Fin B → EuclideanSpace ℝ (Fin n)

/-- Gram matrix `K_{ij} = ⟪z_i, z_j⟫` (THEORY.md §2). -/
noncomputable def gram (z : Reps n B) : Matrix (Fin B) (Fin B) ℝ :=
  fun i j => ⟪z i, z j⟫

/-- Unnormalized softmax scores `⟪z_i, z_j⟫ / τ`. -/
noncomputable def scores (z : Reps n B) (τ : ℝ) : Matrix (Fin B) (Fin B) ℝ :=
  fun i j => ⟪z i, z j⟫ / τ

/-- InfoNCE softmax weights `p_{ij} = softmax_j(scores)` (THEORY.md §2). -/
noncomputable def softmaxWeights (z : Reps n B) (τ : ℝ) : Matrix (Fin B) (Fin B) ℝ :=
  fun i j => Real.exp (scores z τ i j) / (∑ k, Real.exp (scores z τ i k))

/-- Contrastive-signal vector `s_i = z_{i+} − Σ_j p_{ij} z_j` (THEORY.md §3). -/
noncomputable def signal (z zpos : Reps n B) (τ : ℝ) (i : Fin B) :
    EuclideanSpace ℝ (Fin n) :=
  zpos i - ∑ j, (softmaxWeights z τ i j) • z j

/-! ### Gradients as shared-right-factor sums (THEORY.md §3)

We abstract the per-sample **left factors** (`leftLocal`, `leftGlobal`) and the shared
**right factor** `rightFactor i = (y^(ℓ-1)_i)`. The local gradient uses the local
projected signal `P⊥_{z_i} s_i^(ℓ)`; the global one uses the downstream-Jacobian-carried
signal with final-layer weights. At Layer 0 we only record the structural shape. -/

/-- Generic gradient assembled from per-sample left/right factors:
`∇ = Σ_i (left i) ⊗ (right i)`. -/
def gradFromFactors (left : Fin B → Fin n → ℝ) (right : Fin B → Fin m → ℝ) :
    Matrix (Fin n) (Fin m) ℝ :=
  ∑ i, outer (left i) (right i)

end SffProof
