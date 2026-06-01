/-
Layer 2 — Named analytic hypotheses (design.md §3.1).

These are the DEEP ANALYTIC INPUTS to the SCFF alignment theorem. They are
**AXIOMATIZED, NOT proven** — formalizing them needs random-matrix concentration for
products of random matrices restricted to a subspace, which is not in Mathlib and is
explicit non-goal (design §3.1, §6).

This is the ONLY file in the project carrying unproven mathematical assumptions. A CI
grep asserts `sorry` appears nowhere; the assumptions here are honest `structure` fields
(hypotheses), not `sorry`s — they are discharged by whoever *constructs* an
`SCFFInitHypotheses`, and each is tagged NOT YET FORMALIZED with a sketch + citation.

All quantities map to THEORY.md §5.
-/
import SffProof.Skeleton

open scoped RealInnerProductSpace

namespace SffProof

/-- **The two named hypotheses of the SCFF alignment-at-initialization theorem**, bundled
with the structural data they qualify. Gradients are abstract vectors of a real inner
product space `E` (the Frobenius gradient space); `gLoc = ∇g^(ℓ)`, `gGlob = ∇L_con|_ℓ`.

To *use* the main theorem one constructs a term of this type, i.e. supplies the analytic
facts. To *formalize* them is future work (Layer 2 of the proof program). -/
structure SCFFInitHypotheses (E : Type*)
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] where
  /-- Width `n` of the network (THEORY.md §1). -/
  width : ℕ
  width_pos : 0 < width
  /-- The common positive scale `c` relating the two gradients on the contrastive
  subspace `V` — the scalar of the isotropic part `c • I_V` (THEORY.md §5). -/
  c : ℝ
  c_pos : 0 < c
  /-- Local layer-wise InfoNCE gradient `∇_{W^(ℓ)} g^(ℓ)` (THEORY.md §3). -/
  gLoc : E
  /-- Global InfoNCE gradient projected to layer ℓ, `∇_{W^(ℓ)} L_con` (THEORY.md §3). -/
  gGlob : E
  gLoc_ne : gLoc ≠ 0
  gGlob_ne : gGlob ≠ 0
  /-- Error vectors: `errIso` from downstream anisotropy on `V`, `errGram` from the
  softmax mismatch. Their sum is the total deviation of `gGlob` from `c • gLoc`. -/
  errIso : E
  errGram : E
  /-- **Structural decomposition** (consequence of `gradient_shared_right_factor`,
  Layer 1): the global gradient is the scaled local gradient plus the two error terms. -/
  decomp : gGlob = c • gLoc + (errIso + errGram)
  K : ℝ
  δ : ℝ
  K_nonneg : 0 ≤ K
  δ_nonneg : 0 ≤ δ
  /-- **HYPOTHESIS `isotropy_at_init`** (THEORY.md §5: `‖V^T (Mᵀ M) V − c·I‖ ≤ K/√n`).
  Informal statement: at μP initialization the downstream Jacobian's Gram operator,
  restricted to the `o(√n)`-dimensional contrastive subspace `V`, is within `K/√n` of a
  scalar multiple of the identity; propagated through the (shared) right factor this
  bounds the isotropy error term by `K/√n`.
  Why true: products of i.i.d. random matrices concentrate; restricted to a low-dimensional
  subspace the Gram operator is `c·I + O(1/√n)` by matrix Bernstein / operator concentration.
  Citation: Vershynin, *High-Dimensional Probability* (2018), Ch. 4–5; Yang & Hu, μP /
  Tensor Programs.
  NOT YET FORMALIZED (random-matrix concentration is out of scope, design §3.1, §6). -/
  isotropy_at_init : ‖errIso‖ ≤ K / Real.sqrt width
  /-- **HYPOTHESIS `gram_match`** (THEORY.md §5: `‖p^(ℓ) − p^(L)‖ ≤ δ`).
  Informal statement: the layer-ℓ and final-layer InfoNCE softmax weights agree up to `δ`,
  bounding the gram error term by `δ`.
  Why true: at initialization wide-network Gram matrices `K^(ℓ)` are close across layers
  (kernel concentration), so the induced softmax distributions match to `o(1)`.
  Citation: NTK / wide-network kernel concentration (Jacot et al. 2018; Lee et al. 2019).
  NOT YET FORMALIZED. -/
  gram_match : ‖errGram‖ ≤ δ
  /-- **Small-error regime** assumption: the total error is within the isotropic scale,
  `‖errIso + errGram‖ ≤ c‖gLoc‖`. Holds once `n` is large enough that `K/√n + δ ≤ c‖gLoc‖`.
  NOT YET FORMALIZED (follows from the two bounds above for large `n`). -/
  err_small : ‖errIso + errGram‖ ≤ c * ‖gLoc‖

end SffProof
