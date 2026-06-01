/-
Layer 3 — Main theorem (design.md §3.1, obligation 5).

Assemble: feed the Layer-2 named hypotheses (`isotropy_at_init`, `gram_match`) into the
Layer-1 perturbation lemma (`alignment_perturbation_bound_linear`) to obtain the headline
SCFF gradient-alignment-at-initialization bound

    1 − A^(ℓ)  ≤  C/√n  +  C'·δ        (THEORY.md §5)

with `C = 2K/(c‖∇g‖)` and `C' = 2/(c‖∇g‖)`. The hard analytic content lives in the
hypotheses; this composition is fully proven, no `sorry`.
-/
import SffProof.Hypotheses

open scoped RealInnerProductSpace

namespace SffProof

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **Obligation 5 — `scff_alignment_at_init`.** Under the SCFF initialization hypotheses,
the misalignment between the local and global gradients is bounded by a `1/√n` term (from
`isotropy_at_init`) plus a `δ` term (from `gram_match`):

`1 − A^(ℓ) ≤ 2K/(c‖∇g‖)·(1/√n) + 2/(c‖∇g‖)·δ`. -/
theorem scff_alignment_at_init (H : SCFFInitHypotheses E) :
    1 - cosAngle H.gGlob H.gLoc
      ≤ 2 * H.K / (H.c * ‖H.gLoc‖) / Real.sqrt H.width
        + 2 / (H.c * ‖H.gLoc‖) * H.δ := by
  -- Combine the two named hypotheses via the triangle inequality.
  have herr : ‖H.errIso + H.errGram‖ ≤ H.K / Real.sqrt H.width + H.δ :=
    (norm_add_le _ _).trans (add_le_add H.isotropy_at_init H.gram_match)
  -- Layer-1 linear perturbation bound with the bundled structural decomposition.
  have hlin := alignment_perturbation_bound_linear
    H.c_pos H.gLoc_ne H.gGlob_ne H.decomp H.err_small
  have hd : 0 < H.c * ‖H.gLoc‖ := mul_pos H.c_pos (norm_pos_iff.mpr H.gLoc_ne)
  have h2d : 0 ≤ 2 / (H.c * ‖H.gLoc‖) := div_nonneg (by norm_num) hd.le
  refine hlin.trans ?_
  calc 2 * ‖H.errIso + H.errGram‖ / (H.c * ‖H.gLoc‖)
      = 2 / (H.c * ‖H.gLoc‖) * ‖H.errIso + H.errGram‖ := by ring
    _ ≤ 2 / (H.c * ‖H.gLoc‖) * (H.K / Real.sqrt H.width + H.δ) :=
          mul_le_mul_of_nonneg_left herr h2d
    _ = 2 * H.K / (H.c * ‖H.gLoc‖) / Real.sqrt H.width
          + 2 / (H.c * ‖H.gLoc‖) * H.δ := by ring

end SffProof
