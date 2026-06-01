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
import Mathlib.MeasureTheory.Integral.Bochner.Basic

open scoped RealInnerProductSpace
open MeasureTheory

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

/-! ### Expectation-mode headline (structural bridge)

The deterministic `scff_alignment_at_init` takes the analytic bounds as hypotheses. The
random-matrix program (`SffProof.Probability.*`) proves those bounds hold *in expectation*:
`E‖errIso‖ ≤ K/√n` (`gram_subspace_isotropy_bound`) and `E‖errGram‖ ≤ δ` (Gram-entry
concentration + softmax linearization). The theorem below is the bridge: it integrates the
pointwise Layer-1 linear perturbation bound to conclude the headline *in expectation* from a
bound on `E‖Err‖`, the quantity the ensemble theorems control. -/

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω}

/-- **Expectation-mode SCFF alignment-at-initialization.** Model the local/global gradients
and their error as random (functions of the init `ω`). If `G = c•g + Err` pointwise, `g` is
bounded below (`g0 ≤ ‖g‖`), the error is small (`‖Err‖ ≤ c‖g‖`), and the *expected* error
obeys `E‖Err‖ ≤ K/√n + δ` (the random-matrix program's output), then the expected
misalignment satisfies the headline bound:

`E[1 − A^(ℓ)] ≤ (2/(c·g₀))·(K/√n + δ)`.

This discharges `scff_alignment_at_init` in expectation for the random-matrix model. -/
theorem scff_alignment_at_init_expectation {n : ℕ}
    (g G Err : Ω → E) {c g0 K δ : ℝ}
    (hc : 0 < c) (hg0 : 0 < g0)
    (hgnorm : ∀ ω, g0 ≤ ‖g ω‖) (hgne : ∀ ω, g ω ≠ 0) (hGne : ∀ ω, G ω ≠ 0)
    (hdecomp : ∀ ω, G ω = c • g ω + Err ω) (hsmall : ∀ ω, ‖Err ω‖ ≤ c * ‖g ω‖)
    (hErrInt : Integrable (fun ω => ‖Err ω‖) μ)
    (hcosInt : Integrable (fun ω => 1 - cosAngle (G ω) (g ω)) μ)
    (hErrBound : ∫ ω, ‖Err ω‖ ∂μ ≤ K / Real.sqrt n + δ) :
    ∫ ω, (1 - cosAngle (G ω) (g ω)) ∂μ ≤ 2 / (c * g0) * (K / Real.sqrt n + δ) := by
  have hcg0 : 0 < c * g0 := mul_pos hc hg0
  have h2cg0 : 0 ≤ 2 / (c * g0) := div_nonneg (by norm_num) hcg0.le
  -- pointwise: 1 − A ≤ 2‖Err‖/(c‖g‖) ≤ (2/(c·g0))‖Err‖
  have hptwise : ∀ ω, 1 - cosAngle (G ω) (g ω) ≤ 2 / (c * g0) * ‖Err ω‖ := by
    intro ω
    have hcg : c * g0 ≤ c * ‖g ω‖ := mul_le_mul_of_nonneg_left (hgnorm ω) hc.le
    calc 1 - cosAngle (G ω) (g ω)
        ≤ 2 * ‖Err ω‖ / (c * ‖g ω‖) :=
          alignment_perturbation_bound_linear hc (hgne ω) (hGne ω) (hdecomp ω) (hsmall ω)
      _ ≤ 2 * ‖Err ω‖ / (c * g0) := by gcongr
      _ = 2 / (c * g0) * ‖Err ω‖ := by ring
  -- integrate and use the expected-error bound
  calc ∫ ω, (1 - cosAngle (G ω) (g ω)) ∂μ
      ≤ ∫ ω, 2 / (c * g0) * ‖Err ω‖ ∂μ :=
        integral_mono hcosInt (hErrInt.const_mul _) hptwise
    _ = 2 / (c * g0) * ∫ ω, ‖Err ω‖ ∂μ := integral_const_mul _ _
    _ ≤ 2 / (c * g0) * (K / Real.sqrt n + δ) := mul_le_mul_of_nonneg_left hErrBound h2cg0

end SffProof
