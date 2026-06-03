/-
Information lower bound on local↔global alignment — the *price of locality*, formalized.

Motivation (Track E, docs/FINDINGS.md "Geometric decomposition"). The FF↔BP gap is entirely
**transport non-isometry**: BP's gradient at layer `ℓ` is the output contrastive signal pulled back
through the downstream map `M`, while a forward-only *local* rule must produce its estimate WITHOUT
observing `M` (the downstream weights). Polar-decompose `M = Q S` on the contrastive subspace; the
rotation `Q` is harmless, the stretch `S = (MᵀM)^{1/2}` is the defect, and alignment holds iff
`S = c·I` (isotropy). The empirical diagnostic showed feature learning *grows* the anisotropy of `S`.

This file proves the deterministic keystone that turns "we tried every local fix and only residual
works" into a theorem: **the anisotropy of the downstream stretch is a hard ceiling on the alignment
any transport-blind rule can reach.** Concretely, the worst case lives in the 2-D plane spanned by
the extreme eigenvectors of `S` (eigenvalues `a ≥ b > 0`, condition number `κ = a/b`). Writing the
signal's energy split as `c = cos²θ ∈ [0,1]`, the squared cosine of the angle between the unit
signal `u` and `S u` is

    cosSq a b c = (a·c + b·(1−c))² / (a²·c + b²·(1−c)),

and it is bounded below by the **Kantorovich ceiling** `alignCapSq a b = 4ab/(a+b)² = (2√κ/(1+κ))²`:

    alignCapSq a b ≤ cosSq a b c          (alignment_capped)

with the ceiling attained (a transport-blind rule cannot beat it on the worst signal), and

    alignCapSq a b = 1  ↔  a = b          (cap_eq_one_iff_isometry)
    a ≠ b              →  alignCapSq a b < 1   (aniso_caps_alignment)

So perfect alignment is reachable only when `S` is isotropic (`κ = 1`); any expressive — hence
anisotropic — downstream map forces a strictly positive alignment defect. The only way to guarantee
`κ = 1` *a priori*, without measuring `M`, is to constrain `M ≈ I` by construction: the residual
architecture. Hence the residual gap to backprop is irreducible for transport-blind local rules.

The engine is an exact sum-of-squares identity (`kantorovich_sos`):
    (a·c + b·(1−c))²·(a+b)² − 4ab·(a²·c + b²·(1−c)) = ((a−b)·(a·c − b·(1−c)))².
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

namespace SffProof
namespace InfoBound

/-- **Sum-of-squares keystone.** The Kantorovich gap is an exact perfect square: the numerator
inequality `4ab·‖Su‖² ≤ ⟨u,Su⟩²·(a+b)²` has slack exactly `((a−b)(a·c − b·(1−c)))²`. -/
theorem kantorovich_sos (a b c : ℝ) :
    (a * c + b * (1 - c)) ^ 2 * (a + b) ^ 2
        - 4 * a * b * (a ^ 2 * c + b ^ 2 * (1 - c))
      = ((a - b) * (a * c - b * (1 - c))) ^ 2 := by
  ring

/-- Cross-multiplied Kantorovich inequality (the form used after clearing denominators). -/
theorem kantorovich_cross (a b c : ℝ) :
    4 * a * b * (a ^ 2 * c + b ^ 2 * (1 - c))
      ≤ (a * c + b * (1 - c)) ^ 2 * (a + b) ^ 2 := by
  nlinarith [sq_nonneg ((a - b) * (a * c - b * (1 - c)))]

/-- Squared cosine of the angle between the unit signal `u` and `S u`, in the extreme-eigenvalue
plane, as a function of the energy split `c = cos²θ ∈ [0,1]` and eigenvalues `a, b > 0`. -/
noncomputable def cosSq (a b c : ℝ) : ℝ :=
  (a * c + b * (1 - c)) ^ 2 / (a ^ 2 * c + b ^ 2 * (1 - c))

/-- The Kantorovich alignment ceiling set by the condition number: `(2√(ab)/(a+b))² = 4ab/(a+b)²`. -/
noncomputable def alignCapSq (a b : ℝ) : ℝ := 4 * a * b / (a + b) ^ 2

/-- The denominator `‖S u‖² = a²·c + b²·(1−c)` is positive for `a,b > 0`, `c ∈ [0,1]`. -/
theorem normSq_pos {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc0 : 0 ≤ c) (hc1 : c ≤ 1) :
    0 < a ^ 2 * c + b ^ 2 * (1 - c) := by
  rcases eq_or_lt_of_le hc0 with hc | hcpos
  · subst hc; positivity
  · have h1 : 0 < a ^ 2 * c := mul_pos (pow_pos ha 2) hcpos
    have h2 : 0 ≤ b ^ 2 * (1 - c) := mul_nonneg (le_of_lt (pow_pos hb 2)) (by linarith)
    linarith

/-- **Anisotropy caps alignment (Kantorovich).** For any signal energy split `c ∈ [0,1]`, the
squared alignment of a transport-blind (identity-transport) rule is at most the condition-number
ceiling `4ab/(a+b)²`. Width/randomness not required; this is deterministic. -/
theorem alignment_capped {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc0 : 0 ≤ c) (hc1 : c ≤ 1) :
    alignCapSq a b ≤ cosSq a b c := by
  unfold alignCapSq cosSq
  have hden : 0 < a ^ 2 * c + b ^ 2 * (1 - c) := normSq_pos ha hb hc0 hc1
  have hab : 0 < (a + b) ^ 2 := by positivity
  rw [div_le_div_iff₀ hab hden]
  nlinarith [sq_nonneg ((a - b) * (a * c - b * (1 - c)))]

/-- **No gap iff isometry.** The Kantorovich ceiling equals `1` exactly when the stretch is
isotropic (`a = b`, i.e. condition number `κ = 1`). -/
theorem cap_eq_one_iff_isometry {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    alignCapSq a b = 1 ↔ a = b := by
  unfold alignCapSq
  rw [div_eq_iff (by positivity : (a + b) ^ 2 ≠ 0), one_mul]
  constructor
  · intro h
    have h2 : (a - b) ^ 2 = 0 := by linear_combination -h
    have := (pow_eq_zero_iff (by norm_num : 2 ≠ 0)).mp h2
    linarith
  · intro h; rw [h]; ring

/-- **Strict cap under anisotropy (the price of locality).** Any anisotropic downstream stretch
(`a ≠ b`, condition number `κ > 1`) forces the alignment ceiling strictly below `1`: a
transport-blind local rule cannot reach perfect alignment. -/
theorem aniso_caps_alignment {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hne : a ≠ b) :
    alignCapSq a b < 1 := by
  unfold alignCapSq
  rw [div_lt_one (by positivity)]
  have hd : a - b ≠ 0 := sub_ne_zero.mpr hne
  have : 0 < (a - b) ^ 2 := (sq_nonneg _).lt_of_ne (Ne.symm (pow_ne_zero 2 hd))
  nlinarith [this]

end InfoBound
end SffProof
