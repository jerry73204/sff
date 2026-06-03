# The price of locality — an information lower bound on FF↔BP alignment

**Status:** keystone proved in Lean (`lean/SffProof/InfoBound.lean`, sorry-free); minimax wrapper
stated with proof sketch. This document turns the empirical observation *"we tried every local
gap-closer and only residual works"* into a theorem: **no transport-blind local rule can reach
perfect alignment with backprop when the downstream map is anisotropic, and feature learning makes
it anisotropic.** The residual architecture is the unique escape because it forces isotropy *a
priori*, with zero downstream information.

This is the theory that was missing (see `FINDINGS.md`, "is there a theoretical gap"). The original
theory is a *static, initialization-time, geometric* characterization of *when* alignment holds
(`MᵀM|_V = cI`). It does not model (a) the *dynamics* — learning grows the anisotropy — or (b) the
*information cost* — achieving isotropy locally needs the downstream Jacobian, which locality
forbids. The bound below addresses (b) and motivates (a).

## 1. Setup — the gradient, geometrically

Reps live on the unit sphere `S^{n−1}`; the contrastive gradient is a tangent vector. At layer `ℓ`,
restricted to the contrastive subspace `V` (dim `d_V`):

- **BP gradient:** `g_BP = Mᵀ s_L` — the output signal `s_L` pulled back through the downstream map
  `M = M^{ℓ→L}` (the cotangent map of the inter-layer flow).
- **Local (FF) gradient:** `g_FF = Φ(local data)` — produced **without observing `M`** (forward-only,
  no weight transport). The natural local rule uses *identity transport*, `g_FF = s_L` in the layer
  chart (it cannot do otherwise — `M` is downstream).

Polar-decompose `M = Q S` on `V` (`Q` orthogonal, `S = (MᵀM)^{1/2} ⪰ 0` the stretch). The rotation
`Q` is harmless; the stretch `S` is the entire defect (`grad_decomp.py`: `A_full` is a pure function
of the anisotropy of `S`, kernel drift ≈ 0). Alignment is perfect iff `S = c·I`.

## 2. The keystone — anisotropy is a hard ceiling (Kantorovich), *proved*

The worst case lives in the 2-D plane of `S`'s extreme eigenvectors, eigenvalues `a ≥ b > 0`,
condition number `κ = a/b`. Parametrize the signal's energy split by `c = cos²θ ∈ [0,1]`. The squared
cosine of the angle between the unit signal `u` and `S u` is

```
cosSq a b c = (a·c + b·(1−c))² / (a²·c + b²·(1−c)).
```

**Theorem (`alignment_capped`, Lean).** For `a, b > 0`, `c ∈ [0,1]`,

```
alignCapSq a b  ≤  cosSq a b c ,      alignCapSq a b := 4ab/(a+b)² = (2√κ/(1+κ))².
```

**Theorem (`cap_eq_one_iff_isometry`, Lean).** `alignCapSq a b = 1 ↔ a = b` (κ = 1).

**Theorem (`aniso_caps_alignment`, Lean).** `a ≠ b → alignCapSq a b < 1`.

So the alignment a transport-blind rule can reach is capped by the condition number, and the cap is
`1` **iff** the stretch is isotropic. The engine is an exact sum-of-squares identity
(`kantorovich_sos`, proved by `ring`):

```
(a·c + b·(1−c))²·(a+b)² − 4ab·(a²·c + b²·(1−c))  =  ((a−b)·(a·c − b·(1−c)))² .
```

The slack is a perfect square — vanishing exactly at `a = b` (isotropy) or `c = b/(a+b)` (the
worst-case 45°-type signal where the cap is *attained*). Deterministic: no width, no randomness.

## 3. The information step — why a local rule is stuck at the cap

The cap bounds the *identity-transport* rule. Could a cleverer `M`-independent operator `T`
(`g_FF = T s_L`) beat it? Only by undoing `S` — i.e. `T ≈ S^{-1}` (precondition) or by rotating into
`S`'s eigenbasis. **Both require knowing `S`'s eigenbasis, which is downstream information a local
rule cannot observe.**

**Minimax statement (sketch).** Model locality as: the rule fixes `T` before `S`'s eigenbasis is
revealed (the downstream layers have not yet committed their stretch directions; equivalently, over
the orbit of `S` under rotations `Q S Qᵀ`, the rule must use one `T`). Then

```
min_T  max_Q  ∠( T s_L ,  (Q S Qᵀ) s_L )  ≥  arccos( 2√κ/(1+κ) ),
```

because as `Q` ranges over rotations the pulled-back direction sweeps a cone of half-angle
`arccos(2√κ/(1+κ))` (Section 2 applied in each eigenplane), and a single ray `T s_L` cannot lie
within less than that of every cone element. The identity rule attains the bound, so it is minimax-
optimal: **no transport-blind local rule beats the Kantorovich cap.** (Full proof: reduce to the
extreme 2-plane, apply `alignment_capped`; the cone-sweep is the rotational orbit. Formalizing the
orbit argument in Lean is future work; the per-plane keystone it rests on is proved.)

## 4. Corollary — the price of locality is irreducible

Combine with the dynamical observation (`grad_decomp.py`: training grows `aniso`, i.e. `κ > 1` for
any net that actually learns features — expressivity *is* metric distortion):

> **Any forward-only local rule on an expressive (anisotropic, `κ > 1`) network has alignment defect
> `1 − A ≥ 1 − (2√κ/(1+κ))² = (√κ−1)²/(κ+1) > 0`, and this is unavoidable without downstream
> information.** Backprop is exempt: it applies the *exact* pullback `Mᵀ` (it observes `S`), so it
> incurs no transport defect. This is the price of locality, and it grows with the anisotropy that
> expressivity demands.

## 5. Why residual is the unique escape

To force `κ = 1` (defect `0`) *without measuring `M`*, constrain `M ≈ I` by construction. The
residual block `y^ℓ = y^{ℓ−1} + α·φ(W^ℓ y^{ℓ−1})` gives `M = ∏(I + αJ) ≈ I` (proved isotropic:
`residual_isotropy`, `‖MᵀM − 1‖ ≤ 2‖E‖ + ‖E‖²`). Identity is isotropic on **every** subspace, so the
residual rule needs neither the eigenbasis nor `V` itself. It is the only known mechanism that makes
the transport an isometry with **zero** downstream information — hence the unique cheap fix. It does
not remove the expressivity↔isometry tension; it **bounds** how far learning can push `κ` from `1`.

## 6. What remains (the dynamics, Gap 1)

The bound is static in `κ`. The open theorem is a **dynamical** law: prove that the SCFF goodness
gradient *increases* `aniso|_V` over training (feature learning ⇒ anisotropy growth), bounding `κ(t)`
and hence `A(t)`. That would convert the measured `aniso` curve (`0.65 → 0.93` under training) into a
predicted alignment trajectory and close the static↔trained gap end to end.

---

**Lean:** `lean/SffProof/InfoBound.lean` — `kantorovich_sos`, `kantorovich_cross`, `normSq_pos`,
`alignment_capped`, `cap_eq_one_iff_isometry`, `aniso_caps_alignment` (all sorry-free).
**Empirics:** `empirical/experiments/grad_decomp.py` (anisotropy = the whole gap; learning grows it),
`iso_penalty.py` (local isotropy penalty dominated — isotropy ≠ identity, wrong subspace).
