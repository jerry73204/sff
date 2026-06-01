# SCFF / NGD-FF — Empirical Demo + Lean Formalization

Two-track study of **Self-Contrastive Forward-Forward (SCFF)** gradient alignment. Shared
object: the alignment cosine `A^(ℓ) = cos∠(∇g^(ℓ), ∇L_con|_ℓ)`.

- **Track L (Lean)** — formalize the *alignment-at-initialization* theorem:
  `1 − A^(ℓ) ≤ C/√n + C'·δ`, with the deep random-matrix facts isolated as named hypotheses.
- **Track E (Empirical)** — measure `A^(ℓ)` and its governing dynamics across training.

See `design.md` (full spec) and `THEORY.md` (symbol table, single source of truth).

## Status

| Track | Item | Status |
|---|---|---|
| L | Layer 0 defs (`Defs.lean`) | ✅ typechecks |
| L | Obl 1–4 algebraic skeleton (`Skeleton.lean`) | ✅ proven, no `sorry` |
| L | Layer 2 hypotheses (`Hypotheses.lean`) | ✅ axiomatized + documented |
| L | Obl 5 headline theorem `scff_alignment_at_init` (`Main.lean`) | ✅ proven, no `sorry` |
| E | model / scff / gradients / metrics | ✅ built, gradient↔autograd anchor passes (1e-5) |
| E | E1 init-scaling | ✅ **isotropy term scales `n^{-1/2}`** (slopes −0.45, −0.53) — validates the Lean random-matrix proof |
| E | E2 dynamics + probe | ✅ persistence fails — *genuine dynamical anisotropy* (not instability/lr/d_V); motivates Fisher |
| E | fisher (NGD-FF) | ✅ built; local K-FAC does **not** rescue persistence (cross-layer problem) |
| E | E3 batch/width | ⬜ not started |

Headline Lean result `scff_alignment_at_init` depends only on `[propext, Classical.choice,
Quot.sound]` (no `sorryAx`) — fully proven modulo the two bundled analytic hypotheses
(`isotropy_at_init`, `gram_match`).

The open **dynamical hypothesis** (does alignment *persist* during training?) is Track E2,
not yet run.

## Track L — build

```bash
cd lean
lake exe cache get      # first time: pull Mathlib oleans (large)
lake build              # builds SffProof; succeeds on clean checkout
./scripts/check_no_sorry.sh   # asserts no sorry outside Layer 2
```

Toolchain pinned in `lean/lean-toolchain` (`leanprover/lean4:v4.30.0`), Mathlib pinned in
`lean/lake-manifest.json`. `lean/LEAN_NOTES.md` maps every definition to its `THEORY.md`
symbol.

## Track E — run

uv-managed (CPU torch). With direnv: `cd empirical` auto-syncs + activates. Manual:

```bash
cd empirical
uv sync
uv run pytest                              # gradient↔autograd anchor + sanity (10 tests)
uv run python experiments/e1_init_scaling.py   # init-scaling exponent
uv run python experiments/e2_dynamics.py       # training dynamics overlay
```

Outputs land in `empirical/runs/` (CSV + YAML + verdict) and `empirical/plots/`.

**E2 finding (persistence).** Under SCFF training the alignment `A^{(\ell)}` degrades. The
probe (`experiments/e2_probe.py`) isolates the cause: weight norms are exactly preserved
(scale-invariant goodness ⇒ rotational updates — no blow-up), `d_V` stays fixed, and the
decay is lr-independent. So it is **genuine dynamical anisotropy**: SCFF grows the downstream
Jacobian's anisotropy on `V` faster than cross-layer Gram alignment improves. The §2.2
competition resolves against alignment here — a mechanism-identified negative result that
motivates Fisher/NGD-FF preconditioning (natural gradient counters anisotropy).

**Fisher finding (NGD-FF).** Local K-FAC preconditioning (`fisher.py`,
`experiments/e2_fisher.py`) does **not** rescue persistence: final `A` is ~unchanged
(0.23 vs 0.26), and aggressive damping makes it *worse* (`damp=1e-3 -> A~0`). Two reasons:
(i) the breaking anisotropy is in the *downstream/cross-layer* Jacobian, which a *local*-layer
Fisher block cannot control; (ii) the small batches required for `d_V \ll \sqrt n` make the
Fisher factors rank-deficient (rank `<= B`), so the damped inverse is noisy. Conclusion: local
natural-gradient preconditioning is insufficient — persistence is fundamentally a cross-layer
problem.

**E1 finding.** `1 - A^{(\ell)} \le C/\sqrt n + C'\delta`. The **isotropy term** (`Aniso`,
the quantity the Lean `gram_subspace_isotropy_bound` controls) scales as `n^{-1/2}`
empirically (slopes −0.45, −0.53, both in the accept band) — the Lean theorem validated.
The **total** `1-A` stays flat in `n`: the binding term is `\delta` = cross-layer kernel
drift (`p^{(\ell)} \ne p^{(L)}`), a **depth** effect not cured by width. So width alone gives
isotropy, not full BP-alignment; that also needs kernel preservation across layers.

## Layout

```
sff/
  design.md  THEORY.md  README.md
  lean/
    lakefile.toml  lean-toolchain  lake-manifest.json  LEAN_NOTES.md
    SffProof.lean  SffProof/{Defs,Skeleton,Hypotheses,Main}.lean
    scripts/check_no_sorry.sh
  empirical/   # Track E (not started)
```
