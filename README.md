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
| E | model / scff / fisher / gradients / metrics | ⬜ not started |
| E | E1 init-scaling, E2 dynamics, E3 batch/width | ⬜ not started |

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

Not yet implemented. Planned: PyTorch, CPU-fine; see `design.md` §2.

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
