# Project: SCFF / NGD-FF — Empirical Demo + Lean Formalization

A two-track project, run in parallel:

- **Track E (Empirical):** demonstrate that Self-Contrastive Forward-Forward (SCFF)
  with Fisher-weighting produces layer-wise gradients that are *aligned* with the
  global contrastive-loss gradient, and track the dynamics that the theory says
  govern whether that alignment persists during training.
- **Track L (Lean):** formalize the **SCFF gradient-alignment-at-initialization**
  theorem, decomposed so that the provable algebraic skeleton is separated from the
  deep analytic facts (random-matrix concentration), which are introduced as named
  hypotheses rather than proven from scratch.

The two tracks share one object — the alignment cosine
`cos∠(∇g^(ℓ), ∇L_con|_ℓ)` — which Track L proves is `1 − O(1/√n)` at init under stated
hypotheses, and Track E measures empirically across training.

---

## 0. Context the agent needs

This is research code, not production. Correctness, reproducibility, and clarity of the
math↔code correspondence matter more than performance. Prefer small, inspectable networks
over scale. Every empirical quantity must map to a named symbol in `THEORY.md` (the agent
should create this from Section 1 below).

The central claim being tested:

> In a wide network under μP, at initialization, the **local** layer-wise InfoNCE gradient
> and the **global** InfoNCE gradient projected onto that layer are parallel up to `O(1/√n)`,
> provided the contrastive-signal subspace has dimension `o(√n)`. Whether this alignment
> *persists* during training is governed by a competition between (a) Gram-matrix alignment
> across layers improving, and (b) the downstream Jacobian developing anisotropy on the
> contrastive subspace.

Track L proves the "at initialization" clause. Track E tests the "persists" clause.

---

## 1. Mathematical objects (single source of truth)

The agent must create `THEORY.md` restating these precisely; code symbols must match.

- Network: `L` layers, width `n`. Layer map `y^(ℓ) = σ(W^(ℓ) y^(ℓ-1))`. Linear case: `σ = id`.
- Normalized representation: `z^(ℓ) = y^(ℓ) / ‖y^(ℓ)‖`.
- Gram matrix at layer ℓ over a batch: `K^(ℓ)_{ij} = ⟨z^(ℓ)(x_i), z^(ℓ)(x_j)⟩`.
- InfoNCE softmax weights: `p^(ℓ)_{ij} = softmax_j(⟨z_i, z_j⟩ / τ)`.
- Local goodness gradient (derived in THEORY.md, must be reproduced symbolically):
  `∇_{W^(ℓ)} g^(ℓ) = (1/τ) Σ_i P⊥_{z_i} [ z_{i+} − Σ_j p^(ℓ)_{ij} z_j ] (y^(ℓ-1)_i)^T`
  where `P⊥_z = (I − z z^T)/‖y‖` is the normalization Jacobian.
- Global gradient projected to layer ℓ: same right factor `(y^(ℓ-1)_i)^T`, left factor
  carries the downstream Jacobian `M^(ℓ+1→L)` and the *final-layer* softmax weights `p^(L)`.
- **Alignment cosine** `A^(ℓ) = cos∠(∇_{W^(ℓ)} g^(ℓ), ∇_{W^(ℓ)} L_con)`.
- **Gram misalignment** `Δ_Gram^(ℓ) = ‖K^(ℓ) − K^(L)‖_F` (normalized).
- **Subspace anisotropy** `Aniso^(ℓ)`: deviation of `(M^(ℓ+1→L))^T M^(ℓ+1→L)` from a scalar
  multiple of identity, *restricted to the contrastive subspace* V (span of the per-batch
  positive/negative difference directions). Operationalized in 2.3.
- **Contrastive subspace dimension** `d_V`: numerical rank of the matrix of `[z_{i+} − Σ_j p_{ij} z_j]`
  vectors over the batch.

The theorem: `1 − A^(ℓ) = O(1/√n) + O(Δ_Gram^(ℓ))`, valid while `d_V = o(√n)`.

---

## 2. Track E — Empirical

### 2.1 Deliverables
1. `model.py` — MLP with configurable depth/width, μP-style init (1/√fan_in scaling, last
   layer 1/fan_in), linear and ReLU modes. Linear mode is the *primary* mode (matches the
   proven theory); ReLU is the stretch.
2. `scff.py` — SCFF training: per-layer InfoNCE on normalized activations, augmentation-based
   positives, in-batch negatives. Layer-local updates (stop-gradient between layers).
3. `fisher.py` — K-FAC-style local Fisher block `F^(ℓ) ≈ A^(ℓ) ⊗ Ĝ^(ℓ)` from local stats,
   with damped inverse. The Ĝ factor uses the **local goodness** gradient, not a global one.
4. `gradients.py` — compute BOTH `∇g^(ℓ)` (local) and `∇L_con|_ℓ` (global, via one real
   backward pass through the contrastive loss, used ONLY as a measurement probe, never to train).
5. `metrics.py` — `A^(ℓ)`, `Δ_Gram^(ℓ)`, `Aniso^(ℓ)`, `d_V`, per layer per step.
6. `experiments/` — the three studies below, each a script + saved CSV + plot.

### 2.2 The three studies (these ARE the result)
- **E1 — Init scaling.** At initialization only, sweep width `n ∈ {64,128,256,512,1024,2048}`,
  fixed small batch. Measure `1 − A^(ℓ)` vs `n`. **Prediction:** decays like `n^(−1/2)`.
  Fit the exponent; report it with CI. This is the direct empirical analogue of the Lean theorem.
- **E2 — Training dynamics.** Train with SCFF. Track `A^(ℓ)`, `Δ_Gram^(ℓ)`, `Aniso^(ℓ)` vs step.
  **Prediction:** `A^(ℓ)` stays near 1 because `Δ_Gram` falls even as `Aniso` rises. The headline
  plot overlays all three. This tests the open dynamical hypothesis — a *negative* result here
  is still a publishable finding, so instrument it honestly.
- **E3 — Batch/width tradeoff.** Vary batch size `B` at fixed `n`; locate where `d_V` crosses
  `~√n` and check that `A^(ℓ)` degrades past that point. **Prediction:** alignment holds for
  `B ≪ √n`, breaks beyond.

### 2.3 Estimator notes (do not skip)
- `Aniso^(ℓ)`: form V from the top-`d_V` left singular vectors of the stacked contrastive-signal
  vectors. Compute `R = (M^(ℓ+1→L))^T M^(ℓ+1→L)` restricted to V (i.e. `V^T R V`). Report
  `‖V^T R V − (tr/dim)·I‖_F / ‖V^T R V‖_F`. In ReLU mode `M` is the input-dependent product of
  `W^(k) D^(k)`; compute per-sample then average.
- `d_V`: numerical rank at threshold (e.g. singular values > 1e-3 · σ_max).
- Always run ≥5 seeds; plot mean ± std. Never report a single-seed curve.
- The global-gradient probe must use `torch.autograd` on `L_con` with the network in eval-grad
  mode; assert it never enters the optimizer step.

### 2.4 Stack
PyTorch, numpy, matplotlib, pytest. CPU-fine at these sizes; CUDA optional. Pin seeds, log
configs to YAML, one CSV per run under `runs/`.

### 2.5 Track-E acceptance
- E1 exponent within [−0.65, −0.35] (i.e. consistent with −1/2) in linear mode.
- E2 produces the three-quantity overlay with a clear written verdict on the hypothesis.
- E3 shows a detectable alignment knee near `d_V ≈ √n`.
- `pytest` covers: gradient formula matches autograd (linear case, to 1e-5); Fisher inverse
  damping; metric symmetry/range sanity.

---

## 3. Track L — Lean formalization

### 3.1 Strategy (READ FIRST — this determines feasibility)
The full theorem needs concentration for products of random matrices restricted to a subspace.
**That is not in Mathlib and must NOT be the agent's first target.** Instead, layer the proof:

- **Layer 0 — Definitions.** Networks, normalized reps, the gradient expressions, the alignment
  cosine, all over `EuclideanSpace ℝ (Fin n)` / `Matrix`. No proofs, just well-typed defs that
  typecheck and match `THEORY.md`.
- **Layer 1 — Algebraic skeleton (PROVE THIS).** The exact, finite-`n`, deterministic facts:
  1. The local and global gradients share the right factor `(y^(ℓ-1)_i)^T` (structural lemma).
  2. **Alignment ⇔ isotropy reduction:** if the downstream operator restricted to V equals
     `c • I_V` and the softmax weights match (`p^(ℓ) = p^(L)`), then `A^(ℓ) = 1`. This is pure
     linear algebra and is the mathematical heart — fully provable in Lean now.
  3. Cosine bounds: a perturbation lemma giving `1 − A ≤ C·‖E‖` when the downstream operator is
     `c•I_V + E` and softmax weights differ by `δ`. (Lipschitz-style, real-analysis, in scope.)
- **Layer 2 — Named hypotheses (AXIOMATIZE, don't prove).** State as Lean `structure` fields or
  hypotheses:
  - `isotropy_at_init : ‖V^T (Mᵀ M) V − c • I‖ ≤ K / Real.sqrt n`  (the random-matrix fact)
  - `gram_match : ‖p^(ℓ) − p^(L)‖ ≤ δ`
  These are the deep analytic inputs. The agent documents each with the informal proof sketch and
  a literature pointer, and marks them clearly as assumptions.
- **Layer 3 — Main theorem (ASSEMBLE).** `1 − A^(ℓ) ≤ C/√n + C'·δ` follows by feeding the Layer-2
  hypotheses into the Layer-1 perturbation lemma. This composition is the headline Lean result and
  is genuinely provable because the hard parts are hypotheses.

The honest framing in the repo: **"We formalize the reduction of SCFF alignment to two analytic
hypotheses, and prove the reduction is valid. The hypotheses themselves (random-matrix isotropy,
Gram matching) are stated precisely and left as future formalization."** That is a real, defensible
contribution and is achievable; "prove the whole thing from measure theory up" is not.

### 3.2 Concrete first proof obligations (in order)
1. `inner_product_cosine_one_of_parallel` — if `u = c • v`, `c > 0`, then `cos∠(u,v) = 1`. (Warm-up;
   may partly exist in Mathlib — search `inner_mul_le_norm_mul_norm` / Cauchy-Schwarz equality case.)
2. `gradient_shared_right_factor` — the structural lemma 1 above.
3. `alignment_one_of_isotropic_and_matched` — Layer-1 fact 2.
4. `alignment_perturbation_bound` — Layer-1 fact 3.
5. `scff_alignment_at_init` — Layer-3 assembly.

### 3.3 Setup the agent figures out
The user has no specific Lean setup. The agent should: install `elan`, create a Lean 4 project
with `lake`, add Mathlib as a dependency (matching a recent stable toolchain), and confirm a
trivial Mathlib lemma compiles before writing anything else. Document the exact toolchain version
pinned in `lean-toolchain`. If Mathlib's matrix/inner-product API has moved, adapt — do not pin to
memory of the API.

### 3.4 Track-L acceptance
- `lake build` succeeds on a clean checkout.
- Obligations 1–5 compile with **no `sorry`** in Layers 0,1,3.
- Layer-2 hypotheses are isolated in one file, each with a docstring: informal statement, why it's
  true, citation, and "NOT YET FORMALIZED" tag. A CI grep asserts no `sorry` exists *outside* that file.
- A short `LEAN_NOTES.md` maps every Lean definition to its `THEORY.md` symbol.

---

## 4. Repo layout
```
sff-project/
  THEORY.md            # agent writes from §1; the shared symbol table
  PROJECT.md           # this file
  empirical/
    model.py scff.py fisher.py gradients.py metrics.py
    experiments/{e1_init_scaling,e2_dynamics,e3_batch_width}.py
    tests/  runs/  plots/
  lean/
    SffProof/{Defs,Skeleton,Hypotheses,Main}.lean
    lakefile.toml  lean-toolchain  LEAN_NOTES.md
  README.md            # how to run both tracks; current status of the dynamical hypothesis
```

## 5. Milestones
- **M1:** `THEORY.md` + symbol table agreed; repo skeleton + Lean builds a trivial Mathlib lemma;
  `gradients.py` matches autograd in linear case (the math↔code anchor).
- **M2 (Track E):** E1 done — init-scaling exponent measured. (Track L): obligations 1–3 compile.
- **M3:** E2 + E3 done with multi-seed plots and written verdict. Obligations 4–5 compile;
  hypotheses file documented.
- **M4:** README writes up empirical findings vs the Lean theorem; states whether the dynamical
  hypothesis held, with the three-quantity evidence.

## 6. Pitfalls (flagged so the agent doesn't burn time)
- Do NOT let the global-gradient probe leak into training. It is measurement only.
- Do NOT attempt random-matrix concentration in Lean. Axiomatize it (Layer 2).
- Linear mode is primary; if ReLU mode contradicts predictions, report it — the proven theory is
  linear-case, so a ReLU gap is expected and informative, not a bug to hide.
- μP scaling must be correct or E1's exponent will be wrong; unit-test the init variances.
- Prefer exact gradient formulas verified against autograd over clever vectorization.
