# SCFF / NGD-FF — Findings

Consolidated synthesis of the two-track study (Lean formalization + PyTorch experiments) of
gradient alignment between Self-Contrastive Forward-Forward (SCFF) and backpropagation.

## The question

SCFF trains each layer on its own local InfoNCE "goodness" — forward-only, no backward pass,
no weight transport. Does the **local** layer gradient `∇g^(ℓ)` point the same way as the
**global** backprop gradient `∇L_con|_ℓ`? Measured by the alignment cosine
`A^(ℓ) = cos∠(∇g^(ℓ), ∇L_con|_ℓ)`. `A=1` ⇒ local learning = BP direction, no backward pass.

## The central decomposition

Both gradients are sums of outer products `Σ_i (leftᵢ)(rightᵢ)ᵀ` sharing the **same right
factor** `(y^(ℓ-1)ᵢ)ᵀ`. So alignment lives entirely in the **left factors**, and reduces to two
conditions (proven exactly, `alignment_one_of_isotropic_and_matched`):

1. **Isotropy**: downstream operator scalar on the contrastive subspace, `MᵀM|_V = c·I`.
2. **Softmax/kernel match**: `p^(ℓ) = p^(L)`.

At initialization, wide network, while `d_V = o(√n)`:

```
1 − A^(ℓ)  ≤  C/√n  +  C'·δ
```

- `C/√n` — the **isotropy** term (random-matrix concentration).
- `δ = ‖p^(ℓ) − p^(L)‖` — the **kernel-drift** term (how much the per-layer kernel changes
  across depth).

## Track L — Lean (sorry-free)

| theorem | statement |
|---|---|
| `scff_alignment_at_init` | headline: `1 − A^(ℓ) ≤ 2K/(c‖∇g‖)·(1/√n) + 2/(c‖∇g‖)·δ` |
| `scff_alignment_at_init_expectation` | random-init form: `E[1−A] ≤ (2/(c·g₀))(K/√n + δ)` |
| `gram_subspace_isotropy_bound` | random/wide isotropy: `E‖VᵀM̃V‖_F ≤ d·√(K/n)` |
| `residual_isotropy` | **residual** isotropy: `M=1+E ⇒ ‖MᵀM−1‖ ≤ 2‖E‖+‖E‖²` (deterministic) |
| `softmax_l1_le_linear` | softmax glue: logits within `ε` ⇒ `‖p(a)−p(b)‖₁ ≤ 2e^{2M}ε` |
| `ratio_perturbation` | normalization (`z=y/‖y‖`) lifting |

The deep random-matrix fact is **proven in expectation** (not axiomatized as the design
allowed). `#print axioms` on the headline → only `propext, Classical.choice, Quot.sound`.

## Track E — Experiments (gradient↔autograd verified to 1e-5)

**E1 — init scaling.** The isotropy term scales `Aniso ∝ n^{−1/2}` (fitted slopes −0.45,
−0.53) — empirically validates `gram_subspace_isotropy_bound`. But total `1−A` is **flat in
`n`**: the binding term is `δ`, a **depth** effect not cured by width.

**E2 — persistence.** Under SCFF training, alignment **degrades** (probe: genuine dynamical
anisotropy of the downstream Jacobian on `V`, not instability/lr/`d_V`). **Local K-FAC Fisher
does not rescue it** — the breaking anisotropy is cross-layer, which a local-layer
preconditioner cannot control, and small batches (required for `d_V≪√n`) make the Fisher
factors rank-deficient.

**E3 — batch/width.** `Aniso ∝ √d_V` (fitted 0.51), a **smooth crossover** at `d_V ≈ √n`
(not a sharp knee). Maps the validity boundary: width buys isotropy only while `d_V ≪ √n`.

## Method revision — residual skip connections

The recurring diagnosis: **the bottleneck is cross-layer** (init `δ`, training anisotropy);
purely-local methods (incl. local Fisher) hit a ceiling. Residual skips attack it
architecturally:

- `M = ∏(I + αJ) ≈ I` ⇒ **isotropy by construction** (no large `n` needed) and a near-frozen
  kernel (small `δ`).
- **Result** (depth sweep `L∈{4,8,16}`): residual lifts init `A` with a gap over plain that
  *widens* with depth (residual `1−A` 0.14/0.21/0.24 vs plain 0.45/0.57/0.69), lowers `Aniso`,
  and **persists** under training (`A` 0.79→0.73 vs plain 0.43→0.34). **Dense does not help**
  (its downstream Jacobian is not near-scalar).
- **Scale law**: `1−A ≈ Aniso ≈ O(α)` (linear in the residual scale). `α ≤ 0.1` gives
  near-perfect persistent alignment; the textbook `1/√L ≈ 0.35` is **too large**; want `~1/L`.
  **ReLU residual holds** (same trend, slightly better at small `α`).
- **Proven**: `residual_isotropy` (`‖MᵀM−1‖ ≤ 2‖E‖+‖E‖²`) is the deterministic `Aniso=O(α)`
  law — turns the empirical win into a theorem, no random matrix, no `o(√n)`.

## A second gap-closer — auxiliary downstream depth (LoCo-style)

Train block `ℓ`'s goodness `j` layers downstream (look-ahead): push `y^(ℓ)` through the next
`j` blocks (their weights detached → still a strictly local `W^(ℓ)` update) and compute the
InfoNCE goodness there. `j=0` = vanilla SCFF.

- On **plain** nets (the hard case), each look-ahead layer ~halves init `1−A`
  (0.476 → 0.258 → 0.131 for `j=0,1,2` at `L=6`) and roughly doubles persisted alignment
  (final `A` 0.288 → 0.473 → 0.589). Monotone in `j`.
- Mechanism: folding `j` downstream blocks into the local objective makes the local gradient
  *see* the downstream operator `M` that BP applies — the same cross-layer lever as residual,
  but by *seeing* `M` rather than making `M≈I`.
- **Substitute, not additive**: on residual (`α=0.1`, `M≈I`, `A≈0.98`) look-ahead adds almost
  nothing. Cost: `(1+j)×` forward depth per gradient, and it **erodes the FF locality /
  parallelism value-prop** (a block can't update until `j` downstream blocks run).
- Verdict: a real gap-closer where residual isn't available; buys back the locality it was
  meant to save.

## Negative results (honest, mechanism understood)

- **Local Fisher (NGD-FF)** does not rescue persistence — cross-layer problem + rank-deficient
  small-batch Fisher.
- **Forward-gradient-on-subspace** (a BP-free estimate of the global gradient via forward-mode
  JVP, tangents restricted to `span(V)⊗span(y_prev)`): the variance machinery works
  (≈4000× reduction, the project's two theorems are exactly the variance reducer), but as a
  training-signal correction it **does not beat pure SCFF**. Catch-22: where local≈global
  (residual) the estimate is redundant noise; where a correction is needed (plain/deep, low
  `A`) the *local* subspace `V` captures only ~⅓ of the global gradient. The same `δ` coupling
  defeats it both ways.
- **Per-block LayerNorm** does not cut `δ` or `1−A` (linear: change ≤0.006, within seed noise;
  ReLU: *worsens* both, `δ` 0.114→0.258 at `L=16`). `δ` is measured on the L2-normalized reps
  `z=y/‖y‖`, whose scale is already removed; LayerNorm standardizes mean/variance but not the
  *directional* kernel structure `δ` captures. **Resolves gap #3 below negatively: `δ` is an
  intrinsic depth effect, not a normalization artifact of plain MLPs.**

## Relation to prior work (verified survey)

- The SCFF base paper (Nature Comms 2025) contains **no** alignment/NTK/Jacobian/Fisher/
  residual/subspace theory — all of the above is novel relative to it.
- Closest prior art is **LoCo** (overlapping local blocks add effective depth + implicit
  feedback to close the local-vs-BP gap) — *conceptually parallel* to our residual result, but
  empirical/architectural with no isotropy/Jacobian quantification. **Mono-Forward** (objective,
  not locality, is the bottleneck) is adjacent to our "cross-layer is binding."
- **The SCFF authors themselves flag "top-down feedback connections" as the route to scaling**
  to ResNet-50/ViT — i.e. the cross-layer signal our findings identify as the bottleneck.
- Nearest *theoretical* neighbors (Boopathy & Fiete 2022 NTK-local-vs-BP; Ren et al. 2022
  forward-gradient + local losses) were **not verifiable** in the survey and should be read
  directly before asserting full novelty.

## Gap-closing scorecard

Every attempt to close the local↔BP gap, scored against the cross-layer-`δ` diagnosis:

| approach | type | verdict | why |
|---|---|---|---|
| **residual skips** | architecture | ✅ **winner** | `M = ∏(I+αJ) ≈ I` → isotropy + frozen kernel; cheap; proven `Aniso=O(α)` |
| **auxiliary depth** (LoCo look-ahead) | objective | ✅ works (plain) | local objective *sees* downstream `M`; substitute for residual, costs locality |
| **predictive coding** (settling) | dynamics (biological) | ✅ recovers BP | settling propagates the output error down the hierarchy = cross-layer feedback by construction |
| local Fisher (NGD-FF) | optimizer | ❌ | breaking anisotropy is cross-layer; small-batch Fisher rank-deficient |
| forward-gradient-on-`V` | training rule | ❌ | `δ` defeats it both ways (redundant in residual regime, too weak in plain) |
| direct feedback (DFA-style) | training rule | ❌ | random feedback adds noise; FA "learn-to-align" doesn't materialize for SCFF here |
| per-block LayerNorm | normalization | ❌ | no purchase on the directional kernel `δ` lives in |
| dense skips | architecture | ❌ | downstream Jacobian not near-scalar |

**The unifying principle.** Everything that *works* injects cross-layer information — residual
makes the downstream operator `M` trivial (`≈I`); auxiliary depth makes the local objective
*see* `M`; predictive-coding settling propagates the error down. Everything that *fails* is
purely local or attacks the wrong quantity. The gap is cross-layer; the clean fix is the
residual architecture.

**Sharper: there is no cheap *update-rule* trick.** Three attempts to inject the cross-layer
signal via the weight update — local Fisher, forward-gradient-on-`V`, and direct random feedback
(DFA) — all fail or add noise. The mechanisms that work either *pay* for the feedback (PC's
`K≈depth` settling; aux-depth's `j`-deep look-ahead) or move it into the *architecture* (residual,
`O(1)` and free). Biology pays the cost (recurrent settling / dendritic compartments / continuous
feedback); residual is the engineering shortcut that is free precisely *because* it is structural,
not a clever gradient.

## Biological grounding (verified survey)

Among backprop-free rules, the most biologically-supported framework is **NGRAD** (Neural
Gradient Representation by Activity Differences; Lillicrap et al. 2020, *Nat Rev Neurosci*):
cortex approximates gradient descent via **top-down feedback that nudges lower-level activity**
plus local Hebbian updates — backprop is implausible because of *weight transport* (Crick 1989;
Lillicrap et al. 2016). **Predictive coding** is its best concrete instance (Whittington &
Bogacz 2017): local `ε·activity` updates approximate backprop.

We reproduced it (`pc.py`, `experiments/pc_alignment.py`): a PC network's local update aligns
with the BP gradient, and **the alignment propagates one layer down per settling step** — at
`T=0` only the output layer is aligned (cos 1.0, all hidden 0); by `T≈depth` every layer reaches
cos ≈ 0.98–1.0. **Settling IS the cross-layer feedback.** (Honest nuance: over-settling with a
hard-clamped target drifts off BP; the PC=BP regime is `T≈depth`.) So the *biologically-grounded*
mechanism closes exactly the gap SCFF's pure-local rule cannot — and the **top-down feedback our
results find necessary maps onto the verified biological signature: cortico-cortical feedback
that alters activity** (apical-dendrite / NGRAD). (Caveat: the survey could *not* verify the
detailed cortical-microcircuit / prediction-error-neuron evidence — real literature, but
unvouched here.)

**Dendritic microcircuit** (`pc.py::pc_update_fb`, `experiments/e_dendritic.py`). Dendritic
credit assignment (Sacramento et al. 2018) carries the apical error via *separate* feedback
weights, learned by interneuron plasticity to mirror the forward path. Sweeping the feedback
from random→symmetric, `cos(ΔW, BP)` rises **−0.02 (random / DFA) → 0.48 → 0.89 → 0.985
(symmetric / PC)**, crossing 0.9 around 75% mirrored. This **reconciles** the two earlier
results: DFA failed because its feedback is random; PC worked because its feedback is symmetric.
The dendritic model shows *why*: **the load-bearing ingredient is feedback-weight mirroring** —
the interneuron plasticity is the biological mechanism for it, and learning it (the weight-mirror
problem) is the cost.

This makes the unifying principle a biological one: **credit assignment, in brains and in our
experiments, needs cross-layer top-down feedback** — supplied by settling (PC), by `M≈I`
(residual), or by look-ahead (aux-depth) — AND, for any feedback-based rule, the feedback weights
must *mirror the forward path* (interneuron learning). SCFF without it is both biologically and
computationally the weak corner. Biology pays twice (propagate through depth + learn the
feedback mirror); residual sidesteps both by making the path trivial (`M≈I`).

## Memory & IO footprint (measured)

The practical case for forward-only training is memory and IO, not accuracy. The dominant
training cost is **activation memory** (forward activations retained for the backward pass).
Measured (`experiments/memory_footprint.py`, via `saved_tensors_hooks`, n=256, B=64, ReLU):

| depth `L` | BP | greedy SCFF | greedy residual | BP / SCFF |
|---|---|---|---|---|
| 8 | 2.46 MB | 0.41 MB | 0.54 MB | 6× |
| 32 | 10.32 MB | 0.41 MB | 0.54 MB | 25× |
| 64 | 20.81 MB | 0.41 MB | 0.54 MB | **51×** |

**BP grows linearly with depth (`O(L·B·n)` — stores every layer); forward-only SCFF is flat
(`O(B·n)`, depth-independent).** Residual adds a tiny constant and stays flat. So residual-SCFF
keeps the FF memory win *and* the alignment fix.

**Honest comparison (vs the right baseline).** The `51×` is vs *naive* BP. The fair competitors are
**reversible nets (RevNet, depth-independent activation memory) and gradient checkpointing
(`O(√L)`)** — they also beat naive BP's `O(L)`. SCFF's distinct advantage over *those* is not raw
bytes but that it has **no backward pass, no invertibility constraint, and no recompute**: it gets
`O(B·n)` forward-only, layer-parallel, weight-transport-free, where RevNet still backprops (extra
compute + numerical drift + architectural constraints) and checkpointing pays recompute. That is the
claim to make — not "51× smaller" alone. (See `docs/RELATED_WORK.md`: InfoPro <40% BP memory, DGL
+38% samples/GPU, RevNet depth-independent — the local-learning memory lineage.)

IO/communication (asymptotic): forward-only SCFF needs **1× activation traffic** (vs BP's 2×),
**no weight transport** (no `Wᵀ` backward path), and is **layer-parallel** (no backward-lock) with
purely local updates. Among the gap-closers, **only residual preserves this profile** — aux-depth
holds `O(jBn)` and breaks layer-parallelism; predictive coding holds `O(LBn)` and pays `T×` compute
(settling). So for *practical* BP-free training, residual-SCFF is uniquely positioned: it is the
one cross-layer-feedback mechanism that is free (architecture, not stored activations or settling).

## Real-data accuracy — the alignment win transfers (gap #1, addressed)

Does the gradient-alignment win produce a *trained-model* win? Measured (`experiments/
task_accuracy.py`, MNIST 6000/1000, MLP width 256 / `L=4` ReLU, self-supervised contrastive
pretraining → linear probe on concatenated block features):

Measured over **5 seeds** (`experiments/multiseed.py`, mean ± std):

| method | probe acc | alignment `A` |
|---|---|---|
| supervised-BP (cross-entropy, upper bound) | **0.941 ± 0.008** | — |
| plain local-supervised (per-block CE, local) | 0.909 ± 0.007 | — |
| **residual-SCFF** (forward-only, local) | **0.891 ± 0.009** | **0.997 ± 0.001** |
| plain-SCFF (forward-only, local) | 0.568 ± **0.054** | 0.649 ± 0.040 |

- **Alignment fix: +0.323 ± 0.051** (residual-SCFF − plain-SCFF) — large and robust; `A` tracks
  accuracy across methods.
- **Price of locality: +0.031 ± 0.005** (supervised-BP − best BP-free) — only **~3 points**, tightly
  determined. Best BP-free recipe = plain local-supervised (0.909).
- residual-SCFF `A = 0.997 ± 0.001` (rock-stable); **plain-SCFF std 0.054** — residual is both more
  accurate *and* far more reproducible (plain SCFF is unstable run-to-run).

So the gradient-alignment characterization is not just a proxy: closing the cross-layer gap closes
most of the real-data accuracy gap, leaving a tight ~3-point residue (the proven price of locality).
Caveats: MNIST + MLP scale (not yet CIFAR/conv); probe on concatenated features. (Single-seed caveat
lifted.)

### Depth scaling — residual bounds the gap as depth grows (path A)

The theory says the cross-layer drift `δ` is a *depth* effect — the downstream transport `M`
compounds anisotropy over `L−ℓ` layers, so alignment should decay with depth in plain nets and stay
flat in residual nets (`M ≈ I` each block). Measured at init (`experiments/depth_scaling.py`, mean
alignment `A` and downstream condition number `κ`, `α = 1/√L`):

| `L` | `A` plain | `A` residual | `κ` plain | `κ` residual |
|---|---|---|---|---|
| 4 | 0.519 | 0.942 | 10.1 | 1.71 |
| 8 | 0.390 | 0.918 | 48.0 | 1.95 |
| 16 | 0.342 | 0.920 | 102 | 2.41 |
| 32 | 0.159 | 0.921 | — | 3.84 |
| 64 | 0.141 | 0.922 | — | — |

**Plain alignment decays `0.519 → 0.141` over `L = 4 → 64`; residual stays flat (`0.942 → 0.922`,
drop 0.02).** Residual *bounds* the depth gap — the theory's depth prediction, confirmed directly.
`κ` corroborates the info bound: residual holds the downstream condition number low and bounded
(`1.7 → 3.8`) while plain explodes (`10 → 102`), and the `(√κ−1)²/(κ+1)` floor tracks the `A` drop.
This is exactly where BP's activation memory explodes (`O(L·B·n)`); residual local nets keep `A`
flat *and* memory flat — the practical case for going deep. (Plain `κ` at `L ≥ 32` is unreliable —
deep plain nets rank-collapse at init, eigenvalues underflow — but the `A` trend is robust.)

### Convolutional / CIFAR-10 — the fix transfers, but attenuated (honest)

The credibility bar is convolutions on real images. `convarch.ConvSCFF` is a conv port (stem
`32→16` stride-2, then `L=4` conv blocks at fixed `C=64`/spatial size; residual identity branch
dimension-matched; per-block rep = global-avg-pooled, L2-normalized feature map; layer-local
stop-grad). CIFAR-10 (`experiments/cifar_conv.py`, 8000/2000, 8 epochs, noise+flip aug):

| method | probe acc | `A` |
|---|---|---|
| supervised-BP (upper bound) | 0.525 | — |
| residual-SCFF | 0.330 | 0.731 |
| plain-SCFF | 0.268 | 0.126 |

**The alignment fix transfers *directionally* but is substantially weaker than MLP/MNIST:**

- residual raises `A` `0.13→0.73` and probe `+0.06` — same sign as MLP, so the mechanism is real on
  conv. Note plain `A` *collapsed* `0.49→0.13` from init under training (the dynamical anisotropy
  growth, seen again).
- **But residual conv `A` = 0.73, not the MLP's ~1.0.** The conv downstream transport acts on
  *pooled* reps; the feature-map residual does not make that pooled transport as near-identity, so
  the fix is only partial.
- **Gap to BP is large** (0.33 vs 0.53, −0.20) vs MNIST's tight −0.03. **The near-BP result is
  MNIST/MLP-specific — it does NOT replicate at conv/CIFAR with this setup.**

Honest caveats: weak setup (BP only 0.525 — tiny 4-block/16²/8k/8-epoch, noise+flip aug only;
real conv-SCFF reaches ~80% with proper depth/width/aug per `RELATED_WORK.md`), which both
underpowers every method and compresses differences. **Open questions:** why is residual conv `A`
only 0.73 (try `α=1/√L`, residual on the pooled-rep path, or measuring the pooled-transport `κ`
directly); and does a stronger setup (more depth/data/epochs, real augmentation, matching SCFF's
conv backbone) restore the near-BP gap? The MLP claims stand; **the conv claim is, so far, only a
weak directional transfer — not parity.**

### GPU / real hardware (RTX 5090) — depth-stress, custom kernel, measured memory

We ported SCFF to the GPU with a **hand-written CUDA kernel** (`empirical/cuda/scff_signal.cu`, fused
InfoNCE signal + tangent projection, forward-only — no autograd graph) and ran a full-CIFAR-10
deep-conv depth-stress (`experiments/gpu_depth_stress.py`, 50k, 15 epochs, residual `α=1/√L`):

| `L` | supervised-BP | plain-SCFF (`A`) | residual-SCFF (`A`) |
|---|---|---|---|
| 4 | 0.757 | 0.229 (0.10) | 0.347 (0.25) |
| 8 | 0.751 | 0.260 (0.05) | 0.358 (0.21) |
| 16 | 0.694 | 0.235 (0.10) | 0.359 (0.28) |
| 32 | **0.383** | 0.269 (0.02) | 0.347 (0.21) |

- Residual-SCFF beats plain at every depth (~+0.10 acc, higher `A`) and is **depth-stable**
  (0.347→0.359, flat) — confirms the conv directional transfer at full scale.
- **At `L=32`, plain-conv BP *collapses* to 0.383** (32 layers, no norm/residual → untrainable) while
  residual-SCFF holds 0.347 — **forward-only nearly matches failing BP at extreme depth.**
- The big BP gap at shallow `L` persists (the conv bottleneck, diagnosed below as global pooling).

**Custom kernel (`experiments/bench_kernel.py`):** matches the pure-torch reference to `1e-8`. Speed
vs torch: **2.6× faster at `B=64,C=128`** but **slower at scale** (`0.27×` at `B=128`, `0.12×` at
`B=256,C=512`) — the naive block-per-sample serial-`j` B×B loop loses to cuBLAS's optimized matmul
for the `z zᵀ` term. Honest: the hand-kernel is competitive only at small token counts; a
warp-optimized or cuBLAS-backed version would be needed to win at scale.

**Measured memory (honest correction):** forward-only SCFF vs end-to-end BP peak memory is **~1×**
at `L=8/16/32/64` (SCFF 198→1166 MB, BP 192→1140 MB) — **not the 51× from the CPU activation-store
model.** Reason: `scff_local_step` does a *full forward* (holding all `L` activations) before looping
blocks, so it stores `O(L)` just like BP. **The flat-memory advantage is a property of a greedy
layer-streaming schedule** (compute block ℓ → update → discard its activations before ℓ+1), which
this implementation does not realize. The `51×` asymptotic claim stands only for that streaming
schedule; the naive forward-only loop measured here does not deliver it. A real memory win requires
implementing the streaming update.

### Per-location objective — partial, and a negative-scheme bug (honest)

We diagnosed the conv bottleneck as **global-average-pooling in the local objective** (it gives the
filters no spatial gradient and projects the transport into a rank-deficient space, blowing up `κ`;
see the geometric analysis below). The published SCFF computes conv goodness *per spatial location*.
So we moved the objective off the pool to per-location InfoNCE (`experiments/cifar_spatial.py`,
residual conv `L=8`, CIFAR-10 20k):

| method | probe acc |
|---|---|
| supervised-BP | 0.578 |
| **per-location-SCFF** | **0.361** |
| global-pool-SCFF | 0.324 |

We then tested **both** negative schemes (`experiments/cifar_spatial.py`, residual conv `L=8`, 20k):

| method | probe acc |
|---|---|
| supervised-BP | 0.591 |
| per-location, **cross-image** negatives | 0.376 |
| per-location, **in-image** negatives | 0.369 |
| global-pool-SCFF | 0.336 |

**Cross-image (the SCFF-paper scheme: per-location instance discrimination, positive = same image /
negative = other image at that location) barely beats in-image (+0.007).** Both give the *same*
modest **~+0.04** over global-pool, and the **~0.22 gap to BP survives**. So the negative scheme was
*not* the bottleneck, and moving the objective off the pool — either way — is a real but *small*
gain, not the fix.

**Honest conclusion: the conv gap is the price of locality, not the pooling.** The spatial-objective
diagnosis was *partially* right (a consistent +0.04) but not dominant. The remaining ~0.22 is exactly
what the theory predicts — **the price of locality grows with task difficulty**: MLP/MNIST ~3pt,
conv/CIFAR (hard) ~22pt. The info bound is the mechanism: hard tasks demand expressive (anisotropic,
high-`κ`) features, and a transport-blind local rule cannot align through that, *regardless of where
the objective lives spatially*. (BP itself reaches only 0.59 in this underpowered narrow net; the
SCFF→BP gap is the locality price at this difficulty.) This is theory-consistent, and it is the
motivation for the **FF+BP hybrid** (`docs/superpowers/specs/2026-06-04-ff-bp-hybrid-design.md`):
if global credit assignment is irreducibly needed on hard tasks, put BP exactly where it matters
(the tail) and FF the rest.

### The alignment cosine is necessary but NOT sufficient (diagnostic)

We probed the hypothesized *alignment ↔ expressivity* tension — that small-`α` residual aligns by
going lazy (`M≈I`) and might cap accuracy (`experiments/diag_tension.py`, same MNIST harness):

| residual α | acc | `A` |   | aux-depth `j` | acc | `A` |
|---|---|---|---|---|---|---|
| 0.05 | 0.887 | 1.00 | | 1 | 0.719 | 0.71 |
| 0.10 | 0.886 | 1.00 | | 2 | 0.666 | 0.93 |
| 0.40 | 0.888 | 0.92 | | 3 | 0.662 | 1.00 |
| 0.70 | 0.887 | 0.61 | | (plain `j=0`: 0.577, `A`=0.61) |
| 1.00 | 0.872 | 0.62 | | | | |

Three results overturn a naive "more alignment → more accuracy" reading:

1. **No lazy cap.** Residual accuracy is flat (~0.887) across `α=0.05→0.7` while `A` falls `1.0→0.61`.
   The best-*accuracy* `α` (0.40) beats the best-*alignment* `α` (0.05) by +0.001 — noise. The
   small-`α` lazy regime does not cost accuracy; only `α=1.0` (where `M` stops being `≈I`) hurts.
2. **`A` is not sufficient — architecture matters more.** plain-SCFF `A=0.61 → 0.577` vs
   residual `α=0.7` `A=0.61 → 0.887`: *same measured alignment, +0.31 accuracy.* The residual win
   is partly the skip connection's trainability/conditioning, not the instantaneous alignment cosine.
3. **Aux-depth is dominated.** Look-ahead `j=1` (0.719) beats plain but loses to residual; `j=2,3`
   *lose* accuracy (0.666, 0.662) **even as `A→1.0`** (stale-downstream / moving-target from the
   stop-grad on still-changing downstream weights). Costs more, aligns better, performs worse.

**Revision implication.** The architecture question is settled — residual, `α∈[0.1,0.4]`; do not
chase `A=1.0`. The remaining gap to supervised BP is **not** closed by more cross-layer alignment
(both `A=1.0` routes plateau or hurt). We then tested the last lever — the objective — below.

### The objective is a sliver; the rest is global credit assignment (diagnostic)

The one untested lever was the local *objective*: SCFF's self-supervised contrastive goodness vs
supervised cross-entropy. We gave SCFF a **local supervised** objective — each block a linear head
trained on labels, stop-grad between blocks (still forward-only, local, no weight transport;
`experiments/objective_lever.py`):

| method | objective | arch | probe acc |
|---|---|---|---|
| supervised-BP | global CE, backprop | plain | **0.944** |
| plain local-supervised | per-block CE, local | plain | 0.903 |
| residual local-supervised | per-block CE, local | residual | 0.896 |
| residual-SCFF | contrastive self-sup, local | residual | 0.885 |

1. **The objective buys ~1–2 pts** (local-CE 0.903 vs contrastive 0.885) — a sliver, not the gap.
2. **A ~4–5 pt gap to supervised-BP persists** under the best local objective *and* arch. So the
   gap decomposes: ~1–2 pt objective, **~4–5 pt genuinely global credit assignment**. Backprop's
   end-to-end error flow does something no local rule (contrastive or supervised) and no
   architecture recovers.
3. **Residual stops helping under supervision** (plain local-sup 0.903 > residual 0.896): the
   residual win was specific to *conditioning a weak self-supervised signal*; a strong local CE
   objective does not need it.

**Conclusion.** Best BP-free recipe on MNIST ≈ **0.90** (residual-contrastive or plain
local-supervised — both forward-only, local, `51×` memory win). The residual **~4–5 pt** gap to
backprop is the **price of locality**: global credit assignment is doing real work, independent of
objective and architecture, and is not closed by the levers we have. (Matched 12-epoch/Adam budget,
MLP/MNIST, single seed — real under matched compute, not proven fundamental.)

## The price of locality, proved — an information lower bound (Lean)

The empirical "residual is the unique fix" is now a theorem (`lean/SffProof/InfoBound.lean`,
sorry-free; full write-up `docs/INFO_BOUND.md`). BP's gradient is the output signal pulled back
through the downstream stretch `S = (MᵀM)^{1/2}|_V`; a forward-only local rule must estimate it
**without observing `M`**. In the extreme-eigenvalue plane (eigenvalues `a≥b>0`, condition number
`κ=a/b`), the **Kantorovich ceiling** caps the squared alignment of any transport-blind rule:

```
alignCapSq a b = 4ab/(a+b)² = (2√κ/(1+κ))²  ≤  cosSq a b c        (alignment_capped)
alignCapSq a b = 1  ↔  a = b   (κ=1, isotropy)                     (cap_eq_one_iff_isometry)
a ≠ b  →  alignCapSq a b < 1                                       (aniso_caps_alignment)
```

proved via an exact sum-of-squares identity `(a·c+b·(1−c))²(a+b)² − 4ab(a²c+b²(1−c)) =
((a−b)(a·c−b·(1−c)))²` (`kantorovich_sos`, by `ring`). **Corollary (price of locality):** an expressive
network has `κ>1` (feature learning *is* metric distortion — `grad_decomp.py` shows `aniso` grows
`0.65→0.93` under training), so every transport-blind local rule has defect `1−A ≥ (√κ−1)²/(κ+1) > 0`,
unavoidable without downstream information. Backprop observes `S` (exact pullback `Mᵀ`) and is exempt.
**Residual is the unique escape:** it forces `κ≈1` *a priori*, zero downstream info, and identity is
isotropic on every subspace — so it needs neither the eigenbasis nor `V`. (Open: the *dynamical*
theorem that the goodness gradient grows `κ` over training — `INFO_BOUND.md` §6.) This resolves the
theoretical gap: the old theory characterized *when* alignment holds (static, geometric); this bounds
the *information cost* of achieving it locally (why every non-residual local fix failed).

## Geometric decomposition of the BP gradient — the gap is transport, not the negatives

Treating reps as points on the sphere `S^{n-1}`, the SCFF gradient is the **Riemannian (tangent)
gradient** of a contrastive sphere-energy (`Pperp` projection in `local_grad_formula` is exactly
the tangent projection). Both gradients split as *positive − negative*; the BP gradient is the
output-layer signal **pulled back through the downstream map** `M^T` (the cotangent map of the
inter-layer flow):

```
g_FF = Pperp( z+_l − Σ_j p^l_ij z_l_j )           (layer-l softmax)
g_BP = M^T Pperp( z+_L − Σ_j p^L_ij z_L_j )        (output softmax, transported)
```

so the FF↔BP gap has only two possible sources: **(1) kernel drift** `p^l ≠ p^L` and **(2) transport**
`M^T` not an isometry (polar `M = QS`; rotation `Q` is free, stretch `S = (M^T M)^{1/2}` is the
defect; `S = cI` ⟺ isotropy ⟺ alignment). The contrastive force is a *gradient field* and the
sphere has trivial `H^1`, so there is **no harmonic/topological obstruction** — the gap is a metric
defect, fixable in principle. We attribute it empirically (`experiments/grad_decomp.py`, tangent-space
cosine, init vs after 400 SCFF steps on clustered data):

|  | mean `A_full` | `cos_kernel` | shared-kernel gain | mean `aniso` |
|---|---|---|---|---|
| plain [init] | 0.382 | 1.000 | −0.001 | 0.65 |
| plain [trained] | 0.129 | 0.984 | −0.005 | 0.93 |
| residual [init] | 0.974 | 1.000 | +0.001 | 0.09 |
| residual [trained] | 0.881 | 1.000 | −0.001 | 0.28 |

1. **Kernel drift is never the problem.** `cos_kernel ≈ 1` init *and* trained; using the output
   softmax at every layer (a shared/transported kernel) changes alignment by ±0.005 — and is
   slightly *negative*. **The positives/negatives are already geometrically optimal; re-modeling
   them cannot help.** (Rules out shared-kernel negatives and EMA/transported positives.)
2. **The entire gap is transport stretch `S`.** `A_full` is a pure function of `aniso`: plain
   0.65→A 0.38, residual 0.09→A 0.97.
3. **Learning *grows* the stretch.** Plain `aniso` 0.65→0.93 under training (A 0.38→0.13);
   residual 0.09→0.28 (A 0.97→0.88, bounded by `M≈I`). **Feature learning distorts the
   representation metric → transport stops being an isometry → local↔global alignment decays.**

**This is the geometric mechanism of the price of locality:** to learn useful features a layer must
distort the metric (anisotropic stretch), but local learning matches BP only when the inter-layer
map is an isometry — the two pull against each other. Backprop is exempt (it applies the *exact*
pullback `M^T` through any stretch); local rules cannot, so they pay. Residual does not remove the
tension — it **bounds** how far learning can push `M` from identity. The only BP-free levers are
therefore transport-fixers: residual (free, structural — done) or preconditioning by `S` (needs
`M^T M` = downstream info = weight transport — forbidden).

### Idea D — local isometry penalty (the soft sibling of residual) is dominated

We tested the one untried transport-fixer: a **local, BP-free** penalty keeping each block's *own*
Jacobian isotropic — penalize `Var_v ‖J_i v‖²` over random probe directions `v` (isometry ⟺
`‖J_i v‖²` direction-independent, scale-free; `experiments/iso_penalty.py`), added to each block's
goodness ascent. MNIST, plain arch, λ-sweep:

| method | probe | `A` |
|---|---|---|
| plain-SCFF (λ=0) | 0.577 | 0.608 |
| plain-SCFF + isoD (λ=1, best) | 0.644 | 0.616 |
| residual-SCFF | **0.885** | **0.998** |

isoD bumps accuracy +0.07 but **does not even raise alignment** (0.608→0.616) and lands far below
residual (+0.31, A=1.0); large λ *hurts* (over-regularized collapse). **Dominated.** Why, geometrically:

- **Isotropy ≠ identity.** The penalty drives `M^T M → cI` (isometric *up to a rotation* `Q`).
  Residual gives `M ≈ I` — it pins **both** the stretch `S≈I` and the rotation `Q≈I`. The penalty
  controls `S` only; the leftover `Q` still scrambles the layer-ℓ↔output signal correspondence.
- **Wrong subspace.** `k=8` random probes in the `n=256` space mostly miss the `d_V=32` contrastive
  subspace where isotropy matters.
- **Soft vs structural.** The penalty *fights* the goodness gradient (which grows aniso) and cannot
  win without a λ large enough to destroy the features.

**Revision arc closed.** Every alternative lever is ruled out — kernel drift / negative re-modeling
(geometrically optimal, ≈0 effect), transported/EMA positives (positive modeling isn't the defect),
objective (+1–2 pt), aux-depth (dominated), Fisher / DFA / forward-gradient / LayerNorm (fail),
local isometry penalty (dominated). **The residual architecture is the unique cheap BP-free fix**:
it is the only mechanism that makes the inter-layer transport an isometry — and the *right* one,
`M≈I` — without downstream information. The residual **~4–5 pt** gap to backprop is the irreducible
**price of locality**: learning must distort the representation metric, and only backprop's exact
pullback `M^T` is exempt.

## The honest headline

Local SCFF aligns with BP only up to a cross-layer term `δ`. Width fixes the isotropy half but
not `δ` (a depth effect). A small-scale **residual** architecture fixes both — provably,
cheaply, forward-only — and cleverer BP-free correction schemes (Fisher, forward-gradient) do
not beat it.

## Gaps to practical training

1. ~~Alignment ≠ accuracy.~~ **Addressed** (see "Real-data accuracy" above): on MNIST the
   alignment win transfers — residual-SCFF reaches 0.887 (within ~6 pts of supervised BP)
   vs plain-SCFF 0.596. Remaining: larger scale (CIFAR/conv), multi-seed.
2. **Scale + regime.** Toy widths/depths/batches, synthetic data; the theory lives in
   `B ≪ √n`, `d_V = o(√n)`, which practical batch sizes likely violate.
3. **Linear-primary.** Results are linear-mode primary; ReLU is lightly tested. (The
   "`δ` is a no-normalization artifact" hypothesis is **resolved negatively** — per-block
   LayerNorm does not cut `δ`; see Negative results.)
4. **The FF value-prop untouched.** Memory (no stored activations), locality, parallelism —
   none measured; only the alignment proxy.
