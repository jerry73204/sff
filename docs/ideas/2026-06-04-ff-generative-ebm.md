# Idea: FF as a local denoising-energy model of (x, y) — "gen-FF"

**Status:** active design direction (FF family). Models the dataset as pairs `(x,y)` on a manifold,
makes FF's goodness the joint energy, classifies by conditioning on `x`, and — the payoff —
**generates its own negatives** by sampling the model, fixing FF's hand-crafted-negative wound.

## The framing

Dataset = pairs `(x,y)` on a manifold `M ⊂ X×Y` (`x` image, `y` label). Joint input `z = [x ; e_y]`
(image + continuous label embedding; FF already overlays the label). FF goodness `G(z)=Σ_ℓ ‖h_ℓ(z)‖²`
read as a **negative energy**: `p(x,y) ∝ exp(−E)`, `E = −G`, high goodness ⇔ on `M`.

- **Classification = conditioning on `x`** (JEM's exact move, partition function cancels):
  `p(y|x) = softmax_y(−E(x,y))`, `y* = argmax_y −E(x,y)`. FF's "embed label, pick max goodness" *is*
  this. `p(x) = LogSumExp_y(−E(x,y))` is a *free* unconditional density over the finite label set.

## Why this is the right inspiration (research-grounded)

FF is already, in Hinton's words, a local contrastive EBM (Boltzmann + NCE + local goodness) — it just
never closed the loop to a *joint* `p(x,y)` or to *model-sampled* negatives. JEM closed it *with
backprop*; EqProp/predictive-coding are local+generative but not framed as a goodness-EBM with
self-sampled negatives. **Verified open intersection (our novelty):** FF's layer-local backprop-free
goodness × joint `(x,y)` energy with LogSumExp conditioning × model-generated negatives. No prior work
combines all three.

## The forward-only generative recipe (two mechanisms that are secretly one)

1. **Train goodness as a LOCAL denoising score (no backprop).** Noise the pair `z̃ = z + σξ`; train
   each layer so its goodness-gradient `∇goodness_ℓ` denoises `z̃` back toward `z` — a per-layer
   regression to `(z−z̃)/σ²` using only the layer's own input/output (denoising score matching,
   Vincent 2011). Goodness becomes a potential whose gradient is the score toward `M`.
2. **Sample negatives by short-run local Langevin on that score.** `∇_x E` is local (no end-to-end
   backward); noise-initialized short-run chains (Nijkamp 2020) + proximal step (JEM++) → hard
   negatives just off the manifold. Replaces SCFF's weak in-batch negatives / Hinton's hybrid images.

**Unification:** with a trained goodness-score, *"generate a hard negative"* and *"denoise `y` onto
`M` given `x`"* are the same operation at different noise levels. One mechanism = training negatives +
conditional `y|x` inference + principled generation. Annealing the noise level = the diffusion view.

## Why the locality problem CHANGES (the biggest structural win)

SCFF trains a local proxy that must *align with a global contrastive loss* — which is where the whole
transport/`κ`/isometry headache came from. gen-FF gives each layer a **self-contained generative
objective** (model/denoise its own input). There is **no global loss to align to**, so the local
gradient has **no downstream Jacobian, no transport term, no isometry requirement.** The locality
question changes from *"align local proxy to BP"* (where `κ` and pooling bit us) to *"do stacked local
generative objectives compose"* — the **DBN / stacked-denoiser** question, answered positively by
Hinton 2006 (greedy layer-wise generative training works, modest gap). This is the structural reason
gen-FF escapes most of our measured failures, not just the negatives.

## Failure-mode audit (vs everything we measured)

| failure we hit | gen-FF status |
|---|---|
| cross-layer drift δ / transport non-isometry | **sidestepped** — local objective self-contained, no downstream transport |
| anisotropy grows under training (`κ`↑, alignment collapses) | **sidestepped** — not aligning to a global gradient; transport `κ` irrelevant to the local update |
| price of locality (~3pt MLP, ~22pt conv) | **changed, likely smaller** — becomes greedy-generative-vs-end-to-end gap (DBN: modest), not the irreducible credit-assignment gap |
| global pooling severs objective from residual fix (conv) | **gone** — no pooling needed; energy is per-feature/per-location |
| weak in-batch negatives | **fixed** — model-sampled hard (or label-swap) negatives near the manifold |
| objective mismatch (contrastive ≠ task) | **fixed** — `y` in the energy; objective *is* `p(x,y)` |
| alignment ≠ accuracy | **moot** — different framework, no reliance on alignment |
| memory ≈ BP unless streaming | **fixed natively** — layers train independently → DBN-style layer streaming realizes flat memory |
| SGLD instability (new risk) | **avoidable in the core** — noising *is* the contrast; no MCMC needed unless you want generation |

## Compute vs BP — the cost knob (must stay cheaper than BP)

BP/step: 1 forward + 1 backward ≈ **2 forward-units**, `O(L)` memory, weight transport, sequential.

**The trap is where the label lives:**
- **Label-in-input (pure joint):** scoring `E(x, e_y)` re-forwards per label → **inference = C× forward**
  (try every class); training ~3× forward. For `C=10`, ~10× BP at inference. **Violates "cheaper than BP."**
  (This is the diffusion-classifier cost — accepted there for robustness, not for cheapness.)
- **Label-at-top (JEM-style energy head):** early layers = label-free FF goodness (forward-only,
  **closed-form local update, no backward** — `local_grad_formula` is an outer product, *strictly
  cheaper than BP*); top layer outputs `C` class-energies in **one forward** → argmax. Inference = 1 forward.

**Negatives:** Langevin/SGLD = `K` forward/step (expensive, > BP) unless **PCD-amortized** (~1 extra
forward); **label-swap / noised negatives = ~1 forward, no MCMC (cheap).**

| config | inference | training/step | memory | transport |
|---|---|---|---|---|
| BP | 1 fwd | 2 fwd-units | `O(L)` | yes |
| **gen-FF cheap** (label-at-top, FF early, label-swap/noise negs, no MCMC) | **1 fwd** | **~1 fwd + closed-form local updates** | **`O(B)` streamable** | **no** |
| gen-FF rich (label-in-input, Langevin) | C× fwd | (1+K) fwd | `O(B)`+buffer | no |

**Verdict:** the **cheap config is strictly cheaper than BP** (no global backward, streamable memory,
no weight transport, layer-parallel, no MCMC). The **rich (full generative) config is *more* expensive
than BP** → it must be **opt-in**, used only when generation/OOD/robustness is worth the cost.

## The cheap design (concrete — the default)

1. **Early layers:** SCFF-style forward-only goodness, closed-form local update, streamable. (Same
   cost as SCFF — already cheaper than BP; the transport theory no longer needs the residual fix
   because there is no align-to-BP step, though residual may still help conditioning.)
2. **Top:** a **joint-energy head** `E(x,y) = −G_top(x)[y]` — the top layer emits `C` goodnesses (one
   per class) in a single forward. Train it as a small EBM: positives = correct `(x,y)`; **negatives =
   wrong labels + noised pairs (cheap, no MCMC)**; `p(y|x) = softmax_y(−E)`, ascend `log p(y|x)` plus a
   margin pushing wrong-label / noised energies up.
3. **Classify in 1 forward** (argmax over the `C` head-energies). No separate probe.
4. **Opt-in generative mode:** add PCD-amortized Langevin negatives + label-in-input *only* for
   generation / OOD / robustness, accepting the higher cost then.

## Closest prior art

- **JEM** (Grathwohl et al., ICLR 2020) — classifier→joint EBM, SGLD negatives, condition on x. Blueprint, but backprop.
- **EGC** (Guo et al., ICCV 2023) — single net = joint `p(x,y)` energy (forward) + denoising score (backward). Cleanest "joint energy + score" template, but backprop.
- **Diffusion Classifier** (Li et al., ICCV 2023) — classify by lowest noise-prediction error across labels. Borrow as an *inference* rule, not for local training.
- **EqProp** (Scellier & Bengio 2017), **Predictive Coding** (Millidge et al.) — local, backprop-free, generative — proof the local-energy training is feasible. Not framed as goodness-EBM-with-self-negatives.

## Pitfalls / mitigations (worse without backprop)

- **SGLD instability** → regularize each layer's energy *locally* + tune θ per layer (no global backprop for a stabilizer).
- **Short-run MCMC ≠ valid density** (Nijkamp) → fine for negatives; do NOT over-claim calibrated `p(x)`.
- **No-backprop variance compounds** → prefer the local denoising-score target over MCMC-in-the-loop; reserve Langevin for negative generation only.
- **Depth instability** (EqProp/PC) → per-layer goodness normalization (already used in SCFF).

## Minimal first experiment (the cheap config)

MNIST first (forward-only ≈ BP there, so the generative machinery is tested without the conv
price-of-locality confound). **Cheap config:** small MLP, label-free FF goodness early layers
(forward-only), a joint-energy top head `E(x,y)=−G_top(x)[y]` (C class-energies, one forward), trained
with **wrong-label + noised negatives (no MCMC)**. Classify in 1 forward (argmax over the C energies).

Compare to: (a) SCFF + linear probe (our current method), (b) BP classifier (ceiling). Metrics:
- **accuracy** — does the joint-energy head + cheap negatives beat SCFF's probe? (objective-alignment win)
- **cost** — confirm ≤ BP (1-forward inference, forward-only training, streamable memory)
- **calibration / OOD** — the generative-classifier payoff, even in the cheap config (LogSumExp `p(x)`)

Then: (1) does it transfer to conv better than plain SCFF (task-aligned energy may dodge the
global-pool wound)? (2) opt-in: add PCD-Langevin negatives — do harder model-sampled negatives lift it
further, and at what cost?

## Method name

**gen-FF (Generative Forward-Forward):** forward-only, layer-local joint `(x,y)` energy model.
**Default (cheap):** FF goodness early + a one-forward joint-energy top head, wrong-label/noised
negatives, no MCMC — *cheaper than BP*, classify in 1 forward, no probe. **Opt-in (rich):**
label-in-input + PCD-Langevin model-sampled negatives + generation — richer, costs more than BP.
Novel intersection (FF-local × joint-`(x,y)`-EBM × self-generated negatives); fixes SCFF's weak
negatives + objective mismatch; sidesteps the transport/`κ`/pooling failures by making the local
objective self-contained generative.
