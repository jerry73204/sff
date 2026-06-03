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

## Minimal first experiment (when built)

MNIST first (where forward-only ≈ BP, so the generative machinery is tested without the conv price-of-locality confound). `z=[x;e_y]`, small MLP, per-layer denoising-score goodness, short-run Langevin negatives. Compare to: (a) SCFF with in-batch negatives, (b) BP classifier. Metrics: accuracy, calibration/OOD (the generative-classifier payoff), and whether model-sampled negatives beat in-batch. Then ask if it transfers to conv better than plain SCFF (model-sampled hard negatives may strengthen the weak conv signal).

## Method name

**gen-FF (Local Denoising-Energy Forward-Forward):** forward-only, layer-local joint `(x,y)` energy
model; per-layer denoising-score training; negatives sampled from its own score by short-run local
Langevin; classify by conditioning on `x`. Novel intersection; fixes the negative-data problem;
inherits generative-classifier robustness; stays backprop-free.
