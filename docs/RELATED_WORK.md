# Related work — memory-efficient, backprop-free local learning at depth

Survey for positioning SCFF / residual-SCFF / the information lower bound (`INFO_BOUND.md`) and the
depth-scaling study. Claims verified against sources; **[unconfirmed]** flags items not fully checked.

## Most relevant papers

1. **SCFF — Self-Contrastive Forward-Forward** (Ouldali, Momeni et al., *Nat. Commun.* 16, 2025;
   arXiv:2409.11593). Forward-only layer-local InfoNCE goodness, self-generated pos/neg. MNIST 98.7%
   (2-layer FC, 2×2000 wide), CIFAR-10 80.75% (3-layer conv), STL-10 77.3% (beats BP 77.02%).
   **Our base method. Uses NO residual/skip, only 2–4 layers → our residual + depth angle is new.**
   (Reconcile our MNIST 0.88 vs their 0.987: they use very wide FC.)
2. **The Trifecta: training deeper Forward-Forward nets** (Dooms, Tsang, Oramas, 2023;
   arXiv:2311.18130). FF degrades with depth; 3 techniques restore cross-depth signal → ~84% CIFAR-10.
   **Closest "deeper FF" prior art.** Training tricks, not residual-as-isometry + alignment bound.
   **[exact 3 techniques unconfirmed — check if one is a skip/norm]**
3. **InfoPro — locally-supervised DL by maximizing info propagation** (Wang et al., ICLR 2021).
   Greedy local modules are "short-sighted" — collapse task-relevant info early at depth; info loss
   fixes it. CIFAR/SVHN/STL/ImageNet/Cityscapes at **<40% of BP memory**. **Strongest precedent for
   BOTH our depth-effect (≈ our δ) AND memory framing.** Qualitative; our `1−A ≤ C/√n + C′δ` is the
   quantitative alignment version.
4. **Greedy InfoMax / GIM** (Löwe, O'Connor, Veeling, NeurIPS 2019; arXiv:1905.11786). Gradient-
   isolated modules max InfoNCE between consecutive outputs. Template SCFF inherits; still BP within
   modules, no alignment theory.
5. **Decoupled Greedy Learning of CNNs (DGL)** (Belilovsky, Eickenberg, Oyallon, ICML 2020). Layer-
   local greedy scales to **ImageNet, ResNet-152**, ~BP generalization; **+38% samples/GPU**.
   Canonical "local at scale + memory" baseline; per-module BP (less local than us), no depth theory.
6. **SoftHebb** (Journé et al., ICLR 2023; arXiv:2209.11883). No feedback/target/transport, ≤5 layers.
   MNIST 99.4%, CIFAR-10 80.3%, STL-10 76.2%, ImageNet 27.3%. Forward-only baseline SCFF beats.
7. **Direct Feedback Alignment** (Nøkland, NeurIPS 2016). Random fixed feedback output→each layer.
   Still a *global* error signal (downstream info); SCFF is transport-blind — the regime our bound governs.
8. **Feedback Alignment** (Lillicrap et al., *Nat. Commun.* 2016). Forward weights align to random
   feedback ≈ BP. Origin of "gradient alignment" as a measurable angle — what our `1−A` bounds.
9. **Feedback alignment in deep conv nets** (Moskovitz, Litwin-Kumar, Abbott, 2018; arXiv:1812.06488).
   FA alignment angle **deteriorates in early layers of deep nets** (~45° shallow, worse deep).
   **Direct empirical precedent that alignment degrades with depth** — our δ/bound formalize why.
   Their per-layer angle-vs-depth plot is the template for our depth_scaling figure.
10. **Align, then memorise: dynamics of FA learning** (Refinetti et al., ICML 2021; arXiv:2011.12428).
    Alignment governed by **conditioning of the alignment matrices**. **Closest "conditioning controls
    alignment" theory** — but upper-bound dynamics, NOT a Kantorovich lower bound on the downstream
    Jacobian's κ. Our κ-floor is a different, stronger (impossibility) statement.
11. **Whittington & Bogacz** (*Neural Comput.* 2017). Predictive coding → BP-equivalent updates, but
    with iterative equilibria + top-down error (it *accesses* downstream info). Contrast: transport-
    blind locality cannot, hence our irreducible gap.
12. **ReZero** (Bachlechner et al., UAI 2021; arXiv:2003.04887). `x+α·F(x)`, α init 0 → **dynamical
    isometry** (Jacobian SVs≈1), thousands of layers. **Mechanistic backbone of residual→M≈I→isometric
    transport.** We repurpose from optimization speed to local-vs-BP alignment.
13. **RISOTTO — dynamical isometry for ResNets** (Gao, Saxe et al., 2022; arXiv:2210.02411). Exact
    dynamical isometry for finite ResNets. Cite for "residual = well-conditioned Jacobian."
14. **RevNet — Reversible Residual Network** (Gomez et al., NeurIPS 2017; arXiv:1707.04585). Activation
    storage **depth-independent** by recompute in backward. **The honest memory competitor:** still BP
    (extra compute, invertibility, drift). Our edge = no backward pass, no invertibility, no recompute.
15. **Boopathy & Fiete, "train wide nets without backprop"** (ICML 2022; arXiv:2106.08453). In the
    **NTK/infinite-width limit**, input-weight-alignment BP-free rules ≈ GD, match BP. **= our `C/√n`
    term** (width→local≈global in lazy regime). Our `δ+κ` is what survives *outside* lazy (feature
    learning grows κ). Positioning: locality is free when lazy, costly when learning features.

Secondary: Decoupled Neural Interfaces / synthetic gradients (Jaderberg 2017); Predictive Coding
Approximates Backprop on arbitrary graphs (Millidge et al. 2020, arXiv:2006.04182); Mono-Forward
(Gong et al. 2025, arXiv:2501.09238, local projection heads, MLP-only); PETRA (2024, arXiv:2406.02052,
reversible+parallel local). None give an alignment-at-depth bound.

## Novelty assessment (our three results)

- **(a) residual-as-transport-isometry fix for FF — partially new.** Known: residual→isometric
  Jacobian (ReZero, RISOTTO); FF needs depth fixes (Trifecta, InfoPro). New: skip connection framed
  as the *alignment/credit-assignment* mechanism for forward-only learning + a quantified bound.
- **(b) Kantorovich/condition-number lower bound = price of locality — likely new (most novel).**
  No on-target prior art found. Closest: Refinetti (conditioning, upper-bound, no lower bound),
  InfoPro (qualitative info collapse). A genuine information-obstruction/impossibility result.
  **[do a focused manual check on "isometry/condition-number bounds for bio-plausible credit
  assignment" before asserting priority]**.
- **(c) depth + memory co-scaling law — parts known, joint curve is ours.** Memory wins (InfoPro,
  DGL) and depth-degradation (Moskovitz, Trifecta) exist separately; charting alignment-gap *and*
  memory-advantage together vs `L` is new.

## Pointers for the depth-scaling study

1. Start from SCFF's own conv backbone (CIFAR 80.75% / STL 77.3%) then push L=16/32/64.
2. Credibility bar = DGL/InfoPro protocol: ≥ CIFAR-10/100 + STL with ResNet-32/110; ImageNet +
   ResNet-50/152 is gold. Report memory as peak/ samples-per-GPU, not just asymptotic.
3. Use Trifecta ~84% CIFAR as the FF-at-depth baseline to beat; check overlap with residual.
4. Benchmark memory vs **RevNet/checkpointing**, not just naive BP. Frame the win as no-backward-pass
   / no-invertibility / no-recompute, not raw bytes.
5. **Instrument the bound:** measure alignment `A` and downstream condition number `κ` per layer vs
   depth, ±residual. Residual should flatten the `A`-vs-depth curve; the `(√κ−1)²/(κ+1)` floor should
   predict the residual gap. (Template: Moskovitz 2018 per-layer angle-vs-depth.) — `depth_scaling.py`.

## Biological-plausibility motivation (not technical related-work)

- **Raugel, Seitzer, Szafraniec, Vo, Rapin, Labatut, Bojanowski, Wyart, King — "Misalignment Between
  Backpropagation and the Hierarchy of Brain Responses to Images"** (2026, arXiv:2605.28693, Meta FAIR
  + neuroscience). Maps backprop *gradients* onto fMRI/MEG during vision: forward activations track the
  cortical hierarchy, but the **backpropagated gradients' spatial/temporal organization diverges** from
  any biologically-plausible backprop. Independent brain-data evidence that the cortex does not do
  backprop-style credit assignment. *Relevance:* motivation only (not a method, not our local-vs-BP
  alignment metric) — a recent, high-profile citation for *why* forward-only / local learning is worth
  pursuing, complementing the weight-transport (Crick 1989) and NGRAD arguments. File under intro
  motivation, not technical comparison.
