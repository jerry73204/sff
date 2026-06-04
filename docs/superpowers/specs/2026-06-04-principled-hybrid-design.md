# Principled FF/BP hybrid (fine-tuning) — design spec

**Revised project scope:** Find a training method that **matches BP accuracy at less memory and less
compute**, by spending global credit assignment (BP) only where local learning provably fails (high
downstream `κ`) and using cheap forward-only local learning elsewhere. Pure forward-only cannot match
BP on hard tasks (proven: the information lower bound / price of locality; measured: the conv wall).
The hybrid keeps the forward-only performance wins (no backward graph, layer-parallel, low memory) on
the layers that don't need BP, and applies BP only to the layers that do.

**This spec — the fine-tuning/transfer instantiation (the practical win):** under domain shift, early
features are generic (low `κ`) so FF can adapt them label-free; only the tail needs BP. Target:
match full-BP fine-tuning accuracy at the *frozen-baseline's* memory/compute.

## Setup

1. **Pretrain** a conv backbone + linear head with BP on **clean CIFAR-10** → transferable features.
2. **Target = shifted CIFAR-10**: a fixed on-the-fly corruption (gaussian blur + additive noise),
   same 10 labels, new input distribution. (No extra dataset.)
3. **Fine-tune to the target three ways**, from the same pretrained init, and compare.

## The three arms

| arm | layers `≤k` | layers `>k` + head | role |
|---|---|---|---|
| **full-BP fine-tune** | BP | BP | ceiling (max memory/compute) |
| **freeze + BP-tail** | frozen | BP | cheap baseline (no early adaptation) |
| **hybrid: FF-early + BP-tail** | **forward-only gen-FF denoising** (label-free) | BP | the method |

**Win condition:** hybrid accuracy ≈ full-BP-fine-tune accuracy, **at the freeze baseline's peak
memory and compute** (both only backprop the tail). I.e. FF's label-free early adaptation to the new
domain recovers what freezing discards, almost for free.

## The hybrid mechanism

Split at block `k`. Per fine-tuning batch on the corrupted target:
1. **FF-early:** `conv_denoise_step` on blocks `0..k` (per-location denoising-energy, forward-only,
   no backward graph, updates early filters to the target manifold — label-free).
2. **BP-tail:** forward through all blocks (early now updated), **detach at the split** so no gradient
   crosses into the early layers; backprop the cross-entropy through blocks `k+1..L` + head.
The backward graph spans only `L−k` blocks → less compute (no backward on `0..k`) and less peak
memory (no stored graph for `0..k`) than full BP.

## Metrics (per arm, and per `k` for the sweep)

- **target test accuracy** (head argmax, 1 forward).
- **peak GPU memory** during fine-tuning (`torch.cuda.max_memory_allocated`).
- **compute** proxy: backward-block count (`full-BP=L`, `hybrid=freeze=L−k`) + wall-clock fine-tune time.
- **κ analysis (the principled cherry):** measure per-layer downstream anisotropy on the pretrained
  net; check that high-`κ` layers cluster in the tail (justifying BP there) and that the `κ` knee
  predicts the accuracy-vs-`k` knee. Secondary to the accuracy-vs-cost curve.

## Headline output

Accuracy-vs-`k` curves for **hybrid** and **freeze** (with **full-BP** as a flat ceiling line), plus
peak-memory and time vs `k`. The win is a region where **hybrid ≈ full-BP at freeze cost** — and FF
early adaptation visibly beats freezing at the same `k`.

## Components / files

- `empirical/hybrid.py` (create) — `corrupt(x)` (fixed blur+noise domain shift); `pretrain(...)`
  (BP on clean); `finetune_full_bp`, `finetune_freeze_tail`, `finetune_hybrid` (the three arms,
  split at `k`); `eval_acc`; `layer_kappa(model, x)` (per-layer downstream anisotropy, secondary).
- `empirical/experiments/hybrid_cifar.py` (create) — pretrain once, run the three arms over a `k`
  sweep on the corrupted target, report accuracy / peak-mem / time / κ.
- `empirical/tests/test_hybrid.py` (create) — `corrupt` shape/finite; hybrid step updates early
  blocks (FF) and tail+head (BP) but **no gradient crosses the split** (early params unchanged by the
  BP loss); `finetune_freeze_tail` leaves frozen blocks unchanged.

Reuse: `gpu_arch.ConvSCFF` (+ `.features`, `apply_block`), `genff_conv.conv_denoise_step`,
`genff.EnergyHead`, `gpu_pipeline` (CIFAR load, `augment_batch`, metrics).

## Success criteria

- Pipeline pretrains once, runs all three arms over the `k` sweep on GPU, reports acc/mem/time/κ.
- **Headline:** at some split `k`, **hybrid ≈ full-BP-finetune accuracy** (within ~2 pts) at the
  **freeze baseline's memory and compute** (clearly < full BP), and **hybrid > freeze** at that `k`
  (FF early adaptation earns its keep). If hybrid never beats freeze → honest negative (FF early
  adaptation doesn't help under this shift); if it matches full-BP only at `k=0` (all BP) → no win.
- κ analysis: report whether high-`κ` layers concentrate in the tail (principled-allocation evidence).

## Risks / mitigations

- **FF-early adaptation may not help** (freeze already fine if the shift is mild). Mitigation: make
  the corruption strong enough that frozen features genuinely degrade (so there's room to recover);
  sweep corruption strength if needed.
- **gen-FF denoising signal is weak at init** (found earlier) — but here the backbone is *pretrained*
  (not random), so filters already respond to structure; the denoising adaptation starts from a good
  point, which should give a stronger signal than from scratch.
- **Memory measurement honesty:** the hybrid still does a full *forward*; the win is the *backward*
  graph spanning only `L−k`. Report peak memory and the backward-block count; don't over-claim.
- **κ on conv is expensive** (per-sample downstream Jacobian) — compute on a small batch / few layers;
  keep it a secondary analysis, not on the critical path.

## Testing

`pytest` (CUDA-gated where needed): `corrupt(x)` returns `[B,3,32,32]` finite, visibly different from
`x`; one `finetune_hybrid` step changes early-block params (FF) AND tail+head params (BP) while the
BP cross-entropy loss produces **no gradient** into early blocks (split detach verified);
`finetune_freeze_tail` leaves `≤k` params bit-identical.
