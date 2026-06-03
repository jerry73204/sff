# CUDA SCFF on GPU — design spec

**Goal:** Train SCFF fully on the RTX 5090 with a hand-written CUDA kernel for the SCFF-specific hot
path, and run a deep-conv CIFAR-10 **depth-stress** that tests — on real hardware, at real scale —
whether residual bounds the alignment/accuracy gap as depth grows while plain collapses, with the
memory advantage measured. Plus kernel correctness/speed and SCFF-vs-BP memory benchmarks.

**Approach (A):** the `.cu` kernel computes the genuinely SCFF-specific, kernel-worthy op — the B×B
InfoNCE softmax + tangent projection that *is* the local gradient on the rep, forward-only, no
autograd graph. Conv arithmetic stays on cuDNN; the single-block weight gradient uses cuDNN
conv-grad. The kernel is the part that is *ours*; we do not reimplement convolution.

## Architecture

Forward-only, layer-local training. Per block `ℓ`:

1. cuDNN conv → ReLU → (residual add) → feature map `y^ℓ` (input `y^{ℓ-1}` **detached** — cross-block
   stop-grad, the locality + flat-memory contract).
2. Global-avg-pool + L2-normalize → `z ∈ [B,C]` (and `z⁺` from the augmented view).
3. **`scff_signal` (custom `.cu`)**: emits `s⊥ = P⊥_z(z⁺ − softmax(zzᵀ/τ)·z) ∈ [B,C]` and the scalar
   goodness. `s⊥` is exactly `∂g/∂z` — computed without any autograd graph.
4. Backprop `s⊥` through **this one block only** (cuDNN conv-grad) → `grad_Wℓ`; ascend.

Each block's update uses only its own forward pass and the kernel output — no global backward, no
weight transport, layer-local. The kernel is the SCFF heart; cuDNN does the conv math.

## Components / file layout

- `empirical/cuda/scff_signal.cu` — the kernel.
- `empirical/cuda/scff_ext.py` — `torch.utils.cpp_extension.load` wrapper exposing
  `scff_signal(z, z_pos, tau) -> (s_perp, goodness)`; caches the compiled extension.
- `empirical/cuda/__init__.py` — re-export `scff_signal`; `available()` guard (CUDA present).
- `empirical/gpu_arch.py` — `ConvSCFF` GPU variant + forward-only local training step using the
  kernel (or a pure-torch reference path when CUDA absent, for CI).
- `empirical/experiments/gpu_depth_stress.py` — the depth-stress experiment.
- `empirical/experiments/bench_kernel.py` — kernel speed + SCFF-vs-BP memory benchmarks.
- `empirical/tests/test_scff_kernel.py` — correctness vs the existing pure-torch reference.
- `empirical/pyproject.toml` — cu128 torch source/index for GPU; CPU fallback preserved.

## The CUDA kernel (`scff_signal.cu`)

Math (matches `gradients.signal` + tangent projection in `local_grad_formula`):

```
scores_ij = (z_i · z_j) / tau           # [B,B], keys = z (detached)
p_ij      = softmax_j(scores_ij)
s_i       = z_pos_i - Σ_j p_ij z_j       # contrastive signal
s⊥_i      = s_i - z_i (z_i · s_i)        # tangent projection onto T_{z_i} S^{C-1}
goodness  = Σ_i [ (z_i·z_pos_i)/tau - logsumexp_j scores_ij ]
```

- **API:** `scff_signal(z, z_pos: [B,C] float32 CUDA contiguous, tau: float) -> (s_perp: [B,C],
  goodness: scalar)`. `z` treated as keys (detached); gradient flows only conceptually (kernel is
  pure compute, no autograd).
- **Parallelization:** one CUDA block per sample `i`; threads cooperate over `C` (load `z_i`,`z⁺_i`
  to shared mem) and over `j∈[0,B)` (each thread accumulates a chunk of the B dots, block-reduce for
  softmax denom via online logsumexp, then accumulate `Σ_j p_ij z_j`). Final projection in-block.
- **Limits:** `B ≤ 256`, `C ≤ 512` fit shared mem (`z_i`,`z⁺_i`,partial sums). Guard: if exceeded,
  fall back to the pure-torch path (asserted, not silently). float32 (Blackwell native).
- **Determinism:** online-logsumexp + ordered block reduction; tolerance vs reference `< 1e-4`.

## Data flow

CIFAR-10 (full 50k) on GPU → per batch, augmented positive view (random crop + hflip) → forward all
blocks (detached between blocks) → per block: pool/normalize → `scff_signal` kernel → `s⊥` →
single-block conv-grad → SGD-ascent update. Eval: freeze, concat pooled block features, linear probe
(sklearn on CPU or a GPU logistic head) → test accuracy. Alignment `A` and κ measured per depth as in
`depth_scaling.py`/`grad_decomp.py`, on GPU.

## Depth-stress experiment (`gpu_depth_stress.py`)

Depths **4, 8, 16, 32**; arms **plain-SCFF, residual-SCFF (α=1/√L), supervised-BP**. Per
(depth, arm) report: **probe accuracy, mean alignment `A`, downstream κ, peak GPU memory**
(`torch.cuda.max_memory_allocated`). Headline: residual holds accuracy/`A` as depth grows while plain
collapses; SCFF peak memory flat vs BP linear — measured on the 5090. Full CIFAR-10, real aug,
enough epochs that BP is a fair upper bound (GPU makes this cheap).

## Benchmarks (`bench_kernel.py`)

1. **Correctness:** `scff_signal` vs pure-torch reference (`gradients.signal` + projection), random
   inputs, assert max-abs-diff `< 1e-4` and goodness match.
2. **Speed:** custom `.cu` vs the torch-autograd goodness-signal, GPU-timed (`cuda.Event`, warmup +
   median of N), across `B,C` sizes. Report speedup.
3. **Memory:** forward-only SCFF vs end-to-end BP peak `max_memory_allocated` vs depth — the 51×
   claim, measured. (Optional: cuda-streams layer-parallel demo — flagged future, not in scope.)

## Error handling / risks

- **Blackwell `sm_120` newness:** cu128 wheels should target it; if the extension build lacks
  `sm_120`, set `TORCH_CUDA_ARCH_LIST=12.0` or use a torch nightly. **Mitigation (Phase 0 gate):**
  compile + run a trivial `.cu` kernel and confirm `cuda.is_available()` + a real GPU matmul *before*
  building the real kernel. If GPU torch can't be made to work, stop and report (do not silently fall
  back to CPU for the "GPU run").
- **Shared-mem overflow** for large `B·C`: explicit guard → pure-torch fallback, asserted.
- **Probe cost** at 50k: batch feature extraction; use a GPU linear head if sklearn is slow.
- **Numerical:** float32 + online logsumexp; test tolerance `1e-4`, not bitwise.

## Testing

`pytest` (`test_scff_kernel.py`): (a) kernel-vs-reference correctness on random `z,z⁺`; (b)
gradient-direction sanity (`s⊥ ⟂ z`); (c) locality (block-`i` signal independent of block-`j`
weights — reuse the conv stop-grad check); (d) a tiny end-to-end GPU train step decreases a held-out
contrastive loss. Tests skip with a clear message when CUDA is absent (CI-safe); the GPU box runs
them for real.

## Phases

- **Phase 0 — Environment gate.** Install cu128 torch; verify `cuda.is_available()`, a GPU matmul,
  and that a trivial `.cu` extension compiles + runs on `sm_120`. Hard gate: do not proceed until GPU
  torch + nvcc extension both work.
- **Phase 1 — Kernel.** Write `scff_signal.cu` + `scff_ext.py` binding; `test_scff_kernel.py`
  correctness/perp/tolerance vs the pure-torch reference.
- **Phase 2 — GPU training integration.** `gpu_arch.py`: `ConvSCFF` on CUDA, forward-only local step
  using the kernel + single-block conv-grad; tiny end-to-end train-step test passes on GPU.
- **Phase 3 — Depth-stress experiment.** `gpu_depth_stress.py`: full CIFAR-10, depths 4/8/16/32,
  plain/residual/BP; report accuracy, `A`, κ, peak memory per depth.
- **Phase 4 — Benchmarks + writeup.** `bench_kernel.py`: kernel correctness/speed + SCFF-vs-BP
  memory vs depth. Fold results into `docs/FINDINGS.md` (GPU/real-hardware section) and commit.

## Success criteria

- GPU torch + custom `.cu` kernel both run on the 5090 (`sm_120`).
- Kernel matches reference `< 1e-4`; measured speedup over the torch path reported (any sign,
  reported honestly).
- Depth-stress produces accuracy/`A`/κ/peak-memory vs depth for all three arms; the residual-bounds-
  the-gap and flat-vs-linear-memory claims either confirmed at scale or honestly reported if not.
- Tests pass on GPU; skip cleanly on CPU.
