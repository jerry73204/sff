# CUDA SCFF on GPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train SCFF fully on the RTX 5090 with a hand-written CUDA kernel for the SCFF-specific hot path (fused B×B InfoNCE softmax + tangent projection = the local gradient, forward-only), then run a deep-conv CIFAR-10 depth-stress measuring accuracy, alignment `A`, κ, and peak GPU memory vs depth, plus kernel correctness/speed and SCFF-vs-BP memory benchmarks.

**Architecture:** Per conv block: cuDNN conv → pool+normalize → custom `scff_signal` kernel emits `s⊥ = P⊥_z(z⁺ − softmax(zzᵀ/τ)·z)` with no autograd graph → backprop `s⊥` through one block only (cuDNN conv-grad) → ascend. Cross-block stop-grad keeps it local + flat-memory. The `.cu` kernel is the SCFF heart; cuDNN does the conv math.

**Tech Stack:** PyTorch cu128 (CUDA 12.8, `sm_120`), `torch.utils.cpp_extension.load` (nvcc), CUDA C++, sklearn (probe), pytest, uv.

**Reference math the kernel must reproduce** (from `empirical/gradients.py`):
```python
# softmax_weights(z, tau) = softmax((z @ z.t())/tau, dim=1)
# signal:        s = z_pos - softmax(z z^T/tau) @ z
# tangent proj:  s_perp = s - z * (z*s).sum(1, keepdim=True)
# local_goodness g = sum_i [ (z_i·z_pos_i)/tau - logsumexp_j (z_i·z_j)/tau ]
```

---

## File Structure

- `empirical/cuda/__init__.py` — re-export `scff_signal`, `signal_ref`, `available()`.
- `empirical/cuda/reference.py` — pure-torch reference `signal_ref(z, z_pos, tau) -> (s_perp, goodness)`.
- `empirical/cuda/scff_signal.cu` — the CUDA kernel + host launcher + pybind.
- `empirical/cuda/scff_ext.py` — JIT-compile via `cpp_extension.load`, expose `scff_signal`.
- `empirical/gpu_arch.py` — device-aware `ConvSCFF` + forward-only local training step.
- `empirical/experiments/gpu_depth_stress.py` — depth-stress experiment.
- `empirical/experiments/bench_kernel.py` — correctness/speed/memory benchmarks.
- `empirical/tests/test_scff_kernel.py` — kernel-vs-reference, perp, locality, e2e (skip if no CUDA).
- `empirical/pyproject.toml` — cu128 torch index (GPU), CPU fallback preserved.

---

## Task 0: Environment gate (cu128 torch + nvcc extension on sm_120)

**Files:**
- Modify: `empirical/pyproject.toml`

- [ ] **Step 1: Point torch at the cu128 index**

Edit `empirical/pyproject.toml` `[tool.uv.sources]` / index to use cu128 instead of cpu:
```toml
[tool.uv.sources]
torch = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

- [ ] **Step 2: Install and verify CUDA is live**

Run:
```bash
cd /home/aeon/repos/sff/empirical && uv sync && \
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
x=torch.randn(1024,1024,device='cuda'); print('matmul ok', (x@x).sum().item()!=0)"
```
Expected: a `+cu128` version, `True`, `NVIDIA GeForce RTX 5090`, `matmul ok True`.
**If `is_available()` is False or matmul fails:** try `uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`; if still failing, STOP and report — do not proceed to GPU work on CPU.

- [ ] **Step 3: Verify a trivial nvcc extension compiles + runs on sm_120**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -c "
from torch.utils.cpp_extension import load_inline
src='''#include <torch/extension.h>
__global__ void addone(float* x,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=1.f;}
void run(at::Tensor x){addone<<<(x.numel()+63)/64,64>>>(x.data_ptr<float>(), x.numel());}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def(\"run\",&run);}'''
import torch
m=load_inline('smoke',cpp_sources='',cuda_sources=src,functions=['run'],verbose=True)
x=torch.zeros(8,device='cuda'); m.run(x); torch.cuda.synchronize(); print('kernel ok', x.tolist())
"
```
Expected: compiles, prints `kernel ok [1.0, 1.0, ...]`. If nvcc errors on arch, keep `TORCH_CUDA_ARCH_LIST=12.0` exported for all later builds.

- [ ] **Step 4: Commit**
```bash
git add empirical/pyproject.toml empirical/uv.lock
git commit -m "build: switch empirical env to cu128 torch (RTX 5090 / sm_120)"
```

---

## Task 1: Pure-torch reference (the correctness oracle)

**Files:**
- Create: `empirical/cuda/__init__.py`, `empirical/cuda/reference.py`
- Test: `empirical/tests/test_scff_kernel.py`

- [ ] **Step 1: Write the failing test**

Create `empirical/tests/test_scff_kernel.py`:
```python
import torch, pytest
from cuda.reference import signal_ref

def test_reference_matches_gradients_module():
    from gradients import signal
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)
    zp = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)
    s_ref, g_ref = signal_ref(z, zp, 0.5)
    s_grad, _ = signal(z, zp, 0.5)                       # gradients.signal: s = z_pos - p@z
    s_grad_perp = s_grad - z * (z * s_grad).sum(1, keepdim=True)
    assert torch.allclose(s_ref, s_grad_perp, atol=1e-5)
    assert s_ref.shape == (16, 32)
    assert torch.allclose((s_ref * z).sum(1), torch.zeros(16), atol=1e-5)   # s_perp ⟂ z
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_scff_kernel.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cuda.reference'`.

- [ ] **Step 3: Write the reference**

Create `empirical/cuda/__init__.py`:
```python
from .reference import signal_ref

def available():
    import torch
    return torch.cuda.is_available()
```
Create `empirical/cuda/reference.py`:
```python
"""Pure-torch reference for the scff_signal CUDA kernel (correctness oracle)."""
import torch

def signal_ref(z, z_pos, tau):
    """s_perp = P_perp_z( z_pos - softmax(z z^T / tau) @ z ); goodness scalar. Matches
    gradients.signal + tangent projection and gradients.local_goodness."""
    scores = (z @ z.t()) / tau
    p = torch.softmax(scores, dim=1)
    s = z_pos - p @ z
    s_perp = s - z * (z * s).sum(1, keepdim=True)
    good = ((z * z_pos).sum(1) / tau - torch.logsumexp(scores, dim=1)).sum()
    return s_perp, good
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_scff_kernel.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add empirical/cuda/__init__.py empirical/cuda/reference.py empirical/tests/test_scff_kernel.py
git commit -m "feat(cuda): pure-torch reference signal_ref + correctness test"
```

---

## Task 2: The CUDA kernel `scff_signal`

**Files:**
- Create: `empirical/cuda/scff_signal.cu`, `empirical/cuda/scff_ext.py`
- Modify: `empirical/cuda/__init__.py`
- Test: `empirical/tests/test_scff_kernel.py`

- [ ] **Step 1: Write the failing test (kernel vs reference on GPU)**

Append to `empirical/tests/test_scff_kernel.py`:
```python
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

@cuda_only
def test_kernel_matches_reference():
    from cuda.scff_ext import scff_signal
    from cuda.reference import signal_ref
    torch.manual_seed(1)
    for B, C in [(16, 32), (64, 128), (128, 256)]:
        z = torch.nn.functional.normalize(torch.randn(B, C, device="cuda"), dim=1)
        zp = torch.nn.functional.normalize(torch.randn(B, C, device="cuda"), dim=1)
        s_ker, g_ker = scff_signal(z, zp, 0.5)
        s_ref, g_ref = signal_ref(z, zp, 0.5)
        assert torch.allclose(s_ker, s_ref, atol=1e-4), (B, C, (s_ker - s_ref).abs().max())
        assert abs(float(g_ker) - float(g_ref)) < 1e-2 * max(1.0, abs(float(g_ref)))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_scff_kernel.py::test_kernel_matches_reference -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cuda.scff_ext'`.

- [ ] **Step 3: Write the kernel**

Create `empirical/cuda/scff_signal.cu` (block-per-sample, `blockDim=128`; shared mem holds
`z_i, z⁺_i, acc/s_i [C]`, `scores [B]`, reduction scratch `[128]`):
```cpp
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void scff_signal_kernel(
    const float* __restrict__ z, const float* __restrict__ zpos,
    float* __restrict__ s_perp, float* __restrict__ goodness,
    const float tau, const int B, const int C) {
  extern __shared__ float sm[];
  float* zi  = sm;            // [C]
  float* zpi = sm + C;        // [C]
  float* acc = sm + 2*C;      // [C]  (reused to hold s_i)
  float* sc  = sm + 3*C;      // [B]  scores
  float* red = sm + 3*C + B;  // [blockDim.x]
  const int i = blockIdx.x, t = threadIdx.x, nt = blockDim.x;

  for (int c = t; c < C; c += nt) { zi[c]=z[i*C+c]; zpi[c]=zpos[i*C+c]; acc[c]=0.f; }
  __syncthreads();

  float lmax = -1e30f;                                   // pass 1: scores + running max
  for (int j = 0; j < B; ++j) {
    float partial = 0.f;
    for (int c = t; c < C; c += nt) partial += zi[c]*z[j*C+c];
    red[t]=partial; __syncthreads();
    for (int s=nt/2; s>0; s>>=1) { if (t<s) red[t]+=red[t+s]; __syncthreads(); }
    if (t==0) sc[j]=red[0]/tau;
    __syncthreads();
    lmax = fmaxf(lmax, sc[j]);
  }
  float sumexp = 0.f;
  for (int j = 0; j < B; ++j) sumexp += __expf(sc[j]-lmax);
  const float lse = lmax + logf(sumexp);

  for (int c = t; c < C; c += nt) {                      // pass 2: acc = sum_j p_ij z_j
    float a = 0.f;
    for (int j = 0; j < B; ++j) a += __expf(sc[j]-lse) * z[j*C+c];
    acc[c] = a;
  }
  __syncthreads();

  float partial = 0.f;                                   // s_i = zpos - acc ; dot_zs = z.s
  for (int c = t; c < C; c += nt) { float si=zpi[c]-acc[c]; acc[c]=si; partial+=zi[c]*si; }
  red[t]=partial; __syncthreads();
  for (int s=nt/2; s>0; s>>=1) { if (t<s) red[t]+=red[t+s]; __syncthreads(); }
  const float dot_zs = red[0];

  for (int c = t; c < C; c += nt) s_perp[i*C+c] = acc[c] - zi[c]*dot_zs;   // tangent proj

  float gp = 0.f;                                        // goodness_i = (z.zpos)/tau - lse
  for (int c = t; c < C; c += nt) gp += zi[c]*zpi[c];
  red[t]=gp; __syncthreads();
  for (int s=nt/2; s>0; s>>=1) { if (t<s) red[t]+=red[t+s]; __syncthreads(); }
  if (t==0) atomicAdd(goodness, red[0]/tau - lse);
}

void scff_signal_launch(at::Tensor z, at::Tensor zpos, at::Tensor s_perp,
                        at::Tensor goodness, double tau, int64_t B, int64_t C) {
  const int threads = 128;
  const size_t shmem = (3*C + B + threads) * sizeof(float);
  scff_signal_kernel<<<B, threads, shmem>>>(
      z.data_ptr<float>(), zpos.data_ptr<float>(), s_perp.data_ptr<float>(),
      goodness.data_ptr<float>(), (float)tau, (int)B, (int)C);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("scff_signal", &scff_signal_launch); }
```

- [ ] **Step 4: Write the Python binding**

Create `empirical/cuda/scff_ext.py`:
```python
"""JIT-compiled CUDA scff_signal kernel."""
import os, torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _ext = load(name="scff_cuda", sources=[os.path.join(_HERE, "scff_signal.cu")], verbose=False)
    return _ext

def scff_signal(z, z_pos, tau):
    """s_perp [B,C] = P_perp_z(z_pos - softmax(z z^T/tau) z); goodness scalar. float32 CUDA only.
    Requires B <= 256 and C <= 512 (shared-mem budget)."""
    assert z.is_cuda and z.dtype == torch.float32, "float32 CUDA tensor required"
    z = z.contiguous(); z_pos = z_pos.contiguous()
    B, C = z.shape
    assert B <= 256 and C <= 512, f"kernel shared-mem budget exceeded (B={B},C={C}); use reference"
    s_perp = torch.empty_like(z)
    goodness = torch.zeros(1, device=z.device, dtype=torch.float32)
    _get_ext().scff_signal(z, z_pos, s_perp, goodness, float(tau), B, C)
    return s_perp, goodness[0]
```
Add to `empirical/cuda/__init__.py`:
```python
def scff_signal(*a, **k):
    from .scff_ext import scff_signal as _f
    return _f(*a, **k)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_scff_kernel.py -v`
Expected: PASS (first run compiles the extension; ~30s). If the reduction mismatches, check `blockDim=128` is power-of-two and `B<=256`.

- [ ] **Step 6: Commit**
```bash
git add empirical/cuda/scff_signal.cu empirical/cuda/scff_ext.py empirical/cuda/__init__.py empirical/tests/test_scff_kernel.py
git commit -m "feat(cuda): hand-written scff_signal kernel (fused InfoNCE signal + tangent proj)"
```

---

## Task 3: GPU forward-only training integration

**Files:**
- Create: `empirical/gpu_arch.py`
- Test: `empirical/tests/test_scff_kernel.py`

- [ ] **Step 1: Write the failing test (one GPU step lowers held-out goodness)**

Append to `empirical/tests/test_scff_kernel.py`:
```python
@cuda_only
def test_gpu_train_step_improves_goodness():
    from gpu_arch import ConvSCFF, scff_local_step, block_goodness
    torch.manual_seed(0)
    m = ConvSCFF(C=32, n_blocks=3, arch="residual", alpha=0.2).cuda()
    x = torch.randn(64, 3, 32, 32, device="cuda")
    xp = x + 0.1 * torch.randn_like(x)
    g0 = block_goodness(m, x, xp, tau=0.5)
    for _ in range(20):
        scff_local_step(m, x, xp, tau=0.5, lr=0.1)
    g1 = block_goodness(m, x, xp, tau=0.5)
    assert g1 > g0                       # ascended local goodness
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_scff_kernel.py::test_gpu_train_step_improves_goodness -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gpu_arch'`.

- [ ] **Step 3: Write `gpu_arch.py`**

Create `empirical/gpu_arch.py` (device-agnostic conv SCFF; the local step uses the kernel for the
signal and one block's cuDNN conv-grad for the weight grad):
```python
"""Device-aware conv SCFF with a forward-only local step driven by the scff_signal kernel."""
import torch, torch.nn as nn, torch.nn.functional as F

def _pooled(y):
    return F.normalize(y.mean(dim=(2, 3)), dim=1)        # [B,C] on the unit sphere

class ConvSCFF(nn.Module):
    def __init__(self, C=64, n_blocks=8, arch="residual", alpha=0.2, in_ch=3):
        super().__init__()
        assert arch in ("plain", "residual")
        self.arch, self.alpha, self.C, self.n_blocks = arch, alpha, C, n_blocks
        self.stem = nn.Conv2d(in_ch, C, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(C, C, 3, padding=1) for _ in range(n_blocks)])

    def _apply(self, y, l):
        h = F.relu(self.blocks[l](y))
        return y + self.alpha * h if self.arch == "residual" else h

    def forward(self, x):
        ys = [F.relu(self.stem(x))]
        for l in range(self.n_blocks):
            ys.append(self._apply(ys[-1], l))
        return ys

    def pooled(self, y):
        return _pooled(y)

def block_goodness(model, x, xp, tau):
    """Sum of per-block local goodness (kernel's scalar), for tests/logging."""
    from cuda.scff_ext import scff_signal
    ys, ysp = model(x), model(xp)
    tot = 0.0
    for l in range(model.n_blocks):
        _, g = scff_signal(_pooled(ys[l + 1]), _pooled(ysp[l + 1]), tau)
        tot += float(g)
    return tot

def scff_local_step(model, x, xp, tau, lr):
    """One forward-only local update for every block. Per block: kernel emits s_perp on the pooled
    rep; backprop s_perp through THIS block only (input detached) -> grad of conv params; ascend."""
    from cuda.scff_ext import scff_signal
    with torch.no_grad():
        ys = [y.detach() for y in model(x)]
        ysp = [y.detach() for y in model(xp)]
    for l in range(model.n_blocks):
        y_in = ys[l].requires_grad_(False)
        out = model._apply(y_in, l)                      # differentiable wrt block l only
        z = _pooled(out)
        s_perp, _ = scff_signal(z.detach(), _pooled(ysp[l + 1]).detach(), tau)
        # ascend goodness: dz = s_perp/tau; push z along s_perp via vector-Jacobian product
        grads = torch.autograd.grad(z, model.blocks[l].parameters(), grad_outputs=s_perp)
        with torch.no_grad():
            for p, gp in zip(model.blocks[l].parameters(), grads):
                p.add_(lr * gp)                          # + : ascend (s_perp is +dg/dz direction)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_scff_kernel.py::test_gpu_train_step_improves_goodness -v`
Expected: PASS (g1 > g0).

- [ ] **Step 5: Commit**
```bash
git add empirical/gpu_arch.py empirical/tests/test_scff_kernel.py
git commit -m "feat(gpu): forward-only conv SCFF local step driven by scff_signal kernel"
```

---

## Task 4: Depth-stress experiment on CIFAR-10

**Files:**
- Create: `empirical/experiments/gpu_depth_stress.py`

- [ ] **Step 1: Write the experiment**

Create `empirical/experiments/gpu_depth_stress.py` (reuse the CIFAR loader/aug from
`experiments/cifar_conv.py`; everything on `cuda`; depths 4/8/16/32; arms plain/residual/BP; metrics
accuracy, `A`, κ, peak memory):
```python
"""GPU depth-stress (spec docs/superpowers/specs/2026-06-03-cuda-scff-gpu-design.md).
Does residual hold accuracy/alignment as depth grows while plain collapses? Memory flat vs BP?
Run: python experiments/gpu_depth_stress.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_arch import ConvSCFF, _pooled, scff_local_step
from cuda.scff_ext import scff_signal
from experiments.cifar_conv import load_cifar, augment   # reuse loader + aug

DEV = "cuda"
DEPTHS = [4, 8, 16, 32]
CFG = dict(C=64, tau=0.5, batch=128, epochs=15, lr_scff=0.05, lr_bp=1e-3,
           aug_noise=0.06, n_train=50000, n_test=10000, seed=0)

def features(model, X, bs=1000):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ys = model(X[i:i+bs].to(DEV))
            outs.append(torch.cat([_pooled(ys[l]) for l in range(1, model.n_blocks+1)], 1).cpu())
    return torch.cat(outs).numpy()

def probe(model, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, C=1.0).fit(features(model, Xtr), ytr.numpy())
    return float((clf.predict(features(model, Xte)) == yte.numpy()).mean())

def mean_A(model, Xte, cfg):
    x = Xte[:cfg["batch"]].to(DEV); xp = augment(x.cpu(), cfg["aug_noise"]).to(DEV)
    ys, ysp = model(x), model(xp)
    vals = []
    for l in range(model.n_blocks - 1):
        z, zp = _pooled(ys[l+1]).detach(), _pooled(ysp[l+1]).detach()
        s_loc, _ = scff_signal(z, zp, cfg["tau"])
        zL, zpL = _pooled(ys[-1]).detach(), _pooled(ysp[-1]).detach()
        sL, _ = scff_signal(zL, zpL, cfg["tau"])
        # transported global signal via one autograd VJP through blocks l+1..L
        with torch.enable_grad():
            yin = ys[l+1].detach().requires_grad_(True)
            ytmp = yin
            for k in range(l+1, model.n_blocks):
                ytmp = model._apply(ytmp, k)
            g = torch.autograd.grad(_pooled(ytmp), yin, grad_outputs=sL)[0]
        s_bp = _pooled_grad_proj(g, ys[l+1])
        vals.append(float((s_loc.flatten() @ s_bp.flatten()) /
                          (s_loc.norm() * s_bp.norm() + 1e-9)))
    return sum(vals) / len(vals)

def _pooled_grad_proj(g, y):
    # reduce the [B,C,H,W] transported grad to a [B,C] tangent vector comparable to s_loc
    gp = g.mean(dim=(2,3))
    z = _pooled(y)
    return gp - z * (z*gp).sum(1, keepdim=True)

def train_scff(model, Xtr, cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr)-cfg["batch"]+1, cfg["batch"]):
            b = idx[i:i+cfg["batch"]]; xb = Xtr[b].to(DEV)
            xp = augment(Xtr[b], cfg["aug_noise"]).to(DEV)
            scff_local_step(model, xb, xp, cfg["tau"], cfg["lr_scff"])

def train_bp(model, Xtr, ytr, cfg, n_classes=10):
    head = torch.nn.Linear(model.C, n_classes).to(DEV)
    opt = torch.optim.Adam(list(model.parameters())+list(head.parameters()), lr=cfg["lr_bp"])
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr)-cfg["batch"]+1, cfg["batch"]):
            b = idx[i:i+cfg["batch"]]
            logits = head(model.pooled(model(Xtr[b].to(DEV))[-1]))
            loss = lossf(logits, ytr[b].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()

def run_arm(name, make, train, Xtr, ytr, Xte, yte, want_A):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    m = make().to(DEV); train(m)
    mem = torch.cuda.max_memory_allocated()/1e6
    acc = probe(m, Xtr, ytr, Xte, yte)
    A = mean_A(m, Xte, CFG) if want_A else float("nan")
    print(f"  {name:16s} acc={acc:.4f}  A={A:.3f}  peakMB={mem:.0f}", flush=True)
    return acc, A, mem

def main():
    assert torch.cuda.is_available(), "GPU required"
    Xtr, ytr, Xte, yte = load_cifar(CFG)
    print(f"CIFAR-10 train={len(Xtr)} test={len(Xte)} on {torch.cuda.get_device_name(0)}\n")
    for L in DEPTHS:
        print(f"L={L}:")
        run_arm("supervised-BP", lambda: ConvSCFF(CFG["C"], L, "plain"),
                lambda m: train_bp(m, Xtr, ytr, CFG), Xtr, ytr, Xte, yte, False)
        run_arm("plain-SCFF", lambda: ConvSCFF(CFG["C"], L, "plain"),
                lambda m: train_scff(m, Xtr, CFG), Xtr, ytr, Xte, yte, True)
        run_arm("residual-SCFF", lambda: ConvSCFF(CFG["C"], L, "residual", 1.0/math.sqrt(L)),
                lambda m: train_scff(m, Xtr, CFG), Xtr, ytr, Xte, yte, True)

if __name__ == "__main__":
    main()
```
The `mean_A` above computes `A = cos(s_loc, P⊥(transported global signal))`: the local kernel
signal `s_loc` vs the output signal `sL` pulled back through blocks `ℓ+1..L` by one autograd VJP,
reduced to a `[B,C]` tangent vector by `_pooled_grad_proj`. This mirrors `grad_decomp.py` on GPU.

- [ ] **Step 2: Smoke-run tiny (catch bugs fast)**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -c "
import experiments.gpu_depth_stress as E
E.DEPTHS=[4]; E.CFG.update(n_train=2000,n_test=1000,epochs=2)
E.main()"
```
Expected: prints `L=4:` then three arms with `acc=…  A=…  peakMB=…`, no exception.

- [ ] **Step 3: Full run (background, monitor)**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 nohup .venv/bin/python experiments/gpu_depth_stress.py > /tmp/gpu_depth.out 2>&1 &
```
Expected (eventual): 4 depth blocks × 3 arms; residual `A` stays high across depth, plain `A` decays; residual peakMB ≈ flat across depth, BP peakMB grows with depth.

- [ ] **Step 4: Commit**
```bash
git add empirical/experiments/gpu_depth_stress.py
git commit -m "feat(gpu): CIFAR-10 deep-conv depth-stress (acc/A/kappa/peak-mem vs depth)"
```

---

## Task 5: Kernel + memory benchmarks

**Files:**
- Create: `empirical/experiments/bench_kernel.py`

- [ ] **Step 1: Write the benchmark**

Create `empirical/experiments/bench_kernel.py`:
```python
"""Benchmark: scff_signal kernel correctness + speed vs torch; SCFF-vs-BP peak memory vs depth.
Run: python experiments/bench_kernel.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda.scff_ext import scff_signal
from cuda.reference import signal_ref
from gpu_arch import ConvSCFF, _pooled, scff_local_step

def _time(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters                       # ms/iter

def bench_kernel():
    print("kernel correctness + speed (vs torch reference):")
    for B, C in [(64, 128), (128, 256), (256, 512)]:
        z = torch.nn.functional.normalize(torch.randn(B, C, device="cuda"), dim=1)
        zp = torch.nn.functional.normalize(torch.randn(B, C, device="cuda"), dim=1)
        s_k, _ = scff_signal(z, zp, 0.5); s_r, _ = signal_ref(z, zp, 0.5)
        err = float((s_k - s_r).abs().max())
        t_k = _time(lambda: scff_signal(z, zp, 0.5))
        t_r = _time(lambda: signal_ref(z, zp, 0.5))
        print(f"  B={B:4d} C={C:4d}  maxerr={err:.2e}  kernel={t_k:.4f}ms  torch={t_r:.4f}ms  "
              f"speedup={t_r/t_k:.2f}x", flush=True)

def bench_memory():
    print("\npeak memory vs depth (SCFF forward-only vs BP):")
    x = torch.randn(128, 3, 32, 32, device="cuda")
    xp = x + 0.1 * torch.randn_like(x)
    for L in [8, 16, 32, 64]:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        m = ConvSCFF(64, L, "residual", 1.0/math.sqrt(L)).cuda()
        scff_local_step(m, x, xp, 0.5, 0.05)
        scff_mb = torch.cuda.max_memory_allocated()/1e6
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
        m = ConvSCFF(64, L, "residual", 1.0/math.sqrt(L)).cuda()
        head = torch.nn.Linear(64, 10).cuda()
        loss = torch.nn.functional.cross_entropy(head(m.pooled(m(x)[-1])),
                                                 torch.zeros(128, dtype=torch.long, device="cuda"))
        loss.backward()
        bp_mb = torch.cuda.max_memory_allocated()/1e6
        print(f"  L={L:3d}  SCFF={scff_mb:7.1f}MB  BP={bp_mb:7.1f}MB  ratio={bp_mb/scff_mb:.1f}x",
              flush=True)

def main():
    assert torch.cuda.is_available(), "GPU required"
    bench_kernel(); bench_memory()

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the benchmark**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python experiments/bench_kernel.py`
Expected: `maxerr` ≈ 1e-5–1e-4 each size; a kernel-vs-torch `speedup` number (any value, reported honestly); memory `ratio` growing with `L` (BP grows, SCFF ~flat).

- [ ] **Step 3: Commit**
```bash
git add empirical/experiments/bench_kernel.py
git commit -m "feat(gpu): scff_signal kernel speed + SCFF-vs-BP memory benchmarks"
```

---

## Task 6: Fold results into FINDINGS

**Files:**
- Modify: `docs/FINDINGS.md`

- [ ] **Step 1: Add a "GPU / real-hardware (RTX 5090)" section**

After the conv/CIFAR section, add a subsection reporting: the depth-stress table (acc/`A`/κ/peakMB
vs depth for plain/residual/BP from `/tmp/gpu_depth.out`); the kernel speedup vs torch and max-error
from `bench_kernel.py`; the measured SCFF-vs-BP memory ratio vs depth. State honestly whether the
residual-bounds-the-gap and flat-vs-linear-memory claims held at scale. Use the real numbers from the
runs (do not invent — paste from the output files).

- [ ] **Step 2: Run full test suite**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest -q`
Expected: all pass (45 prior + new kernel/GPU tests; GPU tests run on the 5090).

- [ ] **Step 3: Commit + push**
```bash
git add docs/FINDINGS.md
git commit -m "docs: GPU/RTX-5090 results — depth-stress, kernel speedup, measured memory"
git push origin main
```

---

## Notes on correctness/perf to watch during execution

- **`scff_local_step` sign:** `s_perp = +∂g/∂z` (goodness, an ascent objective). `autograd.grad(z, W, grad_outputs=s_perp)` gives `∂(s_perp·z)/∂W`; ascend with `p += lr*grad`. The `test_gpu_train_step_improves_goodness` test guards the sign — if goodness *decreases*, flip to `-=`.
- **`mean_A` cost:** the transported-global VJP runs `L−ℓ` blocks per layer; fine at `L≤32` on GPU. If slow, subsample layers.
- **Probe at 50k:** sklearn LogisticRegression on ~`C·L`-dim concat features × 50k may be slow; if so cap probe-train to 10k and note it.
- **Determinism:** kernel uses `__expf`/`logf`; tolerance is `1e-4`, not bitwise.
