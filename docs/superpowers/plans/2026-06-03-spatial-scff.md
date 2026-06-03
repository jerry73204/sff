# Spatial (per-location) SCFF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the SCFF conv local objective from the global-average-pooled rep to per spatial location, restoring spatial gradient to the conv filters and the residual-skip isometry, and measure the CIFAR-10 win vs the global-pool baseline.

**Architecture:** Reuse the existing `gpu_arch.ConvSCFF` (residual conv blocks), the `scff_signal` CUDA kernel, and the forward-only per-block local step — but feed the kernel per-location tokens (each `(h,w)` C-vector) per image instead of one pooled vector per image. Same-location positives (appearance-only aug), in-image negatives.

**Tech Stack:** PyTorch cu128 (RTX 5090, sm_120), the `scff_signal` CUDA extension, sklearn probe, pytest. Set `TORCH_CUDA_ARCH_LIST=12.0` for any python that loads the kernel.

**Key facts for the implementer:**
- `from gpu_arch import ConvSCFF, _pooled, scff_local_step, block_goodness` already exist. `ConvSCFF(C, n_blocks, arch, alpha, in_ch)`; `.forward(x)` → list `[stem_out, block0_out, …, block(L-1)_out]`; per-block method is `apply_block(y, l)`; `.pooled(y)` = global-avg-pool + L2-norm `[B,C]`.
- `from cuda.scff_ext import scff_signal` → `scff_signal(z, z_pos, tau) -> (s_perp, goodness)`, `[N,C]` float32 CUDA contiguous, **N≤256, C≤512**.
- `from experiments.cifar_conv import load_cifar` (returns `(Xtr,ytr,Xte,yte)` CPU float32; cfg keys `seed,n_train,n_test`). CIFAR-10 cached at `/tmp/cifar`.
- Feature map after the stride-2 stem is **16×16 → H·W = 256** (exactly the kernel's N limit) with C=64.
- Existing tests in `empirical/tests/test_scff_kernel.py` define `cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Run tests: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_scff_kernel.py -v`.

---

## File Structure

- `empirical/gpu_arch.py` (modify) — add `per_location_tokens`, `augment_appearance`, `block_goodness_spatial`, `scff_local_step_spatial`. Keep the existing global-pool functions for the A/B comparison.
- `empirical/experiments/cifar_spatial.py` (create) — compare per-location-SCFF vs global-pool-SCFF vs supervised-BP on CIFAR-10.
- `empirical/tests/test_scff_kernel.py` (modify) — append spatial token/ascent/locality tests.

---

## Task 1: Per-location tokens + appearance-only augmentation

**Files:** Modify `empirical/gpu_arch.py`; Test append `empirical/tests/test_scff_kernel.py`.

- [ ] **Step 1: Write the failing test**

Append to `empirical/tests/test_scff_kernel.py`:
```python
def test_per_location_tokens_shape_and_norm():
    from gpu_arch import per_location_tokens
    y = torch.randn(4, 8, 5, 5)                  # [B, C, H, W]
    tok = per_location_tokens(y)
    assert tok.shape == (4, 25, 8)               # [B, H*W, C]
    norms = tok.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4, 25), atol=1e-5)   # each location unit-norm
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_scff_kernel.py::test_per_location_tokens_shape_and_norm -v`
Expected: FAIL — `ImportError: cannot import name 'per_location_tokens'`.

- [ ] **Step 3: Implement**

Add to `empirical/gpu_arch.py` (after `_pooled`):
```python
def per_location_tokens(y):
    """[B,C,H,W] -> [B, H*W, C] with each location's C-vector L2-normalized (a token on the sphere)."""
    B, C, H, W = y.shape
    t = y.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return F.normalize(t, dim=-1)

def augment_appearance(x, noise):
    """Appearance-only positive view: additive Gaussian noise, NO spatial transform (so location
    i<->i corresponds trivially across views)."""
    return x + noise * torch.randn_like(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_scff_kernel.py::test_per_location_tokens_shape_and_norm -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add empirical/gpu_arch.py empirical/tests/test_scff_kernel.py
git commit -m "feat(gpu): per-location token + appearance-only augmentation helpers"
```

---

## Task 2: Forward-only per-location local step

**Files:** Modify `empirical/gpu_arch.py`; Test append `empirical/tests/test_scff_kernel.py`.

- [ ] **Step 1: Write the failing tests**

Append to `empirical/tests/test_scff_kernel.py`:
```python
@cuda_only
def test_spatial_step_improves_goodness():
    from gpu_arch import ConvSCFF, scff_local_step_spatial, block_goodness_spatial, augment_appearance
    torch.manual_seed(0)
    m = ConvSCFF(C=32, n_blocks=3, arch="residual", alpha=0.2).cuda()
    x = torch.randn(16, 3, 32, 32, device="cuda")
    xp = augment_appearance(x, 0.06)
    g0 = block_goodness_spatial(m, x, xp, tau=0.5)
    for _ in range(20):
        scff_local_step_spatial(m, x, xp, tau=0.5, lr=0.1)
    g1 = block_goodness_spatial(m, x, xp, tau=0.5)
    assert g1 > g0

@cuda_only
def test_spatial_step_is_layer_local():
    # block-l spatial signal must not depend on a different block's params
    from gpu_arch import ConvSCFF, per_location_tokens
    torch.manual_seed(0)
    m = ConvSCFF(C=32, n_blocks=3, arch="residual", alpha=0.2).cuda()
    x = torch.randn(8, 3, 32, 32, device="cuda")
    ys = [y.detach() for y in m(x)]
    out = m.apply_block(ys[0], 0)                 # block 0 output, differentiable wrt block 0 only
    z = per_location_tokens(out)
    g = z.pow(2).sum()
    grads = torch.autograd.grad(g, list(m.blocks[1].parameters()), allow_unused=True)
    assert all(gp is None for gp in grads)        # no gradient flows into block 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_scff_kernel.py::test_spatial_step_improves_goodness -v`
Expected: FAIL — `ImportError: cannot import name 'scff_local_step_spatial'`.

- [ ] **Step 3: Implement**

Add to `empirical/gpu_arch.py`:
```python
def block_goodness_spatial(model, x, xp, tau):
    """Sum of per-image, per-location InfoNCE goodness over all blocks (for tests/logging)."""
    from cuda.scff_ext import scff_signal
    ys, ysp = model(x), model(xp)
    tot = 0.0
    for l in range(model.n_blocks):
        z = per_location_tokens(ys[l + 1]); zp = per_location_tokens(ysp[l + 1])
        for b in range(z.shape[0]):
            _, g = scff_signal(z[b].contiguous(), zp[b].contiguous(), tau)
            tot += float(g)
    return tot

def scff_local_step_spatial(model, x, xp, tau, lr):
    """One forward-only local update per block, with PER-LOCATION InfoNCE (tokens = spatial
    locations, in-image negatives). The scff_signal kernel runs per image over its H*W locations;
    the resulting per-location tangent signal is backpropped through THIS block only."""
    from cuda.scff_ext import scff_signal
    with torch.no_grad():
        ys = [y.detach() for y in model(x)]
        ysp = [y.detach() for y in model(xp)]
    for l in range(model.n_blocks):
        out = model.apply_block(ys[l], l)                 # differentiable wrt block l only
        z = per_location_tokens(out)                      # [B, HW, C], requires grad
        zp = per_location_tokens(ysp[l + 1]).detach()     # [B, HW, C]
        s_perp = torch.empty_like(z)
        for b in range(z.shape[0]):
            sb, _ = scff_signal(z[b].detach().contiguous(), zp[b].contiguous(), tau)  # [HW, C]
            s_perp[b] = sb
        grads = torch.autograd.grad(z, model.blocks[l].parameters(), grad_outputs=s_perp)
        with torch.no_grad():
            for p, gp in zip(model.blocks[l].parameters(), grads):
                p.add_(lr * gp)                           # ascend (s_perp is +dg/dz)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_scff_kernel.py::test_spatial_step_improves_goodness tests/test_scff_kernel.py::test_spatial_step_is_layer_local -v`
Expected: PASS (g1 > g0; block-1 grads all None). If goodness DECREASES, the sign is wrong — flip to `p.add_(-lr * gp)` and re-run until `g1 > g0`; report which sign.

- [ ] **Step 5: Commit**
```bash
git add empirical/gpu_arch.py empirical/tests/test_scff_kernel.py
git commit -m "feat(gpu): forward-only per-location SCFF local step (in-image InfoNCE)"
```

---

## Task 3: CIFAR-10 comparison experiment + FINDINGS

**Files:** Create `empirical/experiments/cifar_spatial.py`; Modify `docs/FINDINGS.md`.

- [ ] **Step 1: Write the experiment**

Create `empirical/experiments/cifar_spatial.py`:
```python
"""Spatial SCFF (spec docs/superpowers/specs/2026-06-03-spatial-scff-design.md): does moving the
local objective from global-pool to per-location fix the conv bottleneck?
Compares supervised-BP vs global-pool-SCFF vs per-location-SCFF on CIFAR-10 (residual conv).
Run: python experiments/cifar_spatial.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_arch import (ConvSCFF, _pooled, scff_local_step, scff_local_step_spatial,
                      augment_appearance)
from experiments.cifar_conv import load_cifar

DEV = "cuda"
CFG = dict(C=64, n_blocks=8, alpha=1.0/math.sqrt(8), tau=0.5, batch=128, epochs=12,
           lr_scff=0.05, lr_bp=1e-3, aug_noise=0.06, n_train=20000, n_test=5000, seed=0)

def features(model, X, bs=1000):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ys = model(X[i:i+bs].to(DEV))
            outs.append(torch.cat([_pooled(ys[l]) for l in range(1, model.n_blocks+1)], 1).cpu())
    return torch.cat(outs).numpy()

def probe(model, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300, C=1.0).fit(features(model, Xtr), ytr.numpy())
    return float((clf.predict(features(model, Xte)) == yte.numpy()).mean())

def batches(n, bs, gen):
    idx = torch.randperm(n, generator=gen)
    for i in range(0, n-bs+1, bs):
        yield idx[i:i+bs]

def train_scff(model, Xtr, cfg, step_fn):
    gen = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        for b in batches(len(Xtr), cfg["batch"], gen):
            xb = Xtr[b].to(DEV); xp = augment_appearance(xb, cfg["aug_noise"])
            step_fn(model, xb, xp, cfg["tau"], cfg["lr_scff"])

def train_bp(model, Xtr, ytr, cfg, n_classes=10):
    head = torch.nn.Linear(model.C, n_classes).to(DEV)
    opt = torch.optim.Adam(list(model.parameters())+list(head.parameters()), lr=cfg["lr_bp"])
    lossf = torch.nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        for b in batches(len(Xtr), cfg["batch"], gen):
            logits = head(model.pooled(model(Xtr[b].to(DEV))[-1]))
            loss = lossf(logits, ytr[b].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()

def run(name, make, train, Xtr, ytr, Xte, yte):
    m = make().to(DEV); train(m)
    acc = probe(m, Xtr, ytr, Xte, yte)
    print(f"  {name:20s} acc={acc:.4f}", flush=True)
    return acc

def main():
    assert torch.cuda.is_available(), "GPU required"
    Xtr, ytr, Xte, yte = load_cifar(CFG)
    L = CFG["n_blocks"]
    print(f"CIFAR-10 train={len(Xtr)} test={len(Xte)} residual conv C={CFG['C']} L={L} "
          f"on {torch.cuda.get_device_name(0)}\n")
    def mk(): return ConvSCFF(CFG["C"], L, "residual", CFG["alpha"])
    a_bp  = run("supervised-BP",      mk, lambda m: train_bp(m, Xtr, ytr, CFG), Xtr, ytr, Xte, yte)
    a_pool= run("global-pool-SCFF",   mk, lambda m: train_scff(m, Xtr, CFG, scff_local_step),
                Xtr, ytr, Xte, yte)
    a_loc = run("per-location-SCFF",  mk, lambda m: train_scff(m, Xtr, CFG, scff_local_step_spatial),
                Xtr, ytr, Xte, yte)
    print("\n=== VERDICT (CIFAR-10 linear probe, residual conv) ===")
    print(f"  supervised-BP      {a_bp:.4f}")
    print(f"  per-location-SCFF  {a_loc:.4f}")
    print(f"  global-pool-SCFF   {a_pool:.4f}")
    print(f"\nspatial fix (per-location - global-pool): {a_loc-a_pool:+.4f}")
    print(f"remaining gap to BP: {a_bp-a_loc:+.4f}")
    print("=> " + ("per-location FIXES the conv bottleneck (>> global-pool, toward BP)"
                   if a_loc - a_pool > 0.05 else
                   "per-location did not clearly beat global-pool -- investigate"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run tiny (catch bugs fast)**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -c "
import experiments.cifar_spatial as E
E.CFG.update(n_blocks=3, n_train=1000, n_test=500, epochs=2)
E.CFG['alpha']=1.0/(3**0.5)
E.main()"
```
Expected: prints three arms each with `acc=…`, no exception. Per-location may already edge out global-pool even tiny.

- [ ] **Step 3: Commit**
```bash
git add empirical/experiments/cifar_spatial.py
git commit -m "feat(gpu): CIFAR-10 per-location vs global-pool vs BP comparison"
```

- [ ] **Step 4: Full run (background, controller monitors)**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 nohup .venv/bin/python experiments/cifar_spatial.py > /tmp/cifar_spatial.out 2>&1 &
echo "launched pid $!"
```
Report the PID. (Per-location training does B kernel calls per block per batch — slower than global-pool; expect a longer run.)

- [ ] **Step 5: Fold results into FINDINGS**

After the full run finishes, add a subsection to `docs/FINDINGS.md` (after the CIFAR conv section) reporting the three accuracies from `/tmp/cifar_spatial.out`, whether per-location ≫ global-pool (the spatial fix), and the remaining gap to BP. Use the real numbers (paste from the output). State honestly whether it reached the paper's ~0.80 territory or only partially closed the gap.

- [ ] **Step 6: Commit**
```bash
git add docs/FINDINGS.md
git commit -m "docs: per-location SCFF CIFAR-10 result (spatial objective fix)"
```

---

## Notes for execution

- **Sign:** `scff_local_step_spatial` uses `+lr*grad` (ascent). `test_spatial_step_improves_goodness` guards it — if `g1 < g0`, flip to `-`.
- **Kernel token limit:** per image `z[b]` is `[H·W, C] = [256, 64]` — exactly at the `N≤256` limit. If a config changes the spatial size above 16×16, downsample before `per_location_tokens` or the kernel assert will fire.
- **Speed:** the per-image kernel loop (B calls/block/batch) makes per-location training notably slower than global-pool; the full run is long. That is expected; the controller monitors the background run.
- **A/κ:** this plan reports accuracy (the success criterion). Measuring spatial alignment `A` is a follow-up; not in scope here.
