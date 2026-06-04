# Conv gen-FF + GPU pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU training pipeline + conv gen-FF (per-location denoising-energy early objective + joint-energy head) and run the headline CIFAR-10 test: does gen-FF-conv beat SCFF's ~0.35 conv wall?

**Architecture:** Reuse `gpu_arch.ConvSCFF` (residual conv, GPU) as the backbone; add a forward-only per-location denoising-energy objective and a GPU-clean EBM head trainer; a GPU CIFAR data module with on-GPU augmentation; a 4-arm experiment (BP / SCFF+probe / SCFF+head / gen-FF).

**Tech Stack:** PyTorch cu128 (RTX 5090), the `scff_signal` CUDA kernel (for the SCFF in-batch arm only — set `TORCH_CUDA_ARCH_LIST=12.0`), sklearn (probe), pytest.

**Reuse (all exist):** `gpu_arch.ConvSCFF` (`.forward(x)`→`[stem,block0,…]`, `apply_block(y,l)`, `pooled(y)`, attrs `C`,`n_blocks`,`alpha`; module fn `_pooled`), `gpu_arch.scff_local_step(model,x,xp,tau,lr)` (in-batch SCFF via kernel), `genff.EnergyHead(feat_dim,n_classes)`, `genff.predict(model,head,X)`, `genff.free_energy(model,head,X)` (device-agnostic; call `model.features`), `experiments.cifar_conv.load_cifar(cfg)` (CPU tensors; cfg keys `seed,n_train,n_test`). CIFAR cached at `/tmp/cifar`. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## File Structure

- `empirical/gpu_arch.py` (modify) — add `ConvSCFF.features(x)`.
- `empirical/genff_conv.py` (create) — `conv_denoise_step`, `block_energy_gap`, `train_head_conv`.
- `empirical/gpu_pipeline.py` (create) — `to_gpu`, `augment_batch`, `ece`, `ood_auroc`.
- `empirical/experiments/genff_cifar.py` (create) — 4-arm experiment.
- `empirical/tests/test_genff_conv.py` (create) — denoise gap, locality, features shape, augment shape.

---

## Task 1: Conv gen-FF objective + features + head trainer

**Files:** Modify `empirical/gpu_arch.py`; Create `empirical/genff_conv.py`; Test `empirical/tests/test_genff_conv.py`.

- [ ] **Step 1: Write the failing tests**

Create `empirical/tests/test_genff_conv.py`:
```python
import torch, pytest
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

@cuda_only
def test_conv_features_shape():
    from gpu_arch import ConvSCFF
    m = ConvSCFF(C=32, n_blocks=4, arch="residual", alpha=0.3).cuda()
    f = m.features(torch.randn(8, 3, 32, 32, device="cuda"))
    assert f.shape == (8, 32 * 4)

@cuda_only
def test_conv_denoise_raises_energy_gap():
    from gpu_arch import ConvSCFF
    from genff_conv import conv_denoise_step, block_energy_gap
    torch.manual_seed(0)
    m = ConvSCFF(C=32, n_blocks=3, arch="residual", alpha=0.3).cuda()
    X = torch.randn(64, 3, 32, 32, device="cuda")
    def gap():
        return block_energy_gap(m, X, X + 0.3 * torch.randn_like(X))
    g0 = gap()
    for _ in range(20):
        conv_denoise_step(m, X, dict(sigma=0.3, lr=0.02))
    assert gap() > g0

@cuda_only
def test_conv_denoise_is_layer_local():
    from gpu_arch import ConvSCFF
    torch.manual_seed(0)
    m = ConvSCFF(C=32, n_blocks=3, arch="residual", alpha=0.3).cuda()
    ys = [y.detach() for y in m(torch.randn(8, 3, 32, 32, device="cuda"))]
    g = m.apply_block(ys[0], 0).pow(2).mean()
    grads = torch.autograd.grad(g, list(m.blocks[1].parameters()), allow_unused=True)
    assert all(gp is None for gp in grads)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_genff_conv.py -v`
Expected: FAIL — `AttributeError: 'ConvSCFF' object has no attribute 'features'` / `ModuleNotFoundError: genff_conv`.

- [ ] **Step 3: Add `ConvSCFF.features`**

In `empirical/gpu_arch.py`, add this method to `class ConvSCFF` (after `pooled`):
```python
    def features(self, x):
        """Concat of per-block global-avg-pooled, L2-normalized reps: [B, C*n_blocks]."""
        ys = self.forward(x)
        return torch.cat([_pooled(ys[l]) for l in range(1, self.n_blocks + 1)], dim=1)
```

- [ ] **Step 4: Create `genff_conv.py`**

Create `empirical/genff_conv.py`:
```python
"""Conv gen-FF: per-location denoising-energy early objective + GPU-clean EBM head trainer.
Spec: docs/superpowers/specs/2026-06-04-genff-conv-gpu-design.md."""
import torch, torch.nn as nn, torch.nn.functional as F


def conv_denoise_step(model, x, cfg):
    """One forward-only local update per block: per-location energy G_{h,w}=mean_C h^2 trained HIGH
    on real x, LOW on noised x via the paired contrast -logsigmoid(G_real - G_noised). Layer-local
    (block input detached). Both G_real and G_noised carry grad wrt block l (we raise one, lower the
    other through the same weights)."""
    sig, lr = cfg["sigma"], cfg["lr"]
    xn = x + sig * torch.randn_like(x)
    with torch.no_grad():
        ys = [y.detach() for y in model(x)]
        ysn = [y.detach() for y in model(xn)]
    for l in range(model.n_blocks):
        Gr = model.apply_block(ys[l], l).pow(2).mean(1)     # [B,H,W] per-location energy
        Gn = model.apply_block(ysn[l], l).pow(2).mean(1)
        loss = -F.logsigmoid(Gr - Gn).mean()
        grads = torch.autograd.grad(loss, model.blocks[l].parameters())
        with torch.no_grad():
            for p, g in zip(model.blocks[l].parameters(), grads):
                p.add_(-lr * g)                              # descend the paired-contrast loss


def block_energy_gap(model, x, xn):
    """Sum over blocks of (mean real energy - mean noised energy). Rises as denoising trains."""
    ys, ysn = model(x), model(xn)
    return sum((ys[l].pow(2).mean() - ysn[l].pow(2).mean()).item()
               for l in range(1, model.n_blocks + 1))


def train_head_conv(model, head, X, y, cfg):
    """GPU-clean EBM head trainer (device-correct noise via randn_like). Loss = CE + lam * softplus(
    lse_noised - lse_real), pushing real-feature free-energy up / noised down. Backbone frozen."""
    opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
    ce = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(cfg["seed"])
    sig, lam = cfg["sigma"], cfg["lam"]
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(X), generator=g)
        for i in range(0, len(X) - cfg["batch"] + 1, cfg["batch"]):
            b = idx[i:i + cfg["batch"]]
            xb, yb = X[b], y[b]
            with torch.no_grad():
                fr = model.features(xb)
                fn = model.features(xb + sig * torch.randn_like(xb))
            logits = head(fr)
            lse_r = torch.logsumexp(logits, dim=1)
            lse_n = torch.logsumexp(head(fn), dim=1)
            loss = ce(logits, yb) + lam * F.softplus(lse_n - lse_r).mean()
            opt.zero_grad(); loss.backward(); opt.step()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_genff_conv.py -v`
Expected: PASS (3 tests). If `test_conv_denoise_raises_energy_gap` fails, raise the loop to 40 or lr to 0.05 (the gap must grow; sign is `add_(-lr*g)`).

- [ ] **Step 6: Commit**
```bash
git add empirical/gpu_arch.py empirical/genff_conv.py empirical/tests/test_genff_conv.py
git commit -m "feat(genff-conv): per-location denoising-energy step + ConvSCFF.features + EBM head trainer"
```

---

## Task 2: GPU CIFAR data module + metrics

**Files:** Create `empirical/gpu_pipeline.py`; Test `empirical/tests/test_genff_conv.py`.

- [ ] **Step 1: Write the failing tests**

Append to `empirical/tests/test_genff_conv.py`:
```python
def test_augment_batch_shape_and_finite():
    from gpu_pipeline import augment_batch
    x = torch.randn(8, 3, 32, 32)
    a = augment_batch(x)
    assert a.shape == (8, 3, 32, 32) and torch.isfinite(a).all()

def test_ece_and_auroc():
    from gpu_pipeline import ece, ood_auroc
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    y = torch.tensor([0, 1, 1])
    assert 0.0 <= ece(probs, y) <= 1.0
    au = ood_auroc(torch.tensor([0.0, 0.1, 0.2]), torch.tensor([1.0, 1.1, 1.2]))
    assert au == 1.0   # ood scores strictly higher -> perfect separation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_genff_conv.py::test_augment_batch_shape_and_finite -v`
Expected: FAIL — `ModuleNotFoundError: gpu_pipeline`.

- [ ] **Step 3: Create `gpu_pipeline.py`**

Create `empirical/gpu_pipeline.py`:
```python
"""GPU CIFAR data module + metrics for the conv gen-FF pipeline."""
import os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.cifar_conv import load_cifar


def to_gpu(cfg, dev="cuda"):
    """Load CIFAR-10 and move all splits to `dev`. Returns (Xtr,ytr,Xte,yte)."""
    Xtr, ytr, Xte, yte = load_cifar(cfg)
    return Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)


def augment_batch(x):
    """On-device batch augmentation: reflect-pad 4 + a random 32x32 crop + random h-flip
    (batch-level offsets — cheap; diversity comes across batches/epochs)."""
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = int(torch.randint(0, 9, (1,))), int(torch.randint(0, 9, (1,)))
    out = xp[:, :, i:i + 32, j:j + 32]
    if torch.rand(1).item() < 0.5:
        out = torch.flip(out, dims=[3])
    return out.contiguous()


def ece(probs, y, bins=15):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    e, edges = 0.0, torch.linspace(0, 1, bins + 1)
    for j in range(bins):
        m = (conf > edges[j]) & (conf <= edges[j + 1])
        if m.any():
            e += (m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()).item()
    return e


def ood_auroc(fe_in, fe_ood):
    from sklearn.metrics import roc_auc_score
    s = torch.cat([fe_in.cpu(), fe_ood.cpu()]).numpy()
    lab = np.r_[np.zeros(len(fe_in)), np.ones(len(fe_ood))]
    return float(roc_auc_score(lab, s))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_genff_conv.py::test_augment_batch_shape_and_finite tests/test_genff_conv.py::test_ece_and_auroc -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add empirical/gpu_pipeline.py empirical/tests/test_genff_conv.py
git commit -m "feat(gpu-pipeline): GPU CIFAR data module + augment_batch + ECE/OOD metrics"
```

---

## Task 3: 4-arm CIFAR-10 experiment

**Files:** Create `empirical/experiments/genff_cifar.py`.

- [ ] **Step 1: Write the experiment**

Create `empirical/experiments/genff_cifar.py`:
```python
"""Conv gen-FF CIFAR-10 4-arm test (spec docs/superpowers/specs/2026-06-04-genff-conv-gpu-design.md):
BP | SCFF-conv+probe | SCFF-conv+head | gen-FF-conv(denoise+head). Does denoising-energy beat SCFF's
conv wall? Accuracy, ECE, OOD-AUROC, peak GPU memory, 1-forward.
Run: python experiments/genff_cifar.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_arch import ConvSCFF, _pooled, scff_local_step
from genff import EnergyHead, predict, free_energy
from genff_conv import conv_denoise_step, train_head_conv
from gpu_pipeline import to_gpu, augment_batch, ece, ood_auroc

DEV = "cuda"
CFG = dict(C=128, n_blocks=6, batch=128, n_train=50000, n_test=10000, seed=0,
           early_epochs=20, head_epochs=30, sigma=0.3,
           lr_denoise=0.02, lr_inbatch=0.05, tau=0.5, aug_noise=0.1,
           lr_head=1e-3, lam=0.2, lr_bp=1e-3, bp_epochs=20)

def mk():
    return ConvSCFF(CFG["C"], CFG["n_blocks"], "residual", 1.0 / math.sqrt(CFG["n_blocks"])).to(DEV)

def batches(n, bs, gen):
    idx = torch.randperm(n, generator=gen)
    for i in range(0, n - bs + 1, bs):
        yield idx[i:i + bs]

def train_denoise(m, Xtr):
    g = torch.Generator().manual_seed(CFG["seed"])
    for _ in range(CFG["early_epochs"]):
        for b in batches(len(Xtr), CFG["batch"], g):
            conv_denoise_step(m, augment_batch(Xtr[b]), dict(sigma=CFG["sigma"], lr=CFG["lr_denoise"]))

def train_inbatch(m, Xtr):
    g = torch.Generator().manual_seed(CFG["seed"])
    for _ in range(CFG["early_epochs"]):
        for b in batches(len(Xtr), CFG["batch"], g):
            xb = augment_batch(Xtr[b]); xp = augment_batch(Xtr[b])
            scff_local_step(m, xb, xp, CFG["tau"], CFG["lr_inbatch"])

def hc():
    return dict(epochs=CFG["head_epochs"], batch=CFG["batch"], lr=CFG["lr_head"],
               lam=CFG["lam"], sigma=CFG["sigma"], seed=CFG["seed"])

def probe(m, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    Ftr = m.features(Xtr).detach().cpu().numpy(); Fte = m.features(Xte).detach().cpu().numpy()
    clf = LogisticRegression(max_iter=200).fit(Ftr, ytr.cpu().numpy())
    acc = float((clf.predict(Fte) == yte.cpu().numpy()).mean())
    return acc, ece(torch.tensor(clf.predict_proba(Fte)), yte.cpu())

def head_eval(m, head, Xte, yte, Xood):
    with torch.no_grad():
        logits = head(m.features(Xte))
        acc = float((logits.argmax(1) == yte).float().mean())
        e = ece(torch.softmax(logits, 1).cpu(), yte.cpu())
    au = ood_auroc(free_energy(m, head, Xte), free_energy(m, head, Xood))
    return acc, e, au

def peakMB():
    return torch.cuda.max_memory_allocated() / 1e6

def main():
    assert torch.cuda.is_available(), "GPU required"
    Xtr, ytr, Xte, yte = to_gpu(CFG, DEV)
    C = int(ytr.max()) + 1; fd = CFG["C"] * CFG["n_blocks"]
    Xood = (3.0 * torch.randn(len(Xte), 3, 32, 32, device=DEV))
    print(f"CIFAR-10 train={len(Xtr)} test={len(Xte)} conv C={CFG['C']} L={CFG['n_blocks']} "
          f"on {torch.cuda.get_device_name(0)}\n")

    # arm 1: supervised-BP
    torch.cuda.reset_peak_memory_stats()
    m = mk(); head = EnergyHead(fd, C).to(DEV)
    opt = torch.optim.Adam(list(m.parameters()) + list(head.parameters()), lr=CFG["lr_bp"])
    lossf = torch.nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(CFG["seed"])
    for _ in range(CFG["bp_epochs"]):
        for b in batches(len(Xtr), CFG["batch"], g):
            loss = lossf(head(m.features(augment_batch(Xtr[b]))), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    a1, e1, au1 = head_eval(m, head, Xte, yte, Xood); mb1 = peakMB()
    print(f"  supervised-BP        acc={a1:.4f} ECE={e1:.4f} OOD={au1:.3f} peakMB={mb1:.0f} (1fwd)", flush=True)

    # arm 2: SCFF-conv + probe
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    m = mk(); train_inbatch(m, Xtr); mb2 = peakMB()
    a2, e2 = probe(m, Xtr, ytr, Xte, yte)
    print(f"  SCFF-conv+probe      acc={a2:.4f} ECE={e2:.4f} peakMB={mb2:.0f} (probe,1fwd)", flush=True)

    # arm 3: SCFF-conv + energy head (reuse the arm-2 backbone)
    head = EnergyHead(fd, C).to(DEV); train_head_conv(m, head, Xtr, ytr, hc())
    a3, e3, au3 = head_eval(m, head, Xte, yte, Xood)
    print(f"  SCFF-conv+head       acc={a3:.4f} ECE={e3:.4f} OOD={au3:.3f} (1fwd)", flush=True)

    # arm 4: gen-FF-conv (denoise + energy head)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    m = mk(); train_denoise(m, Xtr); mb4 = peakMB()
    head = EnergyHead(fd, C).to(DEV); train_head_conv(m, head, Xtr, ytr, hc())
    a4, e4, au4 = head_eval(m, head, Xte, yte, Xood)
    print(f"  gen-FF-conv          acc={a4:.4f} ECE={e4:.4f} OOD={au4:.3f} peakMB={mb4:.0f} (1fwd)", flush=True)

    print("\n=== VERDICT ===")
    print(f"  acc:  BP {a1:.3f} | gen-FF {a4:.3f} | SCFF+head {a3:.3f} | SCFF+probe {a2:.3f}")
    print(f"  gen-FF vs SCFF-wall (a4-a2): {a4-a2:+.4f}")
    print(f"  denoise vs in-batch (a4-a3): {a4-a3:+.4f}   head vs probe (a3-a2): {a3-a2:+.4f}")
    print(f"  ECE: probe {e2:.3f} | heads {e3:.3f}/{e4:.3f}   OOD: heads {au3:.3f}/{au4:.3f}")
    print(f"  peak MB: BP {mb1:.0f} | SCFF {mb2:.0f} | gen-FF {mb4:.0f}")
    print("=> " + ("gen-FF DODGES the conv wall (>> SCFF)" if a4 - a2 > 0.05
                   else "gen-FF ~= SCFF on conv: the conv gap is deeper than the objective"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run tiny**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -c "
import experiments.genff_cifar as E
E.CFG.update(C=32, n_blocks=3, n_train=1000, n_test=500, early_epochs=2, head_epochs=4, bp_epochs=2)
E.main()" 2>&1 | grep -vE "Warning|warn|STOP|Increase|scikit|n_iter|solver|preprocessing|documentation|refer"
```
Expected: four arms print `acc=…`, no exception. Fix only mechanical issues (device/shape); report any change. If the denoise arm diverges (NaN), report it — do not silently change lr.

- [ ] **Step 3: Commit**
```bash
git add empirical/experiments/genff_cifar.py
git commit -m "feat(genff-conv): CIFAR-10 4-arm experiment (BP/SCFF-probe/SCFF-head/gen-FF + metrics)"
```

- [ ] **Step 4: Full run (background, controller monitors)**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 nohup .venv/bin/python experiments/genff_cifar.py > /tmp/genff_cifar.out 2>&1 &
echo "launched pid $!"
```
Report the PID. The controller monitors `/tmp/genff_cifar.out` and writes the FINDINGS section.

- [ ] **Step 5: FINDINGS (controller, after the run)**

Add a `### Conv gen-FF (GPU)` subsection to `docs/FINDINGS.md` with the four arms' acc/ECE/OOD/peak-MB from `/tmp/genff_cifar.out`, the headline delta (gen-FF vs SCFF-wall a4−a2), and the honest verdict: does denoising-energy dodge the conv price-of-locality, or is the conv gap deeper than the objective? Then commit.

---

## Notes for execution

- **`TORCH_CUDA_ARCH_LIST=12.0`** is needed only for the SCFF in-batch arm (it JIT-loads the `scff_signal` kernel); harmless to always export.
- **Signs:** `conv_denoise_step` descends the paired-contrast loss (`add_(-lr*g)`); the energy-gap test guards it.
- **Memory caveat:** the early-train forward holds `O(L)` activations (not streamed) — peak MB will be ~BP-class, as found before; report it, don't claim a memory win here.
- **If gen-FF ≈ SCFF on conv** (a4 ≈ a2): that is the finding — the conv gap is the price of locality, not the objective; the generative reframe helps on MNIST but not on hard conv from scratch. Report honestly; it motivates the FF+BP hybrid (parked).
- **OOD:** uniform-noise is the easy OOD; the AUROC for the head arms is the generative payoff signal.
