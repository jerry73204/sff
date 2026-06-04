# Principled FF/BP hybrid (fine-tuning) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Match BP fine-tuning accuracy at less memory/compute under domain shift, by adapting early conv blocks with forward-only FF (no backward) and backpropagating only the tail.

**Architecture:** Pretrain a ConvSCFF backbone+head with BP on clean CIFAR-10; transfer to corrupted CIFAR-10 (blur+noise) three ways — full-BP, freeze-early+BP-tail, and hybrid (FF-denoise early + BP-tail, split at block `k`, gradient detached at the split). Compare accuracy vs peak memory / compute.

**Tech Stack:** PyTorch cu128 (RTX 5090), reuse `gpu_arch.ConvSCFF` + `genff_conv.conv_denoise_step` + `gpu_pipeline`. Export `TORCH_CUDA_ARCH_LIST=12.0` (denoise path imports nothing from the kernel, but harmless).

**Reuse (exist):** `gpu_arch.ConvSCFF(C,n_blocks,arch,alpha)` (`.forward(x)`→`[stem,block0,…]`, `apply_block(y,l)`, `pooled(y)`, attrs `C`,`n_blocks`,`stem`,`blocks`), `genff_conv.conv_denoise_step(model,x,cfg)` (per-location denoising-energy, forward-only), `gpu_pipeline.to_gpu(cfg,dev)`, `gpu_pipeline.augment_batch`. `experiments.cifar_conv.load_cifar`. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`. Do NOT push (controller pushes).

---

## File Structure

- `empirical/genff_conv.py` (modify) — add optional `layers=` arg to `conv_denoise_step` (adapt a subset of blocks).
- `empirical/hybrid.py` (create) — `corrupt`, `pretrain`, `eval_acc`, `finetune_full_bp`, `finetune_freeze_tail`, `finetune_hybrid`.
- `empirical/experiments/hybrid_cifar.py` (create) — pretrain once + `k`-sweep over freeze/hybrid + full-BP ceiling + metrics.
- `empirical/tests/test_hybrid.py` (create) — corruption, split-detach, freeze-immutability tests.

---

## Task 1: Domain shift + pretrain + eval

**Files:** Modify `empirical/genff_conv.py`; Create `empirical/hybrid.py`; Test `empirical/tests/test_hybrid.py`.

- [ ] **Step 1: Add `layers=` to `conv_denoise_step`**

In `empirical/genff_conv.py`, change the `conv_denoise_step` signature + loop to adapt a subset of blocks (backward-compatible — default = all):
```python
def conv_denoise_step(model, x, cfg, layers=None):
    """... layers: iterable of block indices to update (default: all)."""
    sig, lr = cfg["sigma"], cfg["lr"]
    xn = x + sig * torch.randn_like(x)
    with torch.no_grad():
        ys = [y.detach() for y in model(x)]
        ysn = [y.detach() for y in model(xn)]
    for l in (range(model.n_blocks) if layers is None else layers):
        Gr = _conv_branch(model, ys[l], l).pow(2).mean(1)
        Gn = _conv_branch(model, ysn[l], l).pow(2).mean(1)
        loss = -F.logsigmoid(Gr - Gn).mean()
        grads = torch.autograd.grad(loss, model.blocks[l].parameters())
        with torch.no_grad():
            for p, g in zip(model.blocks[l].parameters(), grads):
                p.add_(-lr * g)
```
Run the existing genff_conv tests to confirm no regression: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_genff_conv.py -q` → 5 passed.

- [ ] **Step 2: Write the failing test**

Create `empirical/tests/test_hybrid.py`:
```python
import torch, pytest
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

def test_corrupt_shifts_input():
    from hybrid import corrupt
    x = torch.rand(8, 3, 32, 32)
    xc = corrupt(x)
    assert xc.shape == (8, 3, 32, 32) and torch.isfinite(xc).all()
    assert (xc - x).abs().mean() > 0.02      # visibly shifted

@cuda_only
def test_pretrain_and_eval_run():
    from gpu_arch import ConvSCFF
    from hybrid import pretrain, eval_acc, head_for
    torch.manual_seed(0)
    X = torch.randn(256, 3, 32, 32, device="cuda"); y = torch.randint(0, 10, (256,), device="cuda")
    m = ConvSCFF(C=32, n_blocks=4, arch="residual", alpha=0.3).cuda(); h = head_for(m, 10)
    pretrain(m, h, X, y, dict(epochs=2, batch=64, lr=1e-3, seed=0))
    acc = eval_acc(m, h, X, y)
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 3: Run it (fails)**

Run: `cd /home/aeon/repos/sff/empirical && .venv/bin/python -m pytest tests/test_hybrid.py::test_corrupt_shifts_input -v`
Expected: FAIL — `ModuleNotFoundError: hybrid`.

- [ ] **Step 4: Create `hybrid.py` (corrupt + pretrain + eval)**

Create `empirical/hybrid.py`:
```python
"""Principled FF/BP hybrid for fine-tuning under domain shift.
Spec: docs/superpowers/specs/2026-06-04-principled-hybrid-design.md."""
import torch, torch.nn as nn, torch.nn.functional as F


def _gauss_kernel(ksize=5, sigma=1.2):
    ax = torch.arange(ksize, dtype=torch.float32) - ksize // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    return torch.outer(g, g)

def corrupt(x, blur_sigma=1.2, noise=0.2):
    """Fixed domain shift: gaussian blur (systematic) + additive noise. Same labels, new input dist."""
    k = _gauss_kernel(5, blur_sigma).to(x.device, x.dtype).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
    xb = F.conv2d(x, k, padding=2, groups=3)
    return xb + noise * torch.randn_like(xb)

def head_for(model, n_classes):
    """Linear classifier head on the pooled last-block feature (C-dim)."""
    return nn.Linear(model.C, n_classes).to(next(model.parameters()).device)

def _logits(model, head, x):
    return head(model.pooled(model(x)[-1]))

def pretrain(model, head, Xtr, ytr, cfg):
    """Full-BP pretraining of backbone+head on CLEAN data."""
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr) - cfg["batch"] + 1, cfg["batch"]):
            b = idx[i:i + cfg["batch"]]
            loss = lossf(_logits(model, head, Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()

def eval_acc(model, head, X, y, corrupt_fn=None, bs=1000):
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]; xb = corrupt_fn(xb) if corrupt_fn else xb
            correct += int((_logits(model, head, xb).argmax(1) == y[i:i + bs]).sum())
    return correct / len(X)
```

- [ ] **Step 5: Run tests (pass)**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_hybrid.py -v`
Expected: PASS (corrupt + pretrain/eval).

- [ ] **Step 6: Commit**
```bash
git add empirical/genff_conv.py empirical/hybrid.py empirical/tests/test_hybrid.py
git commit -m "feat(hybrid): domain-shift corrupt + pretrain + eval; conv_denoise_step layer subset"
```

---

## Task 2: The three fine-tuning arms

**Files:** Modify `empirical/hybrid.py`; Test `empirical/tests/test_hybrid.py`.

- [ ] **Step 1: Write the failing tests**

Append to `empirical/tests/test_hybrid.py`:
```python
@cuda_only
def test_hybrid_step_adapts_early_ff_and_tail_bp_no_cross():
    from gpu_arch import ConvSCFF
    from hybrid import head_for, finetune_hybrid
    torch.manual_seed(0)
    X = torch.randn(128, 3, 32, 32, device="cuda"); y = torch.randint(0, 10, (128,), device="cuda")
    m = ConvSCFF(C=32, n_blocks=4, arch="residual", alpha=0.3).cuda(); h = head_for(m, 10)
    k = 2
    e0 = m.blocks[0].weight.detach().clone()   # early (FF-adapted)
    t0 = m.blocks[3].weight.detach().clone()   # tail (BP)
    finetune_hybrid(m, h, X, y, k, dict(epochs=2, batch=64, lr=1e-3, sigma=0.5, lr_ff=0.5, seed=0))
    assert not torch.equal(m.blocks[0].weight, e0)   # early changed (FF)
    assert not torch.equal(m.blocks[3].weight, t0)   # tail changed (BP)

@cuda_only
def test_freeze_tail_leaves_early_unchanged():
    from gpu_arch import ConvSCFF
    from hybrid import head_for, finetune_freeze_tail
    torch.manual_seed(0)
    X = torch.randn(128, 3, 32, 32, device="cuda"); y = torch.randint(0, 10, (128,), device="cuda")
    m = ConvSCFF(C=32, n_blocks=4, arch="residual", alpha=0.3).cuda(); h = head_for(m, 10)
    e0 = m.blocks[1].weight.detach().clone()
    finetune_freeze_tail(m, h, X, y, 2, dict(epochs=2, batch=64, lr=1e-3, seed=0))
    assert torch.equal(m.blocks[1].weight, e0)       # frozen early bit-identical
```

- [ ] **Step 2: Run them (fail)**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_hybrid.py::test_freeze_tail_leaves_early_unchanged -v`
Expected: FAIL — `ImportError: cannot import name 'finetune_freeze_tail'`.

- [ ] **Step 3: Implement the arms**

Append to `empirical/hybrid.py`:
```python
def _batches(n, bs, gen):
    idx = torch.randperm(n, generator=gen)
    for i in range(0, n - bs + 1, bs):
        yield idx[i:i + bs]

def _tail_forward(model, x, k):
    """Forward early blocks 0..k-1 under no_grad (detached at the split), then tail k..L-1 WITH grad.
    Returns the pooled feature, differentiable wrt tail blocks only."""
    with torch.no_grad():
        h = F.relu(model.stem(x))
        for l in range(k):
            h = model.apply_block(h, l)
    h = h.detach()
    for l in range(k, model.n_blocks):
        h = model.apply_block(h, l)
    return model.pooled(h)

def finetune_full_bp(model, head, Xtr, ytr, cfg):
    """Ceiling: BP all blocks + head on the corrupted target."""
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        for b in _batches(len(Xtr), cfg["batch"], g):
            loss = lossf(_logits(model, head, corrupt(Xtr[b])), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()

def finetune_freeze_tail(model, head, Xtr, ytr, k, cfg):
    """Cheap baseline: blocks 0..k-1 frozen (no update), BP blocks k..L-1 + head."""
    tail = [p for l in range(k, model.n_blocks) for p in model.blocks[l].parameters()] + list(head.parameters())
    opt = torch.optim.Adam(tail, lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        for b in _batches(len(Xtr), cfg["batch"], g):
            loss = lossf(head(_tail_forward(model, corrupt(Xtr[b]), k)), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()

def finetune_hybrid(model, head, Xtr, ytr, k, cfg):
    """Hybrid: blocks 0..k-1 adapt via forward-only FF denoising (label-free); BP blocks k..L-1 + head.
    No gradient crosses the split (detached in _tail_forward)."""
    from genff_conv import conv_denoise_step
    tail = [p for l in range(k, model.n_blocks) for p in model.blocks[l].parameters()] + list(head.parameters())
    opt = torch.optim.Adam(tail, lr=cfg["lr"])
    lossf = nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        for b in _batches(len(Xtr), cfg["batch"], g):
            xc = corrupt(Xtr[b])
            if k > 0:                                            # FF-adapt early blocks (forward-only)
                conv_denoise_step(model, xc, dict(sigma=cfg["sigma"], lr=cfg["lr_ff"]), layers=range(k))
            loss = lossf(head(_tail_forward(model, xc, k)), ytr[b])   # BP tail+head
            opt.zero_grad(); loss.backward(); opt.step()
```

- [ ] **Step 4: Run tests (pass)**

Run: `cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -m pytest tests/test_hybrid.py -v`
Expected: PASS (early changed by FF, tail by BP; freeze leaves early bit-identical).

- [ ] **Step 5: Commit**
```bash
git add empirical/hybrid.py empirical/tests/test_hybrid.py
git commit -m "feat(hybrid): full-BP / freeze-tail / FF-early+BP-tail fine-tuning arms (split-detached)"
```

---

## Task 3: Sweep experiment + metrics

**Files:** Create `empirical/experiments/hybrid_cifar.py`.

- [ ] **Step 1: Write the experiment**

Create `empirical/experiments/hybrid_cifar.py`:
```python
"""Principled hybrid CIFAR fine-tuning (spec docs/superpowers/specs/2026-06-04-principled-hybrid-design.md).
Pretrain on clean CIFAR-10, transfer to corrupted CIFAR-10. Compare full-BP vs freeze+BP-tail vs
hybrid(FF-early+BP-tail) over a split sweep. Accuracy vs peak memory / compute.
Run: python experiments/hybrid_cifar.py"""
import os, sys, math, copy, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_arch import ConvSCFF
from gpu_pipeline import to_gpu
from hybrid import (corrupt, head_for, pretrain, eval_acc,
                    finetune_full_bp, finetune_freeze_tail, finetune_hybrid)

DEV = "cuda"
CFG = dict(C=128, n_blocks=6, batch=128, n_train=50000, n_test=10000, seed=0,
           pre_epochs=20, ft_epochs=12, lr=1e-3, sigma=0.5, lr_ff=0.5)
SPLITS = [1, 2, 3, 4, 5]   # k = number of early blocks (FF/frozen); tail = blocks[k:]+head

def snap(model, head):
    return (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
def restore(model, head, s):
    model.load_state_dict(s[0]); head.load_state_dict(s[1])

def run_arm(name, model, head, base, ft_fn, Xtr, ytr, Xte, yte):
    restore(model, head, base)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    ft_fn(model, head)
    torch.cuda.synchronize(); dt = time.time() - t0
    mb = torch.cuda.max_memory_allocated() / 1e6
    acc = eval_acc(model, head, Xte, yte, corrupt_fn=corrupt)
    print(f"  {name:24s} acc={acc:.4f}  peakMB={mb:.0f}  time={dt:.1f}s", flush=True)
    return acc, mb, dt

def main():
    assert torch.cuda.is_available(), "GPU required"
    Xtr, ytr, Xte, yte = to_gpu(CFG, DEV)
    L = CFG["n_blocks"]
    print(f"CIFAR-10 train={len(Xtr)} test={len(Xte)} conv C={CFG['C']} L={L} on {torch.cuda.get_device_name(0)}")
    m = ConvSCFF(CFG["C"], L, "residual", 1.0 / math.sqrt(L)).to(DEV); h = head_for(m, 10)
    pretrain(m, h, Xtr, ytr, dict(epochs=CFG["pre_epochs"], batch=CFG["batch"], lr=CFG["lr"], seed=CFG["seed"]))
    base = snap(m, h)
    clean = eval_acc(m, h, Xte, yte)
    shifted = eval_acc(m, h, Xte, yte, corrupt_fn=corrupt)
    print(f"\npretrained: clean acc={clean:.4f} | shifted (no finetune) acc={shifted:.4f}\n")

    ftc = dict(epochs=CFG["ft_epochs"], batch=CFG["batch"], lr=CFG["lr"], sigma=CFG["sigma"], lr_ff=CFG["lr_ff"], seed=CFG["seed"])
    a_full, mb_full, t_full = run_arm("full-BP (ceiling)", m, h, base,
        lambda mm, hh: finetune_full_bp(mm, hh, Xtr, ytr, ftc), Xtr, ytr, Xte, yte)
    print()
    print(f"  {'k':>2}  {'freeze acc':>10}  {'hybrid acc':>10}  {'freeze MB':>9}  {'hybrid MB':>9}  bwd-blocks")
    rows = []
    for k in SPLITS:
        af, mbf, _ = run_arm(f"freeze k={k}", m, h, base,
            lambda mm, hh, kk=k: finetune_freeze_tail(mm, hh, Xtr, ytr, kk, ftc), Xtr, ytr, Xte, yte)
        ah, mbh, _ = run_arm(f"hybrid k={k}", m, h, base,
            lambda mm, hh, kk=k: finetune_hybrid(mm, hh, Xtr, ytr, kk, ftc), Xtr, ytr, Xte, yte)
        rows.append((k, af, ah, mbf, mbh)); print(f"  {k:>2}  {af:>10.4f}  {ah:>10.4f}  {mbf:>9.0f}  {mbh:>9.0f}  {L-k}")

    print("\n=== VERDICT ===")
    print(f"  full-BP ceiling: acc={a_full:.4f} peakMB={mb_full:.0f} (bwd-blocks={L})")
    best = max(rows, key=lambda r: r[2])   # best hybrid acc
    print(f"  best hybrid: k={best[0]} acc={best[2]:.4f} peakMB={best[4]:.0f} vs freeze acc={best[1]:.4f}")
    gap_to_bp = a_full - best[2]; lift_over_freeze = best[2] - best[1]
    print(f"  hybrid gap to full-BP: {gap_to_bp:+.4f}   hybrid lift over freeze: {lift_over_freeze:+.4f}")
    print(f"  memory: best-hybrid {best[4]:.0f}MB vs full-BP {mb_full:.0f}MB ({mb_full/max(best[4],1):.2f}x)")
    print("=> " + ("HYBRID WINS: ~full-BP acc, beats freeze, < BP memory" if gap_to_bp < 0.02 and lift_over_freeze > 0.01
                   else "hybrid does not reach full-BP / does not beat freeze -- honest negative"))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run tiny**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 .venv/bin/python -c "
import experiments.hybrid_cifar as E
E.CFG.update(C=32, n_blocks=4, n_train=1000, n_test=500, pre_epochs=2, ft_epochs=2)
E.SPLITS=[1,2]
E.main()" 2>&1 | grep -vE "Warning|warn"
```
Expected: prints pretrained clean/shifted acc, then full-BP + freeze/hybrid rows for k=1,2, no exception. Fix only mechanical issues (device/shape); report any change.

- [ ] **Step 3: Commit**
```bash
git add empirical/experiments/hybrid_cifar.py
git commit -m "feat(hybrid): CIFAR fine-tuning sweep (full-BP vs freeze vs hybrid) + acc/mem/time metrics"
```

- [ ] **Step 4: Full run (background, controller monitors)**

Run:
```bash
cd /home/aeon/repos/sff/empirical && TORCH_CUDA_ARCH_LIST=12.0 nohup .venv/bin/python experiments/hybrid_cifar.py > /tmp/hybrid_cifar.out 2>&1 &
echo "launched pid $!"
```
Report the PID. The controller monitors `/tmp/hybrid_cifar.out` and writes FINDINGS.

- [ ] **Step 5: FINDINGS (controller, after run)**

Add a `## Principled hybrid (fine-tuning under domain shift)` section to `docs/FINDINGS.md` with: pretrained clean vs shifted accuracy; the full-BP ceiling; the freeze-vs-hybrid accuracy/memory table over `k`; and the verdict — does hybrid reach ~full-BP accuracy while beating freeze at less-than-BP memory (the revised-scope win), or honest negative. Then commit.

---

## Notes for execution

- **The split:** `k` = number of early blocks that are FF-adapted (hybrid) or frozen (freeze); tail = blocks `k..L-1` + head get BP. `_tail_forward` detaches at the split so no gradient enters the early blocks. The hybrid's backward graph spans `L-k` blocks → less memory/compute than full BP (`L`).
- **Stem** stays frozen in freeze/hybrid (pretrained input adapter); only `blocks[0:k]` are FF-adapted.
- **Memory honesty:** all arms do a full *forward*; the saving is the *backward* graph (`L-k` vs `L`). Report peak memory + the bwd-block count; the win is freeze/hybrid peakMB < full-BP peakMB.
- **If hybrid ≈ freeze** (FF early adaptation doesn't help): honest negative — under this shift, frozen features suffice or FF can't improve them. If so, try a stronger corruption (raise `blur_sigma`/`noise` in `corrupt`) so frozen features genuinely degrade, leaving room for FF to recover.
- **κ analysis** (does per-layer κ predict the best `k`) is a follow-up, not in this plan — keep the headline the accuracy-vs-cost sweep.
