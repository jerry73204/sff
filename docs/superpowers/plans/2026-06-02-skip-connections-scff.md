# Skip-Connection SCFF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add residual and dense skip-connection architectures to the SCFF empirical harness and measure, across a depth sweep, whether they lift gradient-alignment at init and rescue persistence vs plain SCFF.

**Architecture:** A self-contained `arch.py` holds a new `ArchMLP` (stem + `L` blocks, `arch ∈ {plain,residual,dense}`) and arch-aware autograd helpers (local goodness gradient, generic downstream Jacobian). Existing `model.py`/`gradients.py`/`metrics.py` are reused (arch-agnostic parts only) and left unchanged, so prior experiments keep working. A new experiment sweeps `L ∈ {4,8,16} × arch` measuring init `A`/`Aniso`/`δ` and persistence.

**Tech Stack:** PyTorch (CPU, uv-managed), pytest, matplotlib. Run commands use the venv: `cd empirical && .venv/bin/python ...`, tests `.venv/bin/python -m pytest`.

---

### Task 1: `ArchMLP` forward (stem + plain/residual/dense)

**Files:**
- Create: `empirical/arch.py`
- Test: `empirical/tests/test_arch.py`

- [ ] **Step 1: Write the failing test**

```python
# empirical/tests/test_arch.py
import math
import torch
import pytest
from arch import ArchMLP

torch.set_default_dtype(torch.float64)


def _x(B=4, d=8, seed=0):
    return torch.randn(B, d, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("arch", ["plain", "residual", "dense"])
@pytest.mark.parametrize("L", [4, 8])
def test_forward_shapes(arch, L):
    m = ArchMLP(d_in=8, width=16, n_layers=L, arch=arch, act="linear", seed=1)
    ys = m(_x())
    assert len(ys) == L + 1                      # [y0=stem, y1..yL]
    for y in ys:
        assert y.shape == (4, 16)


def test_residual_alpha_zero_is_identity():
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch="residual", act="linear",
                alpha=0.0, seed=2)
    ys = m(_x())
    for l in range(1, len(ys)):                  # each block is a pure skip
        assert torch.allclose(ys[l], ys[l - 1], atol=1e-12)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q`
Expected: FAIL (`No module named 'arch'`).

- [ ] **Step 3: Write minimal implementation**

```python
# empirical/arch.py
"""Skip-connection architectures (residual, dense) for the SCFF alignment study,
plus arch-aware autograd helpers. See docs/superpowers/specs/2026-06-02-skip-connections-scff-design.md.

ArchMLP: stem (d_in->width) then L blocks at fixed width.
  plain     y^l = act(W^l y^{l-1})
  residual  y^l = y^{l-1} + alpha * act(W^l y^{l-1})        (alpha default 1/sqrt(L))
  dense     y^l = act(W^l concat(y^0,...,y^{l-1}))
forward returns [y0=stem(x), y1, ..., yL].
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from model import normalize


class ArchMLP(nn.Module):
    def __init__(self, d_in, width, n_layers, arch="plain", act="linear",
                 alpha=None, seed=None):
        super().__init__()
        assert arch in ("plain", "residual", "dense")
        assert act in ("linear", "relu")
        self.arch, self.act_name = arch, act
        self.width, self.n_layers = width, n_layers
        self.alpha = (1.0 / math.sqrt(n_layers)) if alpha is None else alpha
        if seed is not None:
            torch.manual_seed(seed)
        self.stem = nn.Parameter(torch.randn(width, d_in) / math.sqrt(d_in))
        self.W = nn.ParameterList()
        for l in range(n_layers):
            fan_in = width * (l + 1) if arch == "dense" else width
            w = torch.randn(width, fan_in) / math.sqrt(fan_in)
            if l == n_layers - 1:                # mu-P last-block extra 1/sqrt(fan_in)
                w = w / math.sqrt(fan_in)
            self.W.append(nn.Parameter(w))

    def act(self, t):
        return t if self.act_name == "linear" else torch.relu(t)

    def _block(self, ys, l):
        if self.arch == "plain":
            return self.act(ys[-1] @ self.W[l].t())
        if self.arch == "residual":
            return ys[-1] + self.alpha * self.act(ys[-1] @ self.W[l].t())
        return self.act(torch.cat(ys, dim=1) @ self.W[l].t())   # dense

    def forward(self, x):
        ys = [self.act(x @ self.stem.t())]
        for l in range(self.n_layers):
            ys.append(self._block(ys, l))
        return ys
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q`
Expected: PASS (6 forward-shape cases + identity).

- [ ] **Step 5: Commit**

```bash
git add empirical/arch.py empirical/tests/test_arch.py
git commit -m "arch: ArchMLP forward (stem + plain/residual/dense)"
```

---

### Task 2: Layer-local rep + `forward_from` (rep helpers for grads/Jacobian)

**Files:**
- Modify: `empirical/arch.py`
- Test: `empirical/tests/test_arch.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_arch.py
def test_forward_from_reconstructs():
    """forward_from at the true intermediate value reproduces y^L."""
    m = ArchMLP(d_in=8, width=16, n_layers=6, arch="dense", act="relu", seed=3)
    x = _x()
    frozen = [y.detach() for y in m(x)]
    for start in range(1, m.n_layers + 1):
        yL = m.forward_from(start, frozen[start], frozen)
        assert torch.allclose(yL, frozen[-1], atol=1e-10)


def test_block_output_differentiable_wrt_W():
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch="residual", act="linear", seed=4)
    x = _x()
    z = m.block_output(x, layer=1)               # z^(2), function of W[1] only
    assert z.shape == (4, 16)
    g = torch.autograd.grad(z.sum(), m.W[1])[0]
    assert g.shape == m.W[1].shape and torch.isfinite(g).all()
    # other blocks' weights get no gradient (layer-local)
    g0 = torch.autograd.grad(z.sum(), m.W[0], allow_unused=True)[0]
    assert g0 is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q -k "forward_from or block_output"`
Expected: FAIL (`ArchMLP has no attribute 'forward_from'`).

- [ ] **Step 3: Write minimal implementation**

```python
# add methods to ArchMLP in arch.py
    def block_output(self, x, layer):
        """z^(layer+1) as a differentiable function of W[layer] only; all block
        inputs detached (layer-local, stop-grad between blocks)."""
        with torch.no_grad():
            ys = [y.detach() for y in self.forward(x)]
        if self.arch == "plain":
            out = self.act(ys[layer] @ self.W[layer].t())
        elif self.arch == "residual":
            out = ys[layer] + self.alpha * self.act(ys[layer] @ self.W[layer].t())
        else:  # dense: concat of y^0..y^layer (detached) @ W[layer]
            out = self.act(torch.cat(ys[:layer + 1], dim=1) @ self.W[layer].t())
        return normalize(out)

    def forward_from(self, start, y_start, frozen):
        """Recompute y^L given y^(start) = y_start, holding frozen[0..start-1].
        `frozen` is a full forward(x) (detached); used for dense concat paths."""
        ys = list(frozen[:start]) + [y_start]
        for l in range(start, self.n_layers):
            ys.append(self._block(ys, l))
        return ys[-1]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q -k "forward_from or block_output"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add empirical/arch.py empirical/tests/test_arch.py
git commit -m "arch: layer-local block_output + forward_from helpers"
```

---

### Task 3: Arch-aware local goodness gradient + alignment

**Files:**
- Modify: `empirical/arch.py`
- Test: `empirical/tests/test_arch.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_arch.py
from gradients import global_grad, alignment_cosine
import arch as A


@pytest.mark.parametrize("a", ["plain", "residual", "dense"])
def test_local_and_global_grad_shapes(a):
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch=a, act="linear", seed=5)
    x = _x(); xp = _x(seed=6)
    gl = A.local_grad(m, x, xp, layer=1, tau=0.5)
    gg = global_grad(m, x, xp, layer=1, tau=0.5)
    assert gl.shape == m.W[1].shape == gg.shape
    c = alignment_cosine(gl, gg)
    assert -1.0 - 1e-9 <= c <= 1.0 + 1e-9


def test_last_block_is_self_aligned():
    """For the final block, local == global goodness gradient => A = 1."""
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch="plain", act="linear", seed=7)
    x = _x(); xp = _x(seed=8)
    last = m.n_layers - 1
    gl = A.local_grad(m, x, xp, layer=last, tau=0.5)
    gg = global_grad(m, x, xp, layer=last, tau=0.5)
    assert alignment_cosine(gl, gg) == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q -k "grad_shapes or self_aligned"`
Expected: FAIL (`module 'arch' has no attribute 'local_grad'`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to arch.py (module-level, after the class)
from gradients import local_goodness


def local_grad(model, x, x_pos, layer, tau):
    """grad_{W[layer]} g^(layer) via autograd of the layer-local block goodness."""
    z = model.block_output(x, layer)
    zp = model.block_output(x_pos, layer)
    g = local_goodness(z, zp.detach(), tau)
    return torch.autograd.grad(g, model.W[layer])[0]
```

`global_grad` (from `gradients.py`) already works on `ArchMLP` unchanged: it calls
`model(x)`, normalizes the last rep, and differentiates `g^(L)` w.r.t. `model.W[layer]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q -k "grad_shapes or self_aligned"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add empirical/arch.py empirical/tests/test_arch.py
git commit -m "arch: arch-aware local goodness gradient + reuse global probe"
```

---

### Task 4: Generic downstream Jacobian (autograd) + match plain linear

**Files:**
- Modify: `empirical/arch.py`
- Test: `empirical/tests/test_arch.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_arch.py
def test_downstream_jacobian_matches_linear_product_plain():
    """For a plain LINEAR ArchMLP, the autograd M = d y^L / d y^(layer+1) equals the
    product of downstream block weights W[L-1] ... W[layer+1] (sample-independent)."""
    m = ArchMLP(d_in=8, width=10, n_layers=4, arch="plain", act="linear", seed=9)
    x = _x()
    for layer in range(m.n_layers):
        M = A.downstream_jacobian(m, x, layer)            # [n, n]
        expect = torch.eye(m.width)
        for k in range(layer + 1, m.n_layers):
            expect = m.W[k] @ expect
        assert torch.allclose(M, expect, atol=1e-8), f"layer {layer}"


def test_downstream_jacobian_identity_last_block():
    m = ArchMLP(d_in=8, width=10, n_layers=4, arch="dense", act="relu", seed=10)
    M = A.downstream_jacobian(m, _x(), layer=m.n_layers - 1)
    assert torch.allclose(M, torch.eye(m.width), atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q -k downstream`
Expected: FAIL (`module 'arch' has no attribute 'downstream_jacobian'`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to arch.py (module-level)
def downstream_jacobian(model, x, layer):
    """M^(l+1->L) = mean_i d y^L_i / d y^(l+1)_i, via per-sample autograd jacobian.
    Arch-agnostic (handles dense concat via forward_from). [n_L, n]. Identity if
    `layer` is the last block."""
    start = layer + 1
    with torch.no_grad():
        frozen_all = [y.detach() for y in model(x)]
    B = x.shape[0]
    Ms = []
    for i in range(B):
        fz = [f[i:i + 1].detach() for f in frozen_all]    # per-sample frozen reps
        y0 = fz[start].squeeze(0)                          # [n]
        def fn(yv, _fz=fz, _start=start):
            return model.forward_from(_start, yv.unsqueeze(0), _fz).squeeze(0)
        Ms.append(torch.autograd.functional.jacobian(fn, y0))   # [n_L, n]
    return torch.stack(Ms).mean(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q -k downstream`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add empirical/arch.py empirical/tests/test_arch.py
git commit -m "arch: generic autograd downstream Jacobian (matches linear product)"
```

---

### Task 5: Per-block measurement (A, Aniso, delta)

**Files:**
- Modify: `empirical/arch.py`
- Test: `empirical/tests/test_arch.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_arch.py
@pytest.mark.parametrize("a", ["plain", "residual", "dense"])
def test_measure_blocks(a):
    m = ArchMLP(d_in=8, width=32, n_layers=4, arch=a, act="linear", seed=11)
    rows = A.measure_blocks(m, _x(), _x(seed=12), tau=0.5)
    assert len(rows) == m.n_layers
    for (l, Aval, dg, an, dV) in rows:
        assert -1.0 - 1e-9 <= Aval <= 1.0 + 1e-9
        assert dg >= 0.0 and an >= 0.0 and dV >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q -k measure`
Expected: FAIL (`no attribute 'measure_blocks'`).

- [ ] **Step 3: Write minimal implementation**

```python
# add to arch.py (module-level)
from gradients import signal
import metrics


def measure_blocks(model, x, x_pos, tau):
    """Per block: (layer, A, delta_gram, aniso, d_V). Reuses metrics.* (arch-agnostic)."""
    ys, ysp = model(x), model(x_pos)
    zL = normalize(ys[-1])
    rows = []
    for l in range(model.n_layers):
        gl = local_grad(model, x, x_pos, l, tau)
        gg = global_grad(model, x, x_pos, l, tau)
        Aval = alignment_cosine(gl, gg)
        z_l, zp_l = normalize(ys[l + 1]), normalize(ysp[l + 1])
        dg = metrics.delta_gram(z_l, zL)
        s, _ = signal(z_l, zp_l.detach(), tau)
        V, dV = metrics.contrastive_subspace(s)
        M = downstream_jacobian(model, x, l)
        an = metrics.aniso(M, V)
        rows.append((l, Aval, dg, an, dV))
    return rows
```

Add the imports `from gradients import global_grad, alignment_cosine` at the top of
`arch.py` if not already present.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd empirical && .venv/bin/python -m pytest tests/test_arch.py -q`
Expected: PASS (all arch tests).

- [ ] **Step 5: Commit**

```bash
git add empirical/arch.py empirical/tests/test_arch.py
git commit -m "arch: per-block measurement (A, delta_gram, aniso, d_V)"
```

---

### Task 6: Depth × arch experiment (init scaling + persistence + plots)

**Files:**
- Create: `empirical/experiments/e_arch_depth.py`

- [ ] **Step 1: Write the experiment script**

```python
# empirical/experiments/e_arch_depth.py
"""Depth x architecture sweep: do residual/dense skips lift SCFF alignment and rescue
persistence vs plain? (design: docs/superpowers/specs/2026-06-02-skip-connections-scff-design.md)

(a) init: 1-A, Aniso, delta vs depth L, per arch.
(b) persistence: A vs step at L=8, per arch.
"""
from __future__ import annotations
import os, sys, csv
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP, measure_blocks, local_grad
import metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS, PLOTS = os.path.join(HERE, "runs"), os.path.join(HERE, "plots")

CFG = dict(d_in=16, width=128, batch=4, tau=0.5, act="linear",
           depths=[4, 8, 16], archs=["plain", "residual", "dense"],
           seeds=[0, 1, 2, 3, 4], persist_L=8, steps=200, log_every=20, lr=0.02)


def _data(seed, cfg):
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(cfg["batch"], cfg["d_in"], generator=g)
    xp = x + 0.1 * torch.randn(x.shape, generator=g)
    return x, xp


def mean_nonfinal(rows, idx):
    nf = rows[:-1] if len(rows) > 1 else rows
    return float(np.mean([r[idx] for r in nf]))


@torch.no_grad()
def scff_step_arch(model, x, xp, tau, lr):
    grads = [local_grad(model, x, xp, l, tau) for l in range(model.n_layers)]
    for l in range(model.n_layers):
        model.W[l].add_(lr * grads[l])


def run_init(cfg):
    rows = []
    for arch in cfg["archs"]:
        for L in cfg["depths"]:
            for seed in cfg["seeds"]:
                x, xp = _data(seed, cfg)
                m = ArchMLP(cfg["d_in"], cfg["width"], L, arch, cfg["act"], seed=seed)
                mb = measure_blocks(m, x, xp, cfg["tau"])
                rows.append(dict(arch=arch, L=L, seed=seed,
                                 one_minus_A=1 - mean_nonfinal(mb, 1),
                                 aniso=mean_nonfinal(mb, 3),
                                 dgram=mean_nonfinal(mb, 2)))
            print(f"init {arch} L={L} done", flush=True)
    return rows


def run_persist(cfg):
    out = {}
    for arch in cfg["archs"]:
        curves = []
        for seed in cfg["seeds"]:
            x, xp = _data(seed, cfg)
            m = ArchMLP(cfg["d_in"], cfg["width"], cfg["persist_L"], arch, cfg["act"], seed=seed)
            A_t, steps = [], []
            for step in range(cfg["steps"] + 1):
                if step % cfg["log_every"] == 0:
                    mb = measure_blocks(m, x, xp, cfg["tau"])
                    A_t.append(mean_nonfinal(mb, 1)); steps.append(step)
                scff_step_arch(m, x, xp, cfg["tau"], cfg["lr"])
            curves.append(A_t)
        out[arch] = dict(steps=steps, A=np.array(curves))
        print(f"persist {arch} done", flush=True)
    return out


def plot_init(rows, cfg, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, ylab in [(axes[0], "one_minus_A", r"$1-A$ (init)"),
                          (axes[1], "aniso", "Aniso (init)")]:
        for arch in cfg["archs"]:
            mu = [np.mean([r[key] for r in rows if r["arch"] == arch and r["L"] == L])
                  for L in cfg["depths"]]
            sd = [np.std([r[key] for r in rows if r["arch"] == arch and r["L"] == L])
                  for L in cfg["depths"]]
            ax.errorbar(cfg["depths"], mu, yerr=sd, marker="o", capsize=3, label=arch)
        ax.set_xlabel("depth L"); ax.set_ylabel(ylab); ax.legend()
    fig.suptitle("Skip connections vs depth (init): does the cross-layer ceiling lift?")
    fig.tight_layout(); fig.savefig(path, dpi=130); print("plot ->", path)


def plot_persist(pdata, cfg, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.3))
    for arch in cfg["archs"]:
        st, Acur = pdata[arch]["steps"], pdata[arch]["A"]
        mu, sd = Acur.mean(0), Acur.std(0)
        ax.plot(st, mu, marker="o", label=arch); ax.fill_between(st, mu - sd, mu + sd, alpha=0.2)
    ax.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("step"); ax.set_ylabel("A (mean non-final)")
    ax.set_title(f"Persistence at L={cfg['persist_L']}"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); print("plot ->", path)


def main():
    os.makedirs(RUNS, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    init_rows = run_init(CFG)
    with open(os.path.join(RUNS, "arch_init.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "L", "seed", "one_minus_A", "aniso", "dgram"])
        w.writeheader(); w.writerows(init_rows)
    with open(os.path.join(RUNS, "arch.yaml"), "w") as f:
        yaml.safe_dump(CFG, f)
    plot_init(init_rows, CFG, os.path.join(PLOTS, "arch_init.png"))
    pdata = run_persist(CFG)
    plot_persist(pdata, CFG, os.path.join(PLOTS, "arch_persist.png"))

    print("\n=== INIT 1-A by (arch, depth) ===")
    for arch in CFG["archs"]:
        vals = [f"L{L}:{np.mean([r['one_minus_A'] for r in init_rows if r['arch']==arch and r['L']==L]):.3f}"
                for L in CFG["depths"]]
        print(f"  {arch:9s} " + "  ".join(vals))
    print("\n=== PERSISTENCE A (init -> final, L=8) ===")
    verdict = []
    for arch in CFG["archs"]:
        A = pdata[arch]["A"]; a0, aN = A[:, 0].mean(), A[:, -1].mean()
        print(f"  {arch:9s} A {a0:.3f} -> {aN:.3f}")
        verdict.append((arch, aN))
    plainN = dict(verdict).get("plain", 0.0)
    print("\nVERDICT: residual/dense beat plain on final A?",
          {a: (v > plainN + 0.05) for a, v in verdict if a != "plain"})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run with a tiny config**

Run:
```bash
cd empirical && .venv/bin/python -c "
import experiments.e_arch_depth as E
E.CFG.update(depths=[4], archs=['plain','residual','dense'], seeds=[0], steps=20, log_every=10, persist_L=4, width=32)
E.main()"
```
Expected: prints init table + persistence lines, writes 2 plots, no errors.

- [ ] **Step 3: Commit**

```bash
git add empirical/experiments/e_arch_depth.py
git commit -m "exp: depth x arch sweep (init scaling + persistence)"
```

---

### Task 7: Full run + record findings

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the full sweep**

Run: `cd empirical && .venv/bin/python experiments/e_arch_depth.py 2>&1 | tail -25`
Expected: init table (1-A per arch per depth), persistence (init->final A per arch), verdict dict; `plots/arch_init.png` and `plots/arch_persist.png` written. (May take a few minutes — per-sample Jacobians.)

- [ ] **Step 2: Run the whole test suite**

Run: `cd empirical && .venv/bin/python -m pytest -q`
Expected: PASS (prior 12 + new arch tests).

- [ ] **Step 3: Record the finding in README**

Add a row/paragraph under Track-E status summarizing the result honestly (whether residual/dense raise init A with a widening depth gap, lower Aniso, and persist — including partial/negative outcomes).

- [ ] **Step 4: Commit + push**

```bash
git add README.md
git commit -m "Track E: skip-connection sweep results (residual/dense vs plain)"
git push origin main
```

---

## Self-Review

**Spec coverage:** §1 architecture → Task 1; §2 gradients/Jacobian → Tasks 2–4; §3 metrics reuse → Task 5; §4 experiment → Tasks 6–7; §5 testing → tests across Tasks 1–5 + Task 7 suite run. All covered.

**Placeholder scan:** no TBD/TODO; every code step has full code; README step (Task 7 Step 3) is intentionally results-dependent (the finding is unknown until the run) — acceptable, it is data entry not code.

**Type consistency:** `ArchMLP` API (`forward`, `_block`, `block_output`, `forward_from`, `.W`, `.width`, `.n_layers`, `.alpha`, `.act_name`, `.act`) used consistently; module fns `local_grad`, `downstream_jacobian`, `measure_blocks` referenced as `arch.local_grad` etc.; `measure_blocks` returns `(l, A, dgram, aniso, dV)` tuples consumed by `mean_nonfinal(rows, idx)` with idx 1=A, 2=dgram, 3=aniso — consistent. `global_grad`, `alignment_cosine`, `local_goodness`, `signal` reused from `gradients.py` with existing signatures; `metrics.delta_gram/contrastive_subspace/aniso` unchanged.
