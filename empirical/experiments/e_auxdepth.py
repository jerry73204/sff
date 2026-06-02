"""LoCo-style auxiliary-depth (look-ahead) SCFF: does giving each block's local goodness
`j` extra DOWNSTREAM layers of context raise alignment A and make it PERSIST during
training, vs j=0 (vanilla SCFF)?

Hypothesis (falsifiable): larger j raises init A AND improves persistence (A stays higher)
vs j=0, because the local objective sees more downstream (the LoCo "overlap adds depth +
implicit feedback" idea). Negative/partial is valuable.

(a) INIT  : mean 1-A over non-final blocks, per j, per arch.
(b) PERSIST: train with aux_scff_step, track mean A vs step (~150 steps), per j, per arch.

Cost: look-ahead j costs j extra forward-block depths per gradient (per block, per step).
"""
from __future__ import annotations
import os, sys, csv

import torch
import numpy as np

torch.set_default_dtype(torch.float64)   # match tests; alignment math is dtype-sensitive

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP, global_grad, alignment_cosine
from auxdepth import aux_local_grad, aux_scff_step

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS, PLOTS = os.path.join(HERE, "runs"), os.path.join(HERE, "plots")

CFG = dict(d_in=64, width=256, n_layers=6, batch=4, tau=0.5, act="linear",
           archs=["plain", "residual"], alpha=0.1, js=[0, 1, 2],
           seeds=[0, 1, 2], steps=150, log_every=15, lr=0.02)


def _data(seed, cfg):
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(cfg["batch"], cfg["d_in"], generator=g)
    xp = x + 0.1 * torch.randn(x.shape, generator=g)
    return x, xp


def mean_A_aux(model, x, xp, tau, j):
    """Mean alignment over non-final blocks. A^(l) = cos(aux_local_grad(j), global_grad).
    The global (BP) gradient is the SAME fixed end-to-end probe regardless of j; only the
    LOCAL gradient changes with the look-ahead j. So a higher A means the look-ahead local
    direction is closer to true backprop."""
    As = []
    for l in range(model.n_layers - 1):           # non-final blocks
        gl = aux_local_grad(model, x, xp, l, tau, j)
        gg = global_grad(model, x, xp, l, tau)
        As.append(alignment_cosine(gl, gg))
    return float(np.mean(As))


def make_model(cfg, arch, seed):
    return ArchMLP(cfg["d_in"], cfg["width"], cfg["n_layers"], arch,
                   cfg["act"], alpha=cfg["alpha"], seed=seed)


def run_init(cfg):
    rows = []
    for arch in cfg["archs"]:
        for j in cfg["js"]:
            for seed in cfg["seeds"]:
                x, xp = _data(seed, cfg)
                m = make_model(cfg, arch, seed)
                A = mean_A_aux(m, x, xp, cfg["tau"], j)
                rows.append(dict(arch=arch, j=j, seed=seed, one_minus_A=1 - A))
        print(f"init {arch} done", flush=True)
    return rows


def run_persist(cfg):
    out = {}
    for arch in cfg["archs"]:
        for j in cfg["js"]:
            curves = []
            for seed in cfg["seeds"]:
                x, xp = _data(seed, cfg)
                m = make_model(cfg, arch, seed)
                A_t, steps = [], []
                for step in range(cfg["steps"] + 1):
                    if step % cfg["log_every"] == 0:
                        # measure A of THIS j's local grad vs the fixed BP probe
                        A_t.append(mean_A_aux(m, x, xp, cfg["tau"], j))
                        steps.append(step)
                    aux_scff_step(m, x, xp, cfg["tau"], cfg["lr"], j)
                curves.append(A_t)
            out[(arch, j)] = dict(steps=steps, A=np.array(curves))
            print(f"persist {arch} j={j} done", flush=True)
    return out


def plot_persist(pdata, cfg, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cfg["archs"]), figsize=(5.5 * len(cfg["archs"]), 4.3),
                             squeeze=False)
    for ax, arch in zip(axes[0], cfg["archs"]):
        for j in cfg["js"]:
            d = pdata[(arch, j)]
            mu, sd = d["A"].mean(0), d["A"].std(0)
            ax.plot(d["steps"], mu, marker="o", label=f"j={j}")
            ax.fill_between(d["steps"], mu - sd, mu + sd, alpha=0.2)
        ax.axhline(1.0, color="gray", lw=0.6, ls="--")
        ax.set_xlabel("step"); ax.set_ylabel("A (mean non-final)")
        ax.set_title(f"{arch} (alpha={cfg['alpha']})"); ax.legend()
    fig.suptitle("Auxiliary-depth look-ahead j: persistence of SCFF<->BP alignment")
    fig.tight_layout(); fig.savefig(path, dpi=130); print("plot ->", path)


def main():
    os.makedirs(RUNS, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    init_rows = run_init(CFG)
    with open(os.path.join(RUNS, "auxdepth_init.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "j", "seed", "one_minus_A"])
        w.writeheader(); w.writerows(init_rows)

    pdata = run_persist(CFG)
    persist_rows = []
    for (arch, j), d in pdata.items():
        for si, curve in enumerate(d["A"]):
            for st, a in zip(d["steps"], curve):
                persist_rows.append(dict(arch=arch, j=j, seed=si, step=st, A=a))
    with open(os.path.join(RUNS, "auxdepth_persist.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch", "j", "seed", "step", "A"])
        w.writeheader(); w.writerows(persist_rows)
    plot_persist(pdata, CFG, os.path.join(PLOTS, "auxdepth_persist.png"))

    # ---- tables + verdict ----
    print("\n=== INIT 1-A by (arch, j)  [lower = better aligned] ===")
    for arch in CFG["archs"]:
        cells = []
        for j in CFG["js"]:
            v = [r["one_minus_A"] for r in init_rows if r["arch"] == arch and r["j"] == j]
            cells.append(f"j{j}:{np.mean(v):.3f}+-{np.std(v):.3f}")
        print(f"  {arch:9s} " + "   ".join(cells))

    print("\n=== PERSISTENCE A: init-step -> final-step (mean non-final) ===")
    summary = {}
    for arch in CFG["archs"]:
        for j in CFG["js"]:
            A = pdata[(arch, j)]["A"]
            a0, aN = A[:, 0].mean(), A[:, -1].mean()
            summary[(arch, j)] = (a0, aN)
        cells = [f"j{j}: {summary[(arch,j)][0]:.3f}->{summary[(arch,j)][1]:.3f}"
                 for j in CFG["js"]]
        print(f"  {arch:9s} " + "   ".join(cells))

    print("\n=== VERDICT ===")
    for arch in CFG["archs"]:
        base0 = 1 - np.mean([r["one_minus_A"] for r in init_rows
                             if r["arch"] == arch and r["j"] == 0])
        for j in CFG["js"][1:]:
            initA = 1 - np.mean([r["one_minus_A"] for r in init_rows
                                 if r["arch"] == arch and r["j"] == j])
            d_init = initA - base0
            d_fin = summary[(arch, j)][1] - summary[(arch, 0)][1]
            print(f"  {arch:9s} j={j}: init A {d_init:+.3f} vs j=0,  "
                  f"final A {d_fin:+.3f} vs j=0  "
                  f"-> {'RAISES' if d_init>0.01 else ('FLAT' if abs(d_init)<=0.01 else 'LOWERS')} init; "
                  f"{'BETTER' if d_fin>0.01 else ('SAME' if abs(d_fin)<=0.01 else 'WORSE')} persist")
    print(f"\n  cost: look-ahead j adds j extra forward-block depths per local gradient "
          f"(per block, per step); total fwd-depth ~ (1+j)x per gradient.")


if __name__ == "__main__":
    main()
