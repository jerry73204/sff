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
import metrics  # noqa: F401  (kept for parity / future metrics)

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


def scff_step_arch(model, x, xp, tau, lr):
    # local_grad uses autograd -> grad must stay enabled; only the update is no_grad.
    grads = [local_grad(model, x, xp, l, tau) for l in range(model.n_layers)]
    with torch.no_grad():
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
    print("\n=== INIT Aniso by (arch, depth) ===")
    for arch in CFG["archs"]:
        vals = [f"L{L}:{np.mean([r['aniso'] for r in init_rows if r['arch']==arch and r['L']==L]):.3f}"
                for L in CFG["depths"]]
        print(f"  {arch:9s} " + "  ".join(vals))
    print("\n=== PERSISTENCE A (init -> final, L=8) ===")
    finals = {}
    for arch in CFG["archs"]:
        Acur = pdata[arch]["A"]; a0, aN = Acur[:, 0].mean(), Acur[:, -1].mean()
        finals[arch] = aN
        print(f"  {arch:9s} A {a0:.3f} -> {aN:.3f}")
    plainN = finals.get("plain", 0.0)
    print("\nVERDICT: residual/dense beat plain final A (+0.05)?",
          {a: bool(v > plainN + 0.05) for a, v in finals.items() if a != "plain"})


if __name__ == "__main__":
    main()
