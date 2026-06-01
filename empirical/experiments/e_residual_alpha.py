"""Residual scale (alpha) sweep + ReLU residual.

Questions:
  (1) How small must the residual scale alpha be for SCFF alignment to hold/persist?
      Theory: M = prod(I + alpha J) ~= I needs small alpha; alpha->large -> plain.
  (2) Does it survive ReLU (M = prod(I + alpha D J), D = activation mask)?

Residual ArchMLP at L=8; for alpha in a sweep and act in {linear, relu}: init 1-A and
Aniso, plus persistence (final A after SCFF training). 5 seeds.

Run: python experiments/e_residual_alpha.py
"""
from __future__ import annotations
import os, sys, csv
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP, measure_blocks
from experiments.e_arch_depth import scff_step_arch, mean_nonfinal, _data

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS, PLOTS = os.path.join(HERE, "runs"), os.path.join(HERE, "plots")

CFG = dict(d_in=16, width=128, batch=4, tau=0.5, L=8,
           alphas=[0.05, 0.1, 0.25, 0.354, 0.5, 1.0, 2.0],   # 0.354 ~ 1/sqrt(8)
           acts=["linear", "relu"], seeds=[0, 1, 2, 3, 4],
           steps=200, log_every=40, lr=0.02)


def one_run(act, alpha, seed, cfg):
    x, xp = _data(seed, cfg)
    m = ArchMLP(cfg["d_in"], cfg["width"], cfg["L"], "residual", act,
                alpha=alpha, seed=seed)
    mb = measure_blocks(m, x, xp, cfg["tau"])
    init_1mA = 1 - mean_nonfinal(mb, 1)
    init_an = mean_nonfinal(mb, 3)
    for step in range(cfg["steps"]):
        scff_step_arch(m, x, xp, cfg["tau"], cfg["lr"])
    final_A = mean_nonfinal(measure_blocks(m, x, xp, cfg["tau"]), 1)
    return init_1mA, init_an, final_A


def main():
    os.makedirs(RUNS, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    rows = []
    for act in CFG["acts"]:
        for alpha in CFG["alphas"]:
            vals = np.array([one_run(act, alpha, s, CFG) for s in CFG["seeds"]])
            mu = vals.mean(0)
            rows.append(dict(act=act, alpha=alpha,
                             init_1mA=mu[0], init_aniso=mu[1], final_A=mu[2],
                             final_A_std=float(vals[:, 2].std())))
            print(f"{act:6s} alpha={alpha:<5g}  init 1-A={mu[0]:.3f}  "
                  f"Aniso={mu[1]:.3f}  final A={mu[2]:.3f}", flush=True)

    with open(os.path.join(RUNS, "residual_alpha.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(RUNS, "residual_alpha.yaml"), "w") as f:
        yaml.safe_dump(CFG, f)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for ax, key, ylab in [(axes[0], "init_1mA", r"$1-A$ (init)"),
                          (axes[1], "init_aniso", "Aniso (init)"),
                          (axes[2], "final_A", "A after training")]:
        for act in CFG["acts"]:
            xs = CFG["alphas"]
            ys = [r[key] for r in rows if r["act"] == act]
            ax.plot(xs, ys, marker="o", label=act)
        ax.set_xscale("log"); ax.set_xlabel(r"residual scale $\alpha$"); ax.set_ylabel(ylab)
        ax.axvline(1 / np.sqrt(CFG["L"]), color="gray", ls="--", lw=0.7)
        ax.legend()
    fig.suptitle(r"Residual scale sweep (L=8): small $\alpha$ -> $M\approx I$ -> alignment "
                 r"holds & persists (dashed = $1/\sqrt{L}$)")
    fig.tight_layout()
    path = os.path.join(PLOTS, "residual_alpha.png")
    fig.savefig(path, dpi=130); print("plot ->", path)

    print("\nReads: small alpha -> low 1-A, low Aniso, high final A (M~=I).")
    print("alpha -> large recovers plain-like behavior. ReLU vs linear: compare curves.")


if __name__ == "__main__":
    main()
