"""Dendritic-microcircuit feedback test (direction A).

Dendritic credit assignment (Sacramento et al. 2018) carries the teaching signal in the APICAL
dendrite via SEPARATE feedback weights — distinct from the forward (basal) weights, and learned
by interneuron plasticity to mirror the forward path. This is the NGRAD/predictive-coding family
but with non-symmetric feedback.

Question: how well must the feedback weights mirror the forward weights for the apical-error
update to recover backprop? We run the PC settling with feedback weights interpolated from
RANDOM (feedback-alignment / DFA-like) to SYMMETRIC (= forward weights, exact). At each mix we
measure cos(ΔW_dendritic, BP gradient).

Reconciles the two earlier results: fixed-random direct feedback FAILED (DFA experiment);
sequential symmetric feedback RECOVERS BP (PC experiment). This sweeps between them.

Run: python experiments/e_dendritic.py
"""
from __future__ import annotations
import os, sys, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pc import PCNet

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(HERE, "plots")

CFG = dict(dims=[16, 32, 32, 32, 8], batch=16, beta=0.1, T=6,
           mixes=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0], seeds=list(range(5)))


def cos(a, b):
    a, b = a.flatten(), b.flatten()
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else 0.0


def random_like(W, seed):
    g = torch.Generator().manual_seed(9000 + seed)
    return [torch.randn(w.shape, generator=g) / math.sqrt(w.shape[1]) for w in W]


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(PLOTS, exist_ok=True)
    Lw = len(CFG["dims"]) - 1
    res = {m: [] for m in CFG["mixes"]}               # mean cos over hidden layers
    for seed in CFG["seeds"]:
        g = torch.Generator().manual_seed(100 + seed)
        x0 = torch.randn(CFG["batch"], CFG["dims"][0], generator=g)
        target = torch.randn(CFG["batch"], CFG["dims"][-1], generator=g)
        net = PCNet(CFG["dims"], seed=seed)
        bp = net.bp_descent(x0, target)
        Brand = random_like(net.W, seed)
        for mix in CFG["mixes"]:                       # mix=1 -> symmetric, 0 -> random
            Bfb = [mix * w + (1 - mix) * r for w, r in zip(net.W, Brand)]
            pc = net.pc_update_fb(x0, target, Bfb, CFG["T"], CFG["beta"])
            # alignment averaged over the non-output (hidden-fed) layers
            res[mix].append(np.mean([cos(pc[l], bp[l]) for l in range(Lw - 1)]))
        print(f"seed {seed} done", flush=True)

    print(f"\nfeedback mirror (0=random .. 1=symmetric=forward weights) vs cos(ΔW, BP):")
    means = {}
    for m in CFG["mixes"]:
        means[m] = float(np.mean(res[m]))
        print(f"  mix={m:<5} cos = {means[m]:+.3f} ± {np.std(res[m]):.3f}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.errorbar(CFG["mixes"], [means[m] for m in CFG["mixes"]],
                yerr=[np.std(res[m]) for m in CFG["mixes"]], marker="o", capsize=3)
    ax.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("feedback ↔ forward mirror (0 = random / DFA, 1 = symmetric)")
    ax.set_ylabel("cos(ΔW_dendritic, BP)")
    ax.set_title("Dendritic apical-error alignment vs feedback-weight mirroring\n"
                 "(random feedback fails; alignment needs feedback ≈ forward path)")
    fig.tight_layout()
    path = os.path.join(PLOTS, "dendritic.png"); fig.savefig(path, dpi=130); print("plot ->", path)

    print(f"\nVERDICT: cos rises {means[0.0]:+.3f} (random) -> {means[1.0]:+.3f} (symmetric).")
    print("The dendritic apical-error recovers BP only when the feedback weights MIRROR the")
    print("forward path -- random feedback (DFA) does not. So the dendritic interneuron-learning")
    print("of feedback weights is the load-bearing ingredient; it is the cost biology pays.")


if __name__ == "__main__":
    main()
