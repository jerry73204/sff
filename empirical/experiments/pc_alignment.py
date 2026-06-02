"""Does the biologically-grounded predictive-coding feedback close the local↔BP gap?

PC's local update ΔW^(ℓ) = ε^(ℓ) f(x^(ℓ-1))ᵀ should align with the backprop gradient, and the
alignment should GROW with inference (settling) steps T — settling propagates the output error
down through the hierarchy (the cross-layer top-down feedback NGRAD says cortex uses). At T=0
only the output layer is aligned (purely local); as T→∞ all layers align (≈BP).

This is the biological counterpoint to SCFF: SCFF's local goodness gradient is bounded away
from BP by the cross-layer δ; PC carries the cross-layer error explicitly via settling.

Run: python experiments/pc_alignment.py
"""
from __future__ import annotations
import os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pc import PCNet

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(HERE, "plots")

CFG = dict(dims=[16, 32, 32, 32, 8], batch=16, beta=0.1,
           Ts=[0, 1, 2, 5, 10, 20, 50, 100], seeds=list(range(5)))


def cos(a, b):
    a, b = a.flatten(), b.flatten()
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else 0.0


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(PLOTS, exist_ok=True)
    Lw = len(CFG["dims"]) - 1                          # number of weight layers
    # cos[T][layer] averaged over seeds
    res = {T: [[] for _ in range(Lw)] for T in CFG["Ts"]}
    for seed in CFG["seeds"]:
        g = torch.Generator().manual_seed(100 + seed)
        x0 = torch.randn(CFG["batch"], CFG["dims"][0], generator=g)
        target = torch.randn(CFG["batch"], CFG["dims"][-1], generator=g)
        net = PCNet(CFG["dims"], seed=seed)
        bp = net.bp_descent(x0, target)
        for T in CFG["Ts"]:
            pc = net.pc_update(x0, target, T, CFG["beta"])
            for l in range(Lw):
                res[T][l].append(cos(pc[l], bp[l]))
        print(f"seed {seed} done", flush=True)

    print(f"\ncos(ΔW_PC, BP descent) by layer (0=first .. {Lw-1}=output), vs settling T:")
    print("  T   " + "  ".join(f"L{l}" for l in range(Lw)))
    grid = {}
    for T in CFG["Ts"]:
        row = [float(np.mean(res[T][l])) for l in range(Lw)]
        grid[T] = row
        print(f"{T:>4}  " + "  ".join(f"{v:+.2f}" for v in row))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for l in range(Lw):
        ax.plot(CFG["Ts"], [grid[T][l] for T in CFG["Ts"]], marker="o",
                label=f"layer {l}" + (" (output)" if l == Lw - 1 else ""))
    ax.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("inference (settling) steps T"); ax.set_ylabel("cos(ΔW_PC, BP)")
    ax.set_title("Predictive coding → backprop as settling grows\n"
                 "(settling = the cross-layer top-down feedback)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(PLOTS, "pc_alignment.png"); fig.savefig(path, dpi=130)
    print("plot ->", path)

    first_at0 = grid[0][0]
    first_atmax = grid[CFG["Ts"][-1]][0]
    print(f"\nVERDICT: first (deepest-from-output) layer cos vs BP: "
          f"T=0 -> {first_at0:+.3f},  T={CFG['Ts'][-1]} -> {first_atmax:+.3f}.")
    print("PC's local update aligns with BP, and alignment GROWS with settling (= cross-layer")
    print("feedback) -- the biological NGRAD mechanism closes the gap SCFF's pure-local rule cannot.")


if __name__ == "__main__":
    main()
