"""E2 -- training dynamics of gradient alignment (design.md 2.2).

Train SCFF; track A^(l), Delta_Gram^(l), Aniso^(l), d_V vs step, per layer, multi-seed.
Prediction: A^(l) stays near 1 because Delta_Gram falls even as Aniso rises. The headline
plot overlays all three. A negative result is still a finding -- instrument honestly.

Run:  python experiments/e2_dynamics.py
Outputs: runs/e2_<...>.csv, runs/e2_<...>.yaml, plots/e2_<...>.png
"""
from __future__ import annotations
import os, sys, csv, math
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MLP, normalize
from gradients import local_grad, global_grad, alignment_cosine, signal
from scff import scff_step
import metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs")
PLOTS = os.path.join(HERE, "plots")

CFG = dict(
    # valid regime: B << sqrt(n) so d_V = o(sqrt n) (design 2.2/2.5).
    # sqrt(512) ~ 22.6, batch 4 -> d_V <= 4.
    d_in=16, width=512, n_layers=4, act="linear",
    batch=4, tau=0.5, lr=0.02, steps=300, log_every=15,
    aug_noise=0.1, seeds=[0, 1, 2, 3, 4],
)


def measure(model, x, x_pos, tau):
    """Per-layer (A, dgram, aniso, dV) at current weights."""
    ys, ysp = model(x), model(x_pos)
    zL = normalize(ys[-1])
    out = []
    for l in range(model.n_layers):
        gl = local_grad(model, x, x_pos, l, tau)
        gg = global_grad(model, x, x_pos, l, tau)
        A = alignment_cosine(gl, gg)
        z_l = normalize(ys[l + 1])
        zp_l = normalize(ysp[l + 1])
        dgram = metrics.delta_gram(z_l, zL)
        s, _ = signal(z_l, zp_l.detach(), tau)
        V, dV = metrics.contrastive_subspace(s)
        if model.act_name == "linear":
            M = metrics.downstream_jacobian_linear(model, l)
        else:
            M = metrics.downstream_jacobian_relu(model, x, l)
        an = metrics.aniso(M, V)
        out.append(dict(layer=l, A=A, dgram=dgram, aniso=an, dV=dV))
    return out


def run_seed(seed, cfg):
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(cfg["batch"], cfg["d_in"], generator=g)
    x_pos = x + cfg["aug_noise"] * torch.randn(x.shape, generator=g)
    model = MLP(cfg["d_in"], cfg["width"], cfg["n_layers"], cfg["act"], seed=seed)
    rows = []
    for step in range(cfg["steps"] + 1):
        if step % cfg["log_every"] == 0:
            for m in measure(model, x, x_pos, cfg["tau"]):
                rows.append(dict(seed=seed, step=step, **m))
        scff_step(model, x, x_pos, cfg["tau"], cfg["lr"])
    return rows


def aggregate(rows, cfg):
    """mean +/- std over seeds, keyed by (step, layer)."""
    import numpy as np
    keys = sorted({(r["step"], r["layer"]) for r in rows})
    agg = {}
    for (step, layer) in keys:
        sub = [r for r in rows if r["step"] == step and r["layer"] == layer]
        agg[(step, layer)] = {
            k + "_mean": float(np.mean([r[k] for r in sub]))
            for k in ("A", "dgram", "aniso", "dV")
        } | {
            k + "_std": float(np.std([r[k] for r in sub]))
            for k in ("A", "dgram", "aniso", "dV")
        }
    return agg


def plot(agg, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    L = cfg["n_layers"]
    steps = sorted({s for (s, _) in agg})
    fig, axes = plt.subplots(1, L, figsize=(4 * L, 3.4), sharex=True)
    if L == 1:
        axes = [axes]
    for l in range(L):
        ax = axes[l]
        for key, color, lab in [("A", "C0", "A (align)"),
                                ("dgram", "C1", r"$\Delta_{Gram}$"),
                                ("aniso", "C2", "Aniso")]:
            mu = np.array([agg[(s, l)][key + "_mean"] for s in steps])
            sd = np.array([agg[(s, l)][key + "_std"] for s in steps])
            ax.plot(steps, mu, color=color, label=lab)
            ax.fill_between(steps, mu - sd, mu + sd, color=color, alpha=0.2)
        ax.axhline(1.0, color="gray", lw=0.6, ls="--")
        ax.set_title(f"layer {l}")
        ax.set_xlabel("step")
        if l == 0:
            ax.legend(fontsize=8)
    fig.suptitle(f"E2 dynamics  (n={cfg['width']}, L={L}, {cfg['act']}, "
                 f"{len(cfg['seeds'])} seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("plot ->", path)


def verdict(agg, cfg) -> str:
    steps = sorted({s for (s, _) in agg})
    s0, s1 = steps[0], steps[-1]
    lines = ["VERDICT (per layer, init -> final):"]
    held = True
    for l in range(cfg["n_layers"]):
        A0, A1 = agg[(s0, l)]["A_mean"], agg[(s1, l)]["A_mean"]
        dg0, dg1 = agg[(s0, l)]["dgram_mean"], agg[(s1, l)]["dgram_mean"]
        an0, an1 = agg[(s0, l)]["aniso_mean"], agg[(s1, l)]["aniso_mean"]
        ok = A1 > 0.9
        held = held and ok
        lines.append(
            f"  layer {l}: A {A0:.3f}->{A1:.3f} | dGram {dg0:.3f}->{dg1:.3f} | "
            f"Aniso {an0:.3f}->{an1:.3f}  [{'HELD' if ok else 'BROKE'}]")
    lines.append(f"Hypothesis (A stays near 1 as dGram falls vs Aniso rises): "
                 f"{'SUPPORTED' if held else 'NOT supported in all layers'}")
    return "\n".join(lines)


def main():
    os.makedirs(RUNS, exist_ok=True)
    os.makedirs(PLOTS, exist_ok=True)
    tag = f"e2_n{CFG['width']}_L{CFG['n_layers']}_{CFG['act']}"
    rows = []
    for seed in CFG["seeds"]:
        print(f"seed {seed} ...", flush=True)
        rows += run_seed(seed, CFG)
    # raw CSV
    csv_path = os.path.join(RUNS, tag + ".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "step", "layer", "A", "dgram", "aniso", "dV"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RUNS, tag + ".yaml"), "w") as f:
        yaml.safe_dump(CFG, f)
    agg = aggregate(rows, CFG)
    plot(agg, CFG, os.path.join(PLOTS, tag + ".png"))
    v = verdict(agg, CFG)
    print(v)
    with open(os.path.join(RUNS, tag + "_verdict.txt"), "w") as f:
        f.write(v + "\n")


if __name__ == "__main__":
    main()
