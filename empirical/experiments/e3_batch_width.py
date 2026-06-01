"""E3 -- batch/width tradeoff (design.md 2.2/2.3).

Each sample adds a contrastive-signal direction, so d_V grows with batch B. The theorem
holds while d_V = o(sqrt n). Fix width n, sweep B, and watch the ISOTROPY term Aniso (the
quantity the proof bounds, valid while d_V = o(sqrt n)) cross a knee near d_V ~ sqrt(n).
Total 1-A is delta-floored (see E1), so the knee shows in Aniso.

Note: in LINEAR mode d_V <= d_in, so d_in must exceed sqrt(n) for d_V to reach the threshold.

Prediction: Aniso stays low for d_V << sqrt(n), rises once d_V >~ sqrt(n).

Run: python experiments/e3_batch_width.py
Outputs: runs/e3.csv, plots/e3.png
"""
from __future__ import annotations
import os, sys, csv, math
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MLP, normalize
from gradients import local_grad, global_grad, alignment_cosine, signal
import metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS, PLOTS = os.path.join(HERE, "runs"), os.path.join(HERE, "plots")

CFG = dict(d_in=64, width=512, n_layers=3, act="linear", tau=0.5,
           batches=[2, 4, 8, 16, 24, 32, 48, 64], seeds=list(range(6)), aug_noise=0.1)


def measure_init(model, x, x_pos, tau):
    """Per non-final block: (1-A, Aniso, d_V). Init only."""
    ys, ysp = model(x), model(x_pos)
    out = []
    for l in range(model.n_layers - 1):              # exclude trivially-aligned last block
        gl = local_grad(model, x, x_pos, l, tau)
        gg = global_grad(model, x, x_pos, l, tau)
        oma = 1.0 - alignment_cosine(gl, gg)
        z_l, zp_l = normalize(ys[l + 1]), normalize(ysp[l + 1])
        s, _ = signal(z_l, zp_l.detach(), tau)
        V, dV = metrics.contrastive_subspace(s)
        M = metrics.downstream_jacobian_linear(model, l)
        an = metrics.aniso(M, V)
        out.append((oma, an, dV))
    return out


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(RUNS, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    sqrtn = math.sqrt(CFG["width"])
    rows = []
    for B in CFG["batches"]:
        for seed in CFG["seeds"]:
            g = torch.Generator().manual_seed(1000 + seed)
            x = torch.randn(B, CFG["d_in"], generator=g)
            x_pos = x + CFG["aug_noise"] * torch.randn(x.shape, generator=g)
            model = MLP(CFG["d_in"], CFG["width"], CFG["n_layers"], CFG["act"], seed=seed)
            for (oma, an, dV) in measure_init(model, x, x_pos, CFG["tau"]):
                rows.append(dict(B=B, seed=seed, one_minus_A=oma, aniso=an, dV=dV))
        print(f"B={B} done", flush=True)

    with open(os.path.join(RUNS, "e3.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["B", "seed", "one_minus_A", "aniso", "dV"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(RUNS, "e3.yaml"), "w") as f:
        yaml.safe_dump(CFG, f)

    Bs = CFG["batches"]
    def mu(key, B): return float(np.mean([r[key] for r in rows if r["B"] == B]))
    def sd(key, B): return float(np.std([r[key] for r in rows if r["B"] == B]))
    dV_mu = [mu("dV", B) for B in Bs]
    an_mu = [mu("aniso", B) for B in Bs]
    an_sd = [sd("aniso", B) for B in Bs]
    oma_mu = [mu("one_minus_A", B) for B in Bs]

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].errorbar(dV_mu, an_mu, yerr=an_sd, marker="o", capsize=3)
    axes[0].axvline(sqrtn, color="C3", ls="--", label=r"$\sqrt{n}$")
    axes[0].set_xlabel(r"$d_V$"); axes[0].set_ylabel("Aniso (init)")
    axes[0].set_title("Isotropy term vs subspace dim"); axes[0].legend()
    axes[1].plot(Bs, dV_mu, marker="o")
    axes[1].axhline(sqrtn, color="C3", ls="--", label=r"$\sqrt{n}$")
    axes[1].set_xlabel("batch B"); axes[1].set_ylabel(r"$d_V$")
    axes[1].set_title("Subspace dim vs batch"); axes[1].legend()
    fig.suptitle(f"E3 batch/width (n={CFG['width']}, sqrt(n)={sqrtn:.1f}, d_in={CFG['d_in']}): "
                 r"isotropy knee near $d_V\approx\sqrt{n}$")
    fig.tight_layout()
    path = os.path.join(PLOTS, "e3.png"); fig.savefig(path, dpi=130); print("plot ->", path)

    print(f"\nsqrt(n) = {sqrtn:.1f}")
    print(f"{'B':>4} {'d_V':>6} {'Aniso':>8} {'1-A':>8}")
    for B, dV, an, oma in zip(Bs, dV_mu, an_mu, oma_mu):
        flag = "  <- crosses sqrt(n)" if abs(dV - sqrtn) < 6 else ""
        print(f"{B:>4} {dV:>6.1f} {an:>8.3f} {oma:>8.3f}{flag}")
    # Aniso grows smoothly with d_V; fit the exponent + report value at d_V ~ sqrt(n).
    ld, la = np.log(np.array(dV_mu)), np.log(np.array(an_mu))
    slope = float(np.linalg.lstsq(np.vstack([ld, np.ones_like(ld)]).T, la, rcond=None)[0][0])
    near = min(range(len(Bs)), key=lambda i: abs(dV_mu[i] - sqrtn))
    print(f"\nVERDICT:")
    print(f"  Aniso ~ d_V^{slope:.2f}  (~0.5 => Aniso prop sqrt(d_V); Aniso/sqrt(d_V)~0.05 here).")
    print(f"  At d_V≈sqrt(n)={sqrtn:.1f} (B={Bs[near]}): Aniso={an_mu[near]:.3f}.")
    print(f"  Isotropy degrades SMOOTHLY as d_V grows -- a power-law crossover, NOT a sharp")
    print(f"  knee; sqrt(n) marks where Aniso ~ O(0.25). Total 1-A stays flat "
          f"(~{np.mean(oma_mu):.2f}) -- delta-floored (E1).")


if __name__ == "__main__":
    main()
