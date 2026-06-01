"""Fisher probe: does NGD-FF (K-FAC local natural gradient) keep alignment alive?

E2/probe found vanilla SCFF loses alignment via dynamical anisotropy of the downstream
Jacobian on V. Natural-gradient preconditioning exists to cancel anisotropy, so this
compares vanilla SCFF vs Fisher-SCFF on the same init: does Aniso stay bounded and A persist?

Run: python experiments/e2_fisher.py
Outputs: plots/e2_fisher_<...>.png
"""
from __future__ import annotations
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MLP, normalize
from gradients import local_grad, global_grad, alignment_cosine, signal
from scff import scff_step, scff_fisher_step
import metrics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(HERE, "plots")

CFG = dict(d_in=16, width=512, n_layers=4, batch=4, tau=0.5,
           steps=300, log_every=15, aug_noise=0.1, lr=0.02, damp=1e-2,
           seeds=[0, 1, 2, 3, 4])


def measure(model, x, x_pos, tau):
    ys, ysp = model(x), model(x_pos)
    zL = normalize(ys[-1])
    A, an = [], []
    for l in range(model.n_layers):
        gl = local_grad(model, x, x_pos, l, tau)
        gg = global_grad(model, x, x_pos, l, tau)
        A.append(alignment_cosine(gl, gg))
        z_l, zp_l = normalize(ys[l + 1]), normalize(ysp[l + 1])
        s, _ = signal(z_l, zp_l.detach(), tau)
        V, _ = metrics.contrastive_subspace(s)
        M = metrics.downstream_jacobian_linear(model, l)
        an.append(metrics.aniso(M, V))
    # mean over non-final layers (final is trivially aligned)
    nf = model.n_layers - 1
    return sum(A[:nf]) / nf, sum(an[:nf]) / nf


def run(method, seed, cfg):
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(cfg["batch"], cfg["d_in"], generator=g)
    x_pos = x + cfg["aug_noise"] * torch.randn(x.shape, generator=g)
    model = MLP(cfg["d_in"], cfg["width"], cfg["n_layers"], "linear", seed=seed)
    A_t, an_t, steps = [], [], []
    for step in range(cfg["steps"] + 1):
        if step % cfg["log_every"] == 0:
            A, an = measure(model, x, x_pos, cfg["tau"])
            A_t.append(A); an_t.append(an); steps.append(step)
        if method == "vanilla":
            scff_step(model, x, x_pos, cfg["tau"], cfg["lr"])
        else:
            scff_fisher_step(model, x, x_pos, cfg["tau"], cfg["lr"], cfg["damp"])
    return steps, A_t, an_t


def main():
    import numpy as np
    os.makedirs(PLOTS, exist_ok=True)
    data = {}
    for method in ("vanilla", "fisher"):
        As, ans = [], []
        for seed in CFG["seeds"]:
            steps, A_t, an_t = run(method, seed, CFG)
            As.append(A_t); ans.append(an_t)
        data[method] = dict(steps=steps,
                            A=np.array(As), an=np.array(ans))
        Am, AN = data[method]["A"][:, 0].mean(), data[method]["A"][:, -1].mean()
        anm, anN = data[method]["an"][:, 0].mean(), data[method]["an"][:, -1].mean()
        print(f"[{method:7s}] A {Am:.3f}->{AN:.3f}   Aniso {anm:.3f}->{anN:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for method, color in [("vanilla", "C3"), ("fisher", "C0")]:
        st = data[method]["steps"]
        for ax, key, ylab in [(axes[0], "A", "A (alignment)"),
                              (axes[1], "an", "Aniso")]:
            mu = data[method][key].mean(0)
            sd = data[method][key].std(0)
            ax.plot(st, mu, color=color, label=method)
            ax.fill_between(st, mu - sd, mu + sd, color=color, alpha=0.2)
            ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.legend(fontsize=9)
    axes[0].axhline(1.0, color="gray", lw=0.6, ls="--")
    fig.suptitle(f"Fisher (NGD-FF) vs vanilla SCFF  (n={CFG['width']}, "
                 f"{len(CFG['seeds'])} seeds, mean+/-std over non-final layers)")
    fig.tight_layout()
    path = os.path.join(PLOTS, f"e2_fisher_n{CFG['width']}.png")
    fig.savefig(path, dpi=130); print("plot ->", path)

    Av = data["vanilla"]["A"][:, -1].mean()
    Af = data["fisher"]["A"][:, -1].mean()
    print(f"\nVERDICT: final A  vanilla={Av:.3f}  fisher={Af:.3f}  "
          f"-> Fisher {'HELPS' if Af > Av + 0.05 else 'does NOT help'} persistence")


if __name__ == "__main__":
    main()
