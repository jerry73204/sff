"""Train SCFF with a BP-free forward-gradient-on-subspace global correction (direction 4).

Update each layer with a blend of the (exact, cheap) LOCAL goodness gradient and the
forward-mode, subspace-restricted estimate of the GLOBAL goodness gradient grad_{W^l} g^(L):

    dW^l = lr [ (1-beta) grad g^(l)_local  +  beta * ghat_VU ]      (ghat_VU is BP-free)

Synthetic data has no classification task, so value is measured by the GLOBAL self-supervised
objective itself: final-layer goodness g^(L) (how well the top reps separate pos/neg), plus
alignment A persistence. Hypothesis: beta>0 raises g^(L) and keeps A higher than pure SCFF
(beta=0), because it BP-free-optimizes toward the global objective.

Run: python experiments/fg_hybrid_train.py
"""
from __future__ import annotations
import os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP
from gradients import local_goodness, global_grad, alignment_cosine, signal
from model import normalize
import arch as A
import metrics
from experiments.fg_subspace import goodness_fn, forward_grad

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(HERE, "plots")

CFG = dict(d_in=64, width=256, n_layers=4, act="linear", tau=0.5, batch=8,
           alpha=0.1, lr=0.02, steps=150, log_every=15, K=32,
           betas=[0.0, 0.5, 1.0], seeds=[0, 1, 2], aug_noise=0.1)


def fg_vu_estimate(model, x, xp, layer, tau, K):
    """BP-free estimate of grad_{W[layer]} g^(L), forward-mode, restricted to
    span(V) (x) span(y_prev)."""
    f = goodness_fn(model, x, xp, layer, tau)
    W0 = model.W[layer]
    ys, ysp = model(x), model(xp)
    s, _ = signal(normalize(ys[layer + 1]), normalize(ysp[layer + 1]).detach(), tau)
    V, dV = metrics.contrastive_subspace(s)
    Ss = torch.linalg.svdvals(ys[layer])
    _, _, Uh = torch.linalg.svd(ys[layer], full_matrices=False)
    dU = int((Ss > 1e-3 * Ss[0]).sum().item())
    U = Uh[:dU].t().contiguous()
    gVU, _ = forward_grad(f, W0, lambda: V @ torch.randn(dV, dU) @ U.t(), K)
    return gVU


def hybrid_step(model, x, xp, tau, lr, beta, K):
    grads = []
    for l in range(model.n_layers):
        gl = A.local_grad(model, x, xp, l, tau)
        if beta > 0:
            gl = (1 - beta) * gl + beta * fg_vu_estimate(model, x, xp, l, tau, K)
        grads.append(gl)
    with torch.no_grad():
        for l in range(model.n_layers):
            model.W[l].add_(lr * grads[l])


def measure(model, x, xp, tau):
    ys, ysp = model(x), model(xp)
    gL = local_goodness(normalize(ys[-1]), normalize(ysp[-1]).detach(), tau).item()
    Avals = []
    for l in range(model.n_layers - 1):
        gl = A.local_grad(model, x, xp, l, tau)
        gg = global_grad(model, x, xp, l, tau)
        Avals.append(alignment_cosine(gl, gg))
    return sum(Avals) / len(Avals), gL


def run(beta, seed, cfg):
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(cfg["batch"], cfg["d_in"], generator=g)
    xp = x + cfg["aug_noise"] * torch.randn(x.shape, generator=g)
    model = ArchMLP(cfg["d_in"], cfg["width"], cfg["n_layers"], "residual",
                    cfg["act"], alpha=cfg["alpha"], seed=seed)
    A_t, gL_t, steps = [], [], []
    for step in range(cfg["steps"] + 1):
        if step % cfg["log_every"] == 0:
            a, gl = measure(model, x, xp, cfg["tau"])
            A_t.append(a); gL_t.append(gl); steps.append(step)
        hybrid_step(model, x, xp, cfg["tau"], cfg["lr"], beta, cfg["K"])
    return steps, A_t, gL_t


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(PLOTS, exist_ok=True)
    data = {}
    for beta in CFG["betas"]:
        As, gLs = [], []
        for seed in CFG["seeds"]:
            steps, A_t, gL_t = run(beta, seed, CFG)
            As.append(A_t); gLs.append(gL_t)
        data[beta] = dict(steps=steps, A=np.array(As), gL=np.array(gLs))
        a0, aN = data[beta]["A"][:, 0].mean(), data[beta]["A"][:, -1].mean()
        g0, gN = data[beta]["gL"][:, 0].mean(), data[beta]["gL"][:, -1].mean()
        print(f"beta={beta:<4}  A {a0:.3f}->{aN:.3f}   g^(L) {g0:.3f}->{gN:.3f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for beta in CFG["betas"]:
        st = data[beta]["steps"]
        for ax, key in [(axes[0], "A"), (axes[1], "gL")]:
            mu, sd = data[beta][key].mean(0), data[beta][key].std(0)
            ax.plot(st, mu, marker="o", label=f"β={beta}")
            ax.fill_between(st, mu - sd, mu + sd, alpha=0.15)
    axes[0].axhline(1.0, color="gray", lw=0.6, ls="--")
    axes[0].set_xlabel("step"); axes[0].set_ylabel("A (mean non-final)"); axes[0].legend()
    axes[0].set_title("Alignment persistence")
    axes[1].set_xlabel("step"); axes[1].set_ylabel(r"final goodness $g^{(L)}$"); axes[1].legend()
    axes[1].set_title("Global objective (BP-free correction helps?)")
    fig.suptitle("Hybrid SCFF + forward-grad-on-subspace global correction (residual, BP-free)")
    fig.tight_layout()
    path = os.path.join(PLOTS, "fg_hybrid.png"); fig.savefig(path, dpi=130); print("plot ->", path)

    g_pure = data[0.0]["gL"][:, -1].mean()
    print(f"\nVERDICT: final g^(L)  " +
          "  ".join(f"β{b}={data[b]['gL'][:,-1].mean():.3f}" for b in CFG["betas"]))
    best = max(CFG["betas"], key=lambda b: data[b]["gL"][:, -1].mean())
    print(f"Best global objective at beta={best}. BP-free correction "
          f"{'HELPS' if data[best]['gL'][:,-1].mean() > g_pure + 0.02 else 'does not help'} "
          f"vs pure SCFF (beta=0).")


if __name__ == "__main__":
    main()
