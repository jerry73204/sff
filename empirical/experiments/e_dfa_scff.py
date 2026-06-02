"""Feedback-modulated SCFF (direction B): inject the cross-layer signal the cheap, biological
way — a one-step DIRECT random feedback broadcast (DFA; Lillicrap et al. 2016), not sequential
settling (PC needs K~depth steps) and not a gradient estimate (forward-grad failed).

Each layer's update blends its local goodness gradient with a top-down teaching signal:
  dW^l = (1-beta) grad g^(l)_local  +  beta * (1/tau) Pperp( B^l s^(L) ) (y^(l-1))^T
where s^(L) is the FINAL-layer contrastive signal and B^l a FIXED RANDOM feedback matrix. The
feedback is one matmul, forward-only, layer-parallel — it keeps FF's efficiency. Question: does
it raise the SCFF<->BP alignment A during training (the FA "learn to align" effect), vs pure
SCFF (beta=0)?

Run: python experiments/e_dfa_scff.py
"""
from __future__ import annotations
import os, sys, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP
from model import normalize
from gradients import signal, global_grad, alignment_cosine
import arch as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(HERE, "plots")

CFG = dict(d_in=64, width=256, n_layers=4, act="linear", tau=0.5, batch=8,
           lr=0.02, steps=150, log_every=15, betas=[0.0, 0.5, 1.0],
           seeds=[0, 1, 2], aug_noise=0.1)


def feedback_weights(model, seed):
    g = torch.Generator().manual_seed(7000 + seed)
    nL = model.width
    return [torch.randn(model.width, nL, generator=g) / math.sqrt(nL)
            for _ in range(model.n_layers)]


def dfa_grad(model, B, x, xp, layer, tau):
    """Top-down teaching grad: final signal s^(L) delivered to `layer` via random feedback."""
    ys, ysp = model(x), model(xp)
    sL, _ = signal(normalize(ys[-1]), normalize(ysp[-1]).detach(), tau)   # [B, n]
    fb = sL @ B[layer].t()                                                 # feedback at layer
    z, y = normalize(ys[layer + 1]), ys[layer + 1]
    proj = (fb - z * (z * fb).sum(1, keepdim=True)) / y.norm(dim=1, keepdim=True)
    if model.act_name != "linear":
        proj = proj * (ys[layer] @ model.W[layer].t() > 0).float()
    return (proj.t() @ ys[layer]) / tau


def step(model, B, x, xp, tau, lr, beta):
    ups = []
    for l in range(model.n_layers):
        gl = A.local_grad(model, x, xp, l, tau)
        ups.append((1 - beta) * gl + beta * dfa_grad(model, B, x, xp, l, tau)
                   if beta > 0 else gl)
    with torch.no_grad():
        for l in range(model.n_layers):
            model.W[l].add_(lr * ups[l])


def mean_align(model, B, x, xp, tau, beta):
    """Mean over non-final layers of cos(actual update, BP global gradient)."""
    vals = []
    for l in range(model.n_layers - 1):
        gl = A.local_grad(model, x, xp, l, tau)
        up = (1 - beta) * gl + beta * dfa_grad(model, B, x, xp, l, tau) if beta > 0 else gl
        vals.append(alignment_cosine(up, global_grad(model, x, xp, l, tau)))
    return sum(vals) / len(vals)


def run(beta, seed, cfg):
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(cfg["batch"], cfg["d_in"], generator=g)
    xp = x + cfg["aug_noise"] * torch.randn(x.shape, generator=g)
    model = ArchMLP(cfg["d_in"], cfg["width"], cfg["n_layers"], "plain", cfg["act"], seed=seed)
    B = feedback_weights(model, seed)
    A_t, steps = [], []
    for s in range(cfg["steps"] + 1):
        if s % cfg["log_every"] == 0:
            A_t.append(mean_align(model, B, x, xp, cfg["tau"], beta)); steps.append(s)
        step(model, B, x, xp, cfg["tau"], cfg["lr"], beta)
    return steps, A_t


def main():
    torch.set_default_dtype(torch.float64)
    os.makedirs(PLOTS, exist_ok=True)
    data = {}
    for beta in CFG["betas"]:
        curves = [run(beta, s, CFG)[1] for s in CFG["seeds"]]
        steps = run(beta, CFG["seeds"][0], CFG)[0]
        data[beta] = dict(steps=steps, A=np.array(curves))
        a = data[beta]["A"]
        print(f"beta={beta:<4} (DFA mix)  A {a[:,0].mean():.3f} -> {a[:,-1].mean():.3f}", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for beta in CFG["betas"]:
        st, a = data[beta]["steps"], data[beta]["A"]
        lab = "pure SCFF" if beta == 0 else ("pure DFA" if beta == 1 else f"SCFF+DFA β={beta}")
        ax.plot(st, a.mean(0), marker="o", label=lab)
        ax.fill_between(st, a.mean(0) - a.std(0), a.mean(0) + a.std(0), alpha=0.15)
    ax.axhline(1.0, color="gray", lw=0.6, ls="--")
    ax.set_xlabel("step"); ax.set_ylabel("A (mean non-final, vs BP)")
    ax.set_title("Feedback-modulated SCFF: does cheap direct random feedback align to BP?")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOTS, "dfa_scff.png"); fig.savefig(path, dpi=130); print("plot ->", path)

    base = data[0.0]["A"][:, -1].mean()
    print(f"\nVERDICT: final A  " +
          "  ".join(f"β{b}={data[b]['A'][:,-1].mean():.3f}" for b in CFG["betas"]))
    best = max(CFG["betas"], key=lambda b: data[b]["A"][:, -1].mean())
    print(f"Direct random feedback {'HELPS' if data[best]['A'][:,-1].mean() > base + 0.05 else 'does NOT help'}"
          f" alignment vs pure SCFF.")


if __name__ == "__main__":
    main()
