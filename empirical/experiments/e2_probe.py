"""Probe E2: WHY does alignment degrade under SCFF training?

E2 showed A^(l) falls during linear training (Aniso rises faster than Delta_Gram falls).
This probes the cause across knobs and tracks extra diagnostics, to tell apart:
  - instability (weights / rep norms drift) vs genuine dynamical anisotropy,
  - learning-rate dependence,
  - linear vs ReLU,
  - whether the contrastive subspace d_V grows during training.

Run: python experiments/e2_probe.py
"""
from __future__ import annotations
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MLP, normalize
from gradients import local_grad, global_grad, alignment_cosine, signal
from scff import scff_step
import metrics

BASE = dict(d_in=16, width=512, n_layers=4, batch=4, tau=0.5,
            steps=300, log_every=50, aug_noise=0.1)


def diag(model, x, x_pos, tau):
    """Per-layer (A, dgram, aniso, dV) + global weight/rep norms."""
    ys, ysp = model(x), model(x_pos)
    zL = normalize(ys[-1])
    rows = []
    for l in range(model.n_layers):
        gl = local_grad(model, x, x_pos, l, tau)
        gg = global_grad(model, x, x_pos, l, tau)
        A = alignment_cosine(gl, gg)
        z_l, zp_l = normalize(ys[l + 1]), normalize(ysp[l + 1])
        dg = metrics.delta_gram(z_l, zL)
        s, _ = signal(z_l, zp_l.detach(), tau)
        V, dV = metrics.contrastive_subspace(s)
        if model.act_name == "linear":
            M = metrics.downstream_jacobian_linear(model, l)
        else:
            M = metrics.downstream_jacobian_relu(model, x, l)
        an = metrics.aniso(M, V)
        rows.append((l, A, dg, an, dV))
    wnorm = max(w.norm().item() for w in model.W)
    ynorm = ys[-1].norm(dim=1).mean().item()
    return rows, wnorm, ynorm


def run_variant(act, lr, seed=0, cfg=BASE):
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(cfg["batch"], cfg["d_in"], generator=g)
    x_pos = x + cfg["aug_noise"] * torch.randn(x.shape, generator=g)
    model = MLP(cfg["d_in"], cfg["width"], cfg["n_layers"], act, seed=seed)
    traj = {}
    for step in range(cfg["steps"] + 1):
        if step % cfg["log_every"] == 0:
            rows, wn, yn = diag(model, x, x_pos, cfg["tau"])
            traj[step] = dict(rows=rows, wnorm=wn, ynorm=yn)
        scff_step(model, x, x_pos, cfg["tau"], lr)
    return traj


def summarize(act, lr, traj):
    s0, sN = min(traj), max(traj)
    # layer-0 (most downstream) is the hardest; report it + averages
    def layer_vals(step, idx):
        return [r[idx] for r in traj[step]["rows"]]
    A0 = [r for r in traj[s0]["rows"]]
    AN = [r for r in traj[sN]["rows"]]
    meanA0 = sum(r[1] for r in A0[:-1]) / max(1, len(A0) - 1)   # exclude last (trivial)
    meanAN = sum(r[1] for r in AN[:-1]) / max(1, len(AN) - 1)
    an0 = sum(r[3] for r in A0[:-1]) / max(1, len(A0) - 1)
    anN = sum(r[3] for r in AN[:-1]) / max(1, len(AN) - 1)
    dg0 = sum(r[2] for r in A0[:-1]) / max(1, len(A0) - 1)
    dgN = sum(r[2] for r in AN[:-1]) / max(1, len(AN) - 1)
    dv0 = sum(r[4] for r in A0[:-1]) / max(1, len(A0) - 1)
    dvN = sum(r[4] for r in AN[:-1]) / max(1, len(AN) - 1)
    print(f"\n[{act:6s} lr={lr:<6g}]  (mean over non-final layers)")
    print(f"   A     {meanA0:.3f} -> {meanAN:.3f}")
    print(f"   Aniso {an0:.3f} -> {anN:.3f}      dGram {dg0:.3f} -> {dgN:.3f}")
    print(f"   d_V   {dv0:.2f} -> {dvN:.2f}")
    print(f"   |W|max {traj[s0]['wnorm']:.2f} -> {traj[sN]['wnorm']:.2f}   "
          f"|y_L|mean {traj[s0]['ynorm']:.2e} -> {traj[sN]['ynorm']:.2e}")


def main():
    print("=" * 64)
    print("E2 PROBE  (width=512, batch=4, L=4)")
    for act in ("linear", "relu"):
        for lr in (0.005, 0.02, 0.08):
            traj = run_variant(act, lr)
            summarize(act, lr, traj)
    print("\n" + "=" * 64)
    print("Read: if |W| / |y_L| blow up -> instability (needs norm/decay).")
    print("If Aniso rises while |W| stable -> genuine dynamical anisotropy.")
    print("If d_V grows past sqrt(n)~22.6 -> subspace outgrows width (E3 regime).")


if __name__ == "__main__":
    main()
