"""Forward-gradient-on-subspace estimator test (direction 4).

Estimate the GLOBAL gradient grad_{W^l} g^(L) BP-free via forward-mode AD (JVP, no backward),
with tangents restricted to the contrastive subspace V (dim d_V). Tests at init:
  H1 (alignment): does the gradient live in V? massInV = ||P_V g|| / ||g||; and does the
                  V-restricted estimate match P_V g_true?
  H2 (variance):  Var(V-restricted) << Var(full-space), ratio ~ n/d_V.

Key question: V is the LOCAL signal subspace, but g_true lives in the GLOBAL signal subspace
(differ by the delta/cross-layer mismatch). PREDICTION: residual (M ~= I) makes local V ~=
global subspace, so massInV jumps and forward-grad-on-V becomes viable.

Run: python experiments/fg_subspace.py
"""
from __future__ import annotations
import os, sys
import torch
from torch.func import functional_call, jvp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MLP, normalize
from arch import ArchMLP
from gradients import local_goodness, global_grad, signal
import metrics

CFG = dict(d_in=64, width=512, n_layers=3, act="linear", tau=0.5, batch=8,
           seeds=[0, 1, 2], K=256, aug_noise=0.1, alpha=0.1)


def cos(a, b):
    a, b = a.flatten(), b.flatten()
    d = a.norm() * b.norm()
    return (a @ b / d).item() if d > 0 else 0.0


def goodness_fn(model, x, xp, layer, tau):
    """g^(L) as a function of W[layer] only, via functional_call (arch-agnostic)."""
    params = dict(model.named_parameters())
    name = f"W.{layer}"
    def f(Wl):
        p = {**params, name: Wl}
        ys = functional_call(model, p, (x,))
        ysp = functional_call(model, p, (xp,))
        return local_goodness(normalize(ys[-1]), normalize(ysp[-1]).detach(), tau)
    return f


def forward_grad(f, W0, sampler, K):
    est = torch.zeros_like(W0); e2 = 0.0
    for _ in range(K):
        T = sampler()
        _, D = jvp(f, (W0,), (T,))
        s = D * T
        est = est + s
        e2 = e2 + float(s.pow(2).sum())
    est = est / K
    var = e2 / K - float(est.pow(2).sum())     # single-sample Frobenius variance
    return est, var


def run_layer(model, x, xp, layer, tau, K):
    g_true = global_grad(model, x, xp, layer, tau)
    f = goodness_fn(model, x, xp, layer, tau)
    W0 = model.W[layer]
    ys, ysp = model(x), model(xp)
    s, _ = signal(normalize(ys[layer + 1]), normalize(ysp[layer + 1]).detach(), tau)
    V, dV = metrics.contrastive_subspace(s)
    n_in = W0.shape[1]
    # right subspace U = span of the layer inputs y_prev (rank <= batch).
    yprev = ys[layer]                                  # [B, n_in]
    Us, Ss, Uh = torch.linalg.svd(yprev, full_matrices=False)
    dU = int((Ss > 1e-3 * Ss[0]).sum().item())
    U = Uh[:dU].t().contiguous()                       # [n_in, dU]
    PVU_g = V @ (V.t() @ g_true @ U) @ U.t()           # project onto span(V) (x) span(U)
    gV, varV = forward_grad(f, W0, lambda: V @ torch.randn(dV, n_in), K)
    gVU, varVU = forward_grad(f, W0, lambda: V @ torch.randn(dV, dU) @ U.t(), K)
    gF, varF = forward_grad(f, W0, lambda: torch.randn_like(W0), K)
    return dict(dV=dV, dU=dU,
                mass_in_V=cos(V @ (V.t() @ g_true), g_true),
                mass_in_VU=cos(PVU_g, g_true),
                cos_gVU_true=cos(gVU, g_true),
                cos_gVU_target=cos(gVU, PVU_g),
                cos_gV_true=cos(gV, g_true),
                cos_gF_true=cos(gF, g_true),
                var_ratio_VU=(varF / varVU if varVU > 0 else float("nan")))


def build(arch, cfg, seed):
    if arch == "plain":
        return MLP(cfg["d_in"], cfg["width"], cfg["n_layers"], cfg["act"], seed=seed)
    return ArchMLP(cfg["d_in"], cfg["width"], cfg["n_layers"], "residual", cfg["act"],
                   alpha=cfg["alpha"], seed=seed)


def main():
    torch.set_default_dtype(torch.float64)
    import numpy as np
    print(f"K={CFG['K']} JVP samples, width={CFG['width']}, residual alpha={CFG['alpha']}\n")
    for arch in ("plain", "residual"):
        agg = {}
        for seed in CFG["seeds"]:
            g = torch.Generator().manual_seed(1000 + seed)
            x = torch.randn(CFG["batch"], CFG["d_in"], generator=g)
            xp = x + CFG["aug_noise"] * torch.randn(x.shape, generator=g)
            model = build(arch, CFG, seed)
            for layer in range(model.n_layers - 1):
                agg.setdefault(layer, []).append(run_layer(model, x, xp, layer, CFG["tau"], CFG["K"]))
        print(f"[{arch}]  L  dV  dU  massVU  cosVU->true cosVU->tgt  massV  cosV->true varRatioVU")
        for layer, rs in agg.items():
            m = lambda k: float(np.mean([r[k] for r in rs]))
            print(f"        {layer:>2} {m('dV'):>3.0f} {m('dU'):>3.0f} {m('mass_in_VU'):>6.3f} "
                  f"{m('cos_gVU_true'):>11.3f} {m('cos_gVU_target'):>10.3f} {m('mass_in_V'):>6.3f} "
                  f"{m('cos_gV_true'):>10.3f} {m('var_ratio_VU'):>10.1f}")
        print()
    print("VU = restrict tangents to span(V) (x) span(y_prev), effective dim dV*dU ~ dV*B.")
    print("If cosVU->target ~ 1 at this K and massVU high => sharp low-variance BP-free estimate.")


if __name__ == "__main__":
    main()
