"""Idea D: local isometry penalty (the soft sibling of residual).

The grad_decomp diagnostic showed the entire FF<->BP gap is transport non-isometry (M^T M != cI),
and that *learning grows it*. Residual fixes transport structurally (M ~= I). Idea D instead adds a
LOCAL, BP-free objective term that keeps each block's OWN Jacobian near-isometric, so the product
M = prod_l J^l stays near-isometric without any downstream information.

Isometry (up to scale c) means ||J_i v||^2 = c for every unit direction v, i.e. INDEPENDENT of v.
So we penalize the variance of ||J_i v||^2 across random probe directions v (scale-free -- exactly
the anisotropy metrics.aniso measures). Per block l, with J_i the per-sample block Jacobian:

  plain/linear   J_i v = D_i ⊙ (W v)          (D_i = ReLU mask, 1 in linear mode)
  residual       J_i v = v + alpha D_i ⊙ (W v)

penalty(W[l]) = mean_i Var_m ||J_i v_m||^2 ,  differentiable wrt W[l], no downstream info.

Training: each block ascends its goodness grad MINUS lambda * grad(penalty). Compare on MNIST:
  plain-SCFF (lambda=0)         baseline (the gap)
  plain-SCFF + iso-D (lambda>0) does the local isometry penalty raise alignment / accuracy?
  residual-SCFF                 the structural transport fix (reference)

Hypothesis (from the price-of-locality tension): iso-D raises alignment A on the plain net but
caps accuracy (expressivity tax), i.e. it buys alignment the same way residual does -- by limiting
metric distortion -- and pays the same way.

Run: python experiments/iso_penalty.py
"""
from __future__ import annotations
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP
import arch as A
from task_accuracy import CFG, load_data, probe, train_scff, mean_alignment, epochs_iter


def iso_penalty(model, x, layer, k, gen):
    """mean_i Var_m ||J_i v_m||^2 for k random unit probe directions v_m. Differentiable wrt
    W[layer]; block input detached (layer-local). Scale-free (per-sample mean subtracted)."""
    with torch.no_grad():
        ys = model(x)
    y_prev = ys[layer].detach()                       # [B, in]
    W = model.W[layer]                                # live (carries grad)
    pre = y_prev @ W.t()                              # [B, out]
    d = (pre > 0).double() if model.act_name == "relu" else torch.ones_like(pre)
    v = torch.randn(k, y_prev.shape[1], generator=gen)
    v = v / v.norm(dim=1, keepdim=True)               # [k, in], unit
    Wv = v @ W.t()                                    # [k, out]
    Jv = d[:, None, :] * Wv[None, :, :]               # [B, k, out]  (plain/linear)
    if model.arch == "residual":
        Jv = v[None, :, :] + model.alpha * Jv         # in==out width, residual identity branch
    q = (Jv ** 2).sum(-1)                             # [B, k] = ||J_i v_m||^2
    c = q.mean(dim=1, keepdim=True)                   # free per-sample scale
    return ((q - c) ** 2).mean()


def train_scff_isoD(model, Xtr, cfg, gen, lam, k=8):
    pg = torch.Generator().manual_seed(cfg["seed"] + 7)
    for _ in range(cfg["epochs"]):
        for xb in epochs_iter(Xtr, cfg["batch"], gen):
            xp = xb + cfg["aug_noise"] * torch.randn(xb.shape)
            g_good = [A.local_grad(model, xb, xp, l, cfg["tau"]) for l in range(model.n_layers)]
            g_iso = []
            for l in range(model.n_layers):
                pen = iso_penalty(model, xb, l, k, pg)
                g_iso.append(torch.autograd.grad(pen, model.W[l])[0])
            with torch.no_grad():
                for l in range(model.n_layers):
                    model.W[l].add_(cfg["lr_scff"] * (g_good[l] - lam * g_iso[l]))  # ascend goodness, descend penalty


def main():
    torch.set_default_dtype(torch.float64)
    name, Xtr, ytr, Xte, yte = load_data(CFG)
    d_in = Xtr.shape[1]
    print(f"data={name}  d_in={d_in}  train={len(Xtr)} test={len(Xte)}  "
          f"width={CFG['width']} L={CFG['n_layers']} {CFG['act']}  epochs={CFG['epochs']}\n")

    def g():
        return torch.Generator().manual_seed(CFG["seed"])

    print(f"  {'method':28s} {'probe':>7}  {'A':>6}")
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "plain", CFG["act"], seed=CFG["seed"])
    train_scff(m, Xtr, CFG, g())
    print(f"  {'plain-SCFF (lam=0)':28s} {probe(m, Xtr, ytr, Xte, yte):>7.4f}  {mean_alignment(m, Xte, CFG):>6.3f}", flush=True)

    rows = []
    for lam in (0.3, 1.0, 3.0, 10.0):
        m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "plain", CFG["act"], seed=CFG["seed"])
        train_scff_isoD(m, Xtr, CFG, g(), lam)
        p, a = probe(m, Xtr, ytr, Xte, yte), mean_alignment(m, Xte, CFG)
        rows.append((lam, p, a))
        print(f"  {'plain-SCFF + isoD lam='+format(lam,'.1f'):28s} {p:>7.4f}  {a:>6.3f}", flush=True)

    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "residual", CFG["act"], alpha=0.2, seed=CFG["seed"])
    train_scff(m, Xtr, CFG, g())
    pr, ar = probe(m, Xtr, ytr, Xte, yte), mean_alignment(m, Xte, CFG)
    print(f"  {'residual-SCFF (a=0.2)':28s} {pr:>7.4f}  {ar:>6.3f}", flush=True)

    print("\n=== VERDICT ===")
    best = max(rows, key=lambda r: r[1])
    print(f"best isoD: lam={best[0]:.1f}  probe={best[1]:.4f}  A={best[2]:.3f}")
    print(f"residual : probe={pr:.4f}  A={ar:.3f}")
    print("=> " + ("isoD raises A AND matches residual accuracy -- real alternative"
                   if best[1] >= pr - 0.01 and best[2] > 0.7 else
                   "isoD raises alignment but does NOT match residual accuracy -- expectancy tax / dominated"))


if __name__ == "__main__":
    main()
