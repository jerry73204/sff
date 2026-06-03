"""Geometric space-decomposition of the BP gradient, and how FF can match it.

At layer l, both gradients are tangent vectors on the sphere, each = positive - negative:

  g_FF  = Pperp( z+_l  -  sum_j p^l_ij z_l_j )            local softmax p^l
  g_BP  = M^T Pperp( z+_L -  sum_j p^L_ij z_L_j )         output softmax p^L, transported by M^T

The FF<->BP gap splits into two geometric defects:
  (1) KERNEL DRIFT : p^l != p^L  (negatives weighted by a different softmax per layer)
  (2) TRANSPORT    : M^T not an isometry; polar M=QS, the stretch S=(M^T M)^1/2 is the shear.

We attribute the gap by inserting a SHARED-KERNEL local signal (use the output softmax p^L on
layer-l reps):
  g_FF        : layer-l softmax  (the real method)
  g_FF_sharedK: output  softmax  (Idea B: transported/shared kernel for negatives)

Then:
  A_full     = cos(g_FF,         g_BP)   -- real FF<->BP tangent alignment
  A_sharedK  = cos(g_FF_sharedK, g_BP)   -- after removing kernel drift
  cos_kernel = cos(g_FF, g_FF_sharedK)   -- how much kernel drift alone rotates the local signal
  aniso      = ||M^T M - cI|| on V       -- the transport shear (metrics.aniso)

If A_sharedK >> A_full, kernel drift was a big, CHEAPLY-FIXABLE chunk of the gap (Idea B works).
If A_sharedK ~ A_full but both low on plain / high on residual, the residual transport (S~I) is
what matters and the kernel is a sideshow.

Run: python experiments/grad_decomp.py
"""
from __future__ import annotations
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import normalize
from arch import ArchMLP, downstream_jacobian
import arch as A
from gradients import softmax_weights
import metrics

torch.set_default_dtype(torch.float64)


def pperp(s, z):
    """Project per-sample tangent: remove radial part at z. s,z: [B,n]."""
    return s - z * (z * s).sum(1, keepdim=True)


def cos(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def signals_at(z, zp, tau, p=None):
    """Contrastive tangent signal Pperp( zp - p @ z ) at reps z. If p given, use it (shared
    kernel); else compute layer-local softmax on z."""
    if p is None:
        p = softmax_weights(z, tau)
    return pperp(zp - p @ z, z), p


def decompose(model, x, xp, tau):
    ys, ysp = model(x), model(xp)
    zL, zpL = normalize(ys[-1]), normalize(ysp[-1])
    sL, pL = signals_at(zL, zpL, tau)           # output signal + output softmax (shared kernel)
    rows = []
    for l in range(model.n_layers - 1):          # last block is identity transport, skip
        z, zp = normalize(ys[l + 1]), normalize(ysp[l + 1])
        s_loc, _ = signals_at(z, zp, tau)                    # real FF: layer-l softmax
        s_shared, _ = signals_at(z, zp, tau, p=pL)           # Idea B: output softmax on layer-l reps
        M = downstream_jacobian(model, x, l)                  # [n_L, n], batch-mean Jacobian
        s_bp = pperp(sL @ M, z)                                # transport output signal to layer l
        V, dV = metrics.contrastive_subspace(s_loc)
        an = metrics.aniso(M, V)
        rows.append(dict(l=l, A_full=cos(s_loc, s_bp), A_sharedK=cos(s_shared, s_bp),
                         cos_kernel=cos(s_loc, s_shared), aniso=an, dV=dV))
    return rows


def train_scff_inplace(model, X, tau, steps, lr, B, seed):
    """Quick plain-SCFF ascent so representations develop structure (hard negatives), to test
    whether kernel drift appears post-training. Self-supervised: positives = noise-aug views."""
    g = torch.Generator().manual_seed(seed)
    for t in range(steps):
        idx = torch.randint(0, len(X), (B,), generator=g)
        xb = X[idx]; xp = xb + 0.3 * torch.randn(xb.shape, generator=g)
        grads = [A.local_grad(model, xb, xp, l, tau) for l in range(model.n_layers)]
        with torch.no_grad():
            for l in range(model.n_layers):
                model.W[l].add_(lr * grads[l])


def report(tag, model, x, xp, tau):
    rows = decompose(model, x, xp, tau)
    print(f"=== {tag} ===")
    print(f"  {'l':>2}  {'A_full':>7}  {'A_sharedK':>9}  {'cos_kernel':>10}  {'aniso':>6}  {'dV':>3}")
    for r in rows:
        print(f"  {r['l']:>2}  {r['A_full']:>7.3f}  {r['A_sharedK']:>9.3f}  "
              f"{r['cos_kernel']:>10.3f}  {r['aniso']:>6.3f}  {r['dV']:>3}", flush=True)
    meanfull = sum(r['A_full'] for r in rows) / len(rows)
    meanshared = sum(r['A_sharedK'] for r in rows) / len(rows)
    meankern = sum(r['cos_kernel'] for r in rows) / len(rows)
    print(f"  mean A_full={meanfull:.3f}  A_sharedK={meanshared:.3f}  "
          f"cos_kernel={meankern:.3f}  kernel-drift gain={meanshared-meanfull:+.3f}\n")


def main():
    n, L, B, tau = 256, 6, 32, 0.5
    # structured data so trained reps develop real hard negatives (4 gaussian clusters)
    g = torch.Generator().manual_seed(1)
    centers = 3.0 * torch.randn(4, 64, generator=g)
    Xpool = torch.cat([c + torch.randn(200, 64, generator=g) for c in centers])
    x = torch.randn(B, 64, generator=g)
    xp = x + 0.3 * torch.randn(B, 64, generator=g)
    print(f"width={n} L={L} B={B} tau={tau}  (tangent-space FF<->BP gradient decomposition)\n")
    for arch, alpha in (("plain", None), ("residual", 0.2)):
        tag = f"{arch}" + (f" a={alpha}" if alpha else "")
        m = ArchMLP(64, n, L, arch, "relu", alpha=alpha, seed=0)
        report(f"{tag}  [init]", m, x, xp, tau)
        train_scff_inplace(m, Xpool, tau, steps=400, lr=0.05, B=B, seed=2)
        report(f"{tag}  [trained 400 steps]", m, x, xp, tau)


if __name__ == "__main__":
    main()
