"""Path A: does residual BOUND the FF<->BP alignment gap as depth grows?

The theory says the cross-layer drift delta (the binding term) is a DEPTH effect: in a plain net
the downstream transport M = M^{l->L} compounds anisotropy over L-l layers, so alignment A should
DECAY with depth. Residual makes each block M = I + alpha*J near-identity, so the product stays
near-identity and A should stay HIGH as L grows. This is the one theory prediction not yet measured,
and it is exactly where BP's memory cost explodes (O(L) activations) -- so a bounded gap at large L
is the strongest practical case for residual local nets.

We measure mean alignment A (local goodness grad vs BP-through-final-goodness grad, averaged over
non-final blocks) at init, for plain vs residual, across L = 4..64. Alignment-only (no downstream
Jacobian) so it is cheap even at L=64. Residual scale alpha = 1/sqrt(L) (stable/ReZero-style, keeps
||M-I|| ~ O(1) as depth grows).

Prediction: A_plain(L) decays toward 0; A_residual(L) stays high and roughly flat.

Run: python experiments/depth_scaling.py
"""
from __future__ import annotations
import os, sys
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP
import arch as A
from gradients import global_grad, alignment_cosine

DEPTHS = [4, 8, 16, 32, 64]
WIDTH, B, TAU, DIN, NOISE = 128, 32, 0.5, 64, 0.3


def mean_A(model, x, xp, tau):
    vals = [alignment_cosine(A.local_grad(model, x, xp, l, tau),
                             global_grad(model, x, xp, l, tau))
            for l in range(model.n_layers - 1)]
    return sum(vals) / len(vals)


def main():
    torch.set_default_dtype(torch.float64)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, DIN, generator=g)
    xp = x + NOISE * torch.randn(B, DIN, generator=g)
    print(f"width={WIDTH} B={B} tau={TAU}  alpha=1/sqrt(L)  (mean alignment A at init vs depth)\n")
    print(f"  {'L':>3}  {'A_plain':>8}  {'A_resid':>8}  {'alpha':>6}")
    rows = []
    for L in DEPTHS:
        mp = ArchMLP(DIN, WIDTH, L, "plain", "relu", seed=0)
        Ap = mean_A(mp, x, xp, TAU)
        alpha = 1.0 / math.sqrt(L)
        mr = ArchMLP(DIN, WIDTH, L, "residual", "relu", alpha=alpha, seed=0)
        Ar = mean_A(mr, x, xp, TAU)
        rows.append((L, Ap, Ar, alpha))
        print(f"  {L:>3}  {Ap:>8.3f}  {Ar:>8.3f}  {alpha:>6.3f}", flush=True)

    print("\n=== VERDICT ===")
    p0, pN = rows[0][1], rows[-1][1]
    r0, rN = rows[0][2], rows[-1][2]
    print(f"plain   A: L={DEPTHS[0]} -> {p0:.3f}   L={DEPTHS[-1]} -> {pN:.3f}   (drop {p0-pN:+.3f})")
    print(f"residual A: L={DEPTHS[0]} -> {r0:.3f}   L={DEPTHS[-1]} -> {rN:.3f}   (drop {r0-rN:+.3f})")
    print("=> " + ("residual BOUNDS the depth gap (A flat) while plain decays -- theory confirmed"
                   if (p0 - pN) > 0.1 and (r0 - rN) < (p0 - pN) / 2 else
                   "depth effect weaker than predicted at this width/scale"))
    print("memory (measured, memory_footprint.py): BP O(L) activations vs SCFF flat O(1); 51x at L=64.")


if __name__ == "__main__":
    main()
