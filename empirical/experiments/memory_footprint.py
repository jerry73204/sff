"""Empirical activation-memory footprint: BP vs greedy SCFF vs residual SCFF.

Claim (from docs/FINDINGS.md architecture review): BP must retain every layer's activations for
the backward pass -> activation memory O(L*B*n); forward-only layer-local SCFF retains only one
layer's local graph at a time -> O(B*n), depth-INDEPENDENT.

We measure the exact quantity with `torch.autograd.graph.saved_tensors_hooks`: the total bytes
of tensors saved for backward. For BP that's the whole-network forward graph; for greedy SCFF
it's the peak over single-block local-goodness graphs.

Run: python experiments/memory_footprint.py
"""
from __future__ import annotations
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MLP, normalize
from arch import ArchMLP
from gradients import local_goodness

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS = os.path.join(HERE, "plots")

CFG = dict(d_in=128, width=256, batch=64, depths=[2, 4, 8, 16, 32, 64], tau=0.5)


class ByteCounter:
    """Sums the bytes of distinct tensors saved for backward (activation memory)."""
    def __init__(self):
        self.total = 0
        self._seen = set()

    def pack(self, t):
        key = (t.data_ptr(), t.nelement())
        if key not in self._seen and t.data_ptr() != 0:
            self._seen.add(key)
            self.total += t.element_size() * t.nelement()
        return t

    def unpack(self, t):
        return t


def bp_saved_bytes(d_in, n, L, B):
    """Bytes retained for a full-network backward pass (BP)."""
    x = torch.randn(B, d_in)
    target = torch.randn(B, n)
    model = MLP(d_in, n, L, act="relu", seed=0)
    c = ByteCounter()
    with torch.autograd.graph.saved_tensors_hooks(c.pack, c.unpack):
        ys = model(x)
        loss = ((ys[-1] - target) ** 2).mean()
    loss.backward()
    return c.total


def greedy_peak_saved_bytes(d_in, n, L, B, tau, arch):
    """Peak bytes retained while training ONE block's local goodness at a time."""
    x = torch.randn(B, d_in)
    xp = x + 0.1 * torch.randn(B, d_in)
    model = ArchMLP(d_in, n, L, arch, act="relu", alpha=0.1, seed=0)
    peak = 0
    for layer in range(model.n_layers):
        c = ByteCounter()
        with torch.autograd.graph.saved_tensors_hooks(c.pack, c.unpack):
            z = model.block_output(x, layer)          # single-block local graph
            zp = model.block_output(xp, layer)
            g = local_goodness(z, zp.detach(), tau)
        torch.autograd.grad(g, model.W[layer])
        peak = max(peak, c.total)
    return peak


def main():
    torch.set_default_dtype(torch.float32)            # realistic dtype for memory
    os.makedirs(PLOTS, exist_ok=True)
    rows = []
    for L in CFG["depths"]:
        bp = bp_saved_bytes(CFG["d_in"], CFG["width"], L, CFG["batch"])
        gr = greedy_peak_saved_bytes(CFG["d_in"], CFG["width"], L, CFG["batch"], CFG["tau"], "plain")
        rs = greedy_peak_saved_bytes(CFG["d_in"], CFG["width"], L, CFG["batch"], CFG["tau"], "residual")
        rows.append((L, bp, gr, rs))
        print(f"L={L:>3}  BP={bp/1e6:7.2f} MB   greedy-plain={gr/1e6:6.3f} MB   "
              f"greedy-residual={rs/1e6:6.3f} MB   BP/greedy={bp/max(1,gr):6.1f}x", flush=True)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ls = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ls, [r[1] / 1e6 for r in rows], "o-", label="BP (store all activations)")
    ax.plot(Ls, [r[2] / 1e6 for r in rows], "s-", label="greedy SCFF (one block)")
    ax.plot(Ls, [r[3] / 1e6 for r in rows], "^-", label="greedy residual SCFF")
    ax.set_xlabel("depth L"); ax.set_ylabel("activation memory retained (MB)")
    ax.set_title(f"Backward-pass memory vs depth (n={CFG['width']}, B={CFG['batch']})\n"
                 "BP ~ O(L·B·n); forward-only SCFF ~ O(B·n), depth-independent")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PLOTS, "memory_footprint.png"); fig.savefig(path, dpi=130)
    print("plot ->", path)

    bp0, bpN = rows[0][1], rows[-1][1]
    gr0, grN = rows[0][2], rows[-1][2]
    print(f"\nVERDICT: BP activation memory grows {bpN/bp0:.1f}x from L={Ls[0]} to L={Ls[-1]}; "
          f"greedy SCFF grows {grN/gr0:.2f}x (flat).")
    print(f"At L={Ls[-1]}: BP keeps {bpN/1e6:.1f} MB vs SCFF {grN/1e6:.2f} MB "
          f"= {bpN/max(1,grN):.0f}x less for the forward-only rule.")


if __name__ == "__main__":
    main()
