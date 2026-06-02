"""Skip-connection architectures (residual, dense) for the SCFF alignment study,
plus arch-aware autograd helpers.
See docs/superpowers/specs/2026-06-02-skip-connections-scff-design.md.

ArchMLP: stem (d_in->width) then L blocks at fixed width.
  plain     y^l = act(W^l y^{l-1})
  residual  y^l = y^{l-1} + alpha * act(W^l y^{l-1})        (alpha default 1/sqrt(L))
  dense     y^l = act(W^l concat(y^0,...,y^{l-1}))
forward returns [y0=stem(x), y1, ..., yL].

With norm=True, each block output passes through a parameter-free LayerNorm
`(y-mean)/std` (per sample, over the width) before the next block sees it. This is
the kernel-drift (delta) probe of docs/FINDINGS.md gap #3: does normalization, which
real nets use, cut the cross-layer delta that plain MLPs exhibit? norm=False (default)
is exactly the original behavior (the 30 anchor tests stay green).

The local goodness gradient uses autograd (block_output); the downstream Jacobian uses
autograd (forward_from). Both are arch-agnostic. metrics.* and gradients.{global_grad,
alignment_cosine,local_goodness,signal} are reused unchanged (arch-agnostic).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from model import normalize
from gradients import local_goodness, global_grad, alignment_cosine, signal
import metrics


class ArchMLP(nn.Module):
    def __init__(self, d_in, width, n_layers, arch="plain", act="linear",
                 alpha=None, seed=None, norm=False):
        super().__init__()
        assert arch in ("plain", "residual", "dense")
        assert act in ("linear", "relu")
        self.arch, self.act_name = arch, act
        self.width, self.n_layers = width, n_layers
        self.norm = norm
        self.alpha = (1.0 / math.sqrt(n_layers)) if alpha is None else alpha
        if seed is not None:
            torch.manual_seed(seed)
        self.stem = nn.Parameter(torch.randn(width, d_in) / math.sqrt(d_in))
        self.W = nn.ParameterList()
        for l in range(n_layers):
            fan_in = width * (l + 1) if arch == "dense" else width
            w = torch.randn(width, fan_in) / math.sqrt(fan_in)
            if l == n_layers - 1:                # mu-P last-block extra 1/sqrt(fan_in)
                w = w / math.sqrt(fan_in)
            self.W.append(nn.Parameter(w))

    def act(self, t):
        return t if self.act_name == "linear" else torch.relu(t)

    def _ln(self, y, eps=1e-5):
        """Parameter-free LayerNorm: (y-mean)/std per sample over the width.
        Identity (returns y) when norm=False."""
        if not self.norm:
            return y
        mu = y.mean(dim=1, keepdim=True)
        var = y.var(dim=1, unbiased=False, keepdim=True)
        return (y - mu) / torch.sqrt(var + eps)

    def _block(self, ys, l):
        if self.arch == "plain":
            return self._ln(self.act(ys[-1] @ self.W[l].t()))
        if self.arch == "residual":
            return self._ln(ys[-1] + self.alpha * self.act(ys[-1] @ self.W[l].t()))
        return self._ln(self.act(torch.cat(ys, dim=1) @ self.W[l].t()))   # dense

    def forward(self, x):
        ys = [self.act(x @ self.stem.t())]
        for l in range(self.n_layers):
            ys.append(self._block(ys, l))
        return ys

    def block_output(self, x, layer):
        """z^(layer+1) as a differentiable function of W[layer] only; all block inputs
        detached (layer-local, stop-grad between blocks)."""
        with torch.no_grad():
            ys = [y.detach() for y in self.forward(x)]
        if self.arch == "plain":
            out = self._ln(self.act(ys[layer] @ self.W[layer].t()))
        elif self.arch == "residual":
            out = self._ln(ys[layer] + self.alpha * self.act(ys[layer] @ self.W[layer].t()))
        else:  # dense
            out = self._ln(self.act(torch.cat(ys[:layer + 1], dim=1) @ self.W[layer].t()))
        return normalize(out)

    def forward_from(self, start, y_start, frozen):
        """Recompute y^L given y^(start) = y_start, holding frozen[0..start-1].
        `frozen` is a full forward(x) (detached); used for dense concat paths."""
        ys = list(frozen[:start]) + [y_start]
        for l in range(start, self.n_layers):
            ys.append(self._block(ys, l))
        return ys[-1]


def local_grad(model, x, x_pos, layer, tau):
    """grad_{W[layer]} g^(layer) via autograd of the layer-local block goodness."""
    z = model.block_output(x, layer)
    zp = model.block_output(x_pos, layer)
    g = local_goodness(z, zp.detach(), tau)
    return torch.autograd.grad(g, model.W[layer])[0]


def downstream_jacobian(model, x, layer):
    """M^(l+1->L) = mean_i d y^L_i / d y^(l+1)_i, per-sample autograd jacobian.
    Arch-agnostic (dense concat via forward_from). [n_L, n]. Identity at last block."""
    start = layer + 1
    with torch.no_grad():
        frozen_all = [y.detach() for y in model(x)]
    B = x.shape[0]
    Ms = []
    for i in range(B):
        fz = [f[i:i + 1].detach() for f in frozen_all]
        y0 = fz[start].squeeze(0)
        def fn(yv, _fz=fz, _start=start):
            return model.forward_from(_start, yv.unsqueeze(0), _fz).squeeze(0)
        Ms.append(torch.autograd.functional.jacobian(fn, y0))
    return torch.stack(Ms).mean(0)


def measure_blocks(model, x, x_pos, tau):
    """Per block: (layer, A, delta_gram, aniso, d_V). Reuses metrics.* (arch-agnostic)."""
    ys, ysp = model(x), model(x_pos)
    zL = normalize(ys[-1])
    rows = []
    for l in range(model.n_layers):
        gl = local_grad(model, x, x_pos, l, tau)
        gg = global_grad(model, x, x_pos, l, tau)
        Aval = alignment_cosine(gl, gg)
        z_l, zp_l = normalize(ys[l + 1]), normalize(ysp[l + 1])
        dg = metrics.delta_gram(z_l, zL)
        s, _ = signal(z_l, zp_l.detach(), tau)
        V, dV = metrics.contrastive_subspace(s)
        M = downstream_jacobian(model, x, l)
        an = metrics.aniso(M, V)
        rows.append((l, Aval, dg, an, dV))
    return rows
