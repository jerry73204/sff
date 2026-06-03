"""Convolutional SCFF (conv port of arch.ArchMLP) for the CIFAR credibility test.

Stem conv (3->C, stride 2: 32x32 -> 16x16) then `n_blocks` conv blocks at fixed C and spatial size
(padding=same), so the residual identity branch is dimension-matched:

  plain     y^l = relu(conv_l(y^{l-1}))
  residual  y^l = y^{l-1} + alpha * relu(conv_l(y^{l-1}))        (alpha ~ 1/sqrt(n_blocks) or fixed)

The per-block "rep" for the local InfoNCE goodness is the **global-average-pooled, normalized**
feature map: pooled(y) = normalize(mean_{H,W} y) in R^C. Layer-local: block_output detaches the
block input, so the local goodness gradient flows only into conv_l (stop-grad between blocks,
forward-only, no weight transport) -- exactly the MLP contract, lifted to conv.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import normalize
from gradients import local_goodness


class ConvSCFF(nn.Module):
    def __init__(self, C=64, n_blocks=4, arch="residual", alpha=0.2, in_ch=3, seed=0):
        super().__init__()
        assert arch in ("plain", "residual")
        torch.manual_seed(seed)
        self.arch, self.alpha, self.C, self.n_blocks = arch, alpha, C, n_blocks
        self.stem = nn.Conv2d(in_ch, C, 3, stride=2, padding=1)      # 32 -> 16
        self.blocks = nn.ModuleList([nn.Conv2d(C, C, 3, padding=1) for _ in range(n_blocks)])

    def act(self, t):
        return F.relu(t)

    def _apply_block(self, y, l):
        h = self.act(self.blocks[l](y))
        return y + self.alpha * h if self.arch == "residual" else h

    def forward(self, x):
        ys = [self.act(self.stem(x))]                                # ys[0] = stem out
        for l in range(self.n_blocks):
            ys.append(self._apply_block(ys[-1], l))                  # ys[l+1] = block l out
        return ys

    def pooled(self, y):
        """Global-avg-pool + L2-normalize: [B,C,H,W] -> [B,C] on the unit sphere."""
        return normalize(y.mean(dim=(2, 3)))

    def block_output(self, x, l):
        """pooled rep of block l, differentiable wrt conv_l ONLY (block input detached)."""
        with torch.no_grad():
            ys = [y.detach() for y in self.forward(x)]
        y_in = ys[l]                                                 # detached input to block l
        out = self._apply_block(y_in, l)
        return self.pooled(out)


def local_grad(model, x, x_pos, layer, tau):
    """grad of block-`layer` local InfoNCE goodness wrt conv_`layer` params (weight, bias)."""
    z = model.block_output(x, layer)
    zp = model.block_output(x_pos, layer)
    g = local_goodness(z, zp.detach(), tau)
    return torch.autograd.grad(g, list(model.blocks[layer].parameters()))


def global_grad(model, x, x_pos, layer, tau):
    """Probe: grad of the FINAL-block goodness wrt conv_`layer` params (full backward). Measures
    the BP-through-final-goodness direction at layer `layer`. Never fed to an optimizer."""
    ys, ysp = model(x), model(x_pos)
    g = local_goodness(model.pooled(ys[-1]), model.pooled(ysp[-1]).detach(), tau)
    return torch.autograd.grad(g, list(model.blocks[layer].parameters()))


def flat_cos(gs_a, gs_b):
    a = torch.cat([t.reshape(-1) for t in gs_a])
    b = torch.cat([t.reshape(-1) for t in gs_b])
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))
