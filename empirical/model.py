"""MLP with mu-P-style init (THEORY.md sec 1). Linear mode is primary.

Symbols (THEORY.md):
  L        n_layers
  n        width
  W[l]     weight of layer l, shape [out, in]
  y[l]     activation, y[l] = act(W[l] @ y[l-1]); y[0] = x
  z[l]     normalized rep, y[l] / ||y[l]||  (per sample)
Tensors are batch-first: y has shape [B, dim].
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Configurable-depth/width MLP, mu-P init, linear or ReLU.

    mu-P: hidden weights ~ N(0, 1/fan_in); the last layer is scaled by an extra
    1/fan_in (variance 1/fan_in^2) -- THEORY.md sec 1. Unit-tested in tests/.
    """

    def __init__(self, d_in: int, width: int, n_layers: int, act: str = "linear",
                 seed: int | None = None):
        super().__init__()
        assert act in ("linear", "relu")
        self.act_name = act
        self.width = width
        self.n_layers = n_layers
        if seed is not None:
            torch.manual_seed(seed)
        dims = [d_in] + [width] * n_layers
        self.W = nn.ParameterList()
        for l in range(n_layers):
            fan_in = dims[l]
            w = torch.randn(dims[l + 1], fan_in) / math.sqrt(fan_in)
            if l == n_layers - 1:  # mu-P last-layer extra 1/fan_in scaling
                w = w / math.sqrt(fan_in)
            self.W.append(nn.Parameter(w))

    def act(self, t: torch.Tensor) -> torch.Tensor:
        return t if self.act_name == "linear" else torch.relu(t)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return [y0=x, y1, ..., yL]. y[l] = act(y[l-1] @ W[l].T)."""
        ys = [x]
        h = x
        for l in range(self.n_layers):
            h = self.act(h @ self.W[l].t())
            ys.append(h)
        return ys


def normalize(y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """z = y / ||y|| per sample (rows)."""
    return y / (y.norm(dim=1, keepdim=True) + eps)
