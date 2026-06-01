"""SCFF training: per-layer InfoNCE on normalized activations, layer-local updates
with stop-gradient between layers (THEORY.md sec 2.1; design.md 2.1).

Augmentation-based positives, in-batch negatives. The local goodness gradient is the
closed-form `gradients.local_grad` (verified against autograd in tests/). Updates ascend
goodness. The global gradient is NEVER used here -- it is a measurement probe only.
"""
from __future__ import annotations
import torch
from model import normalize
from gradients import local_grad, local_goodness


def augment(x: torch.Tensor, noise: float = 0.1, seed_gen: torch.Generator | None = None):
    """Simple augmentation: additive Gaussian noise -> positive partner."""
    return x + noise * torch.randn(x.shape, generator=seed_gen)


@torch.no_grad()
def scff_step(model, x, x_pos, tau: float, lr: float) -> None:
    """One simultaneous SCFF update: each layer ascends its own local goodness using
    the current weights (stop-grad between layers). Grads computed from one shared
    forward pass per layer input."""
    grads = [local_grad(model, x, x_pos, l, tau) for l in range(model.n_layers)]
    for l in range(model.n_layers):
        model.W[l].add_(lr * grads[l])


def total_goodness(model, x, x_pos, tau: float) -> float:
    """Sum of per-layer goodness (diagnostic)."""
    g = 0.0
    ys, ysp = model(x), model(x_pos)
    for l in range(1, model.n_layers + 1):
        z, zp = normalize(ys[l]), normalize(ysp[l])
        g += local_goodness(z, zp.detach(), tau).item()
    return g
