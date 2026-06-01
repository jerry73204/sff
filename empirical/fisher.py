"""K-FAC-style local Fisher preconditioning (NGD-FF; design.md 2.1 item 3).

The local Fisher block factorizes  F^(l) ~= A^(l) (x) G_hat^(l), with
  A^(l)     = E[ y^(l-1) y^(l-1)^T ]   (input covariance,  [in, in]),
  G_hat^(l) = E[ delta delta^T ]       (LOCAL goodness left-factor cov,  [out, out]),
where delta_i = (1/tau) Pperp_{z_i} s_i is the per-sample left factor from
gradients.local_grad_factors. Crucially G_hat uses the *local* goodness gradient, not a
global one -- the update stays forward-only / layer-local.

Damped inverses (A + lambda I)^{-1}, (G_hat + lambda I)^{-1}. The natural gradient for a
weight W [out, in] is  G_hat_damp^{-1} @ grad @ A_damp^{-1}.
"""
from __future__ import annotations
import torch
from gradients import local_grad_factors


def kfac_factors(left: torch.Tensor, y_prev: torch.Tensor):
    """Empirical K-FAC factors from the gradient's left/right factors.
    A = E[y_prev y_prev^T] [in,in]; G = E[left left^T] [out,out]."""
    B = left.shape[0]
    A = (y_prev.t() @ y_prev) / B
    G = (left.t() @ left) / B
    return A, G


def _damped_inv(M: torch.Tensor, damp: float) -> torch.Tensor:
    n = M.shape[0]
    return torch.linalg.inv(M + damp * torch.eye(n, dtype=M.dtype))


def natural_grad(model, x, x_pos, layer, tau, damp: float = 1e-2):
    """K-FAC natural gradient for `layer`: G_hat_damp^{-1} @ grad @ A_damp^{-1}.
    Reduces to the ordinary local goodness gradient as damp -> infinity (up to scale)."""
    left, y_prev = local_grad_factors(model, x, x_pos, layer, tau)
    grad = left.t() @ y_prev
    A, G = kfac_factors(left, y_prev)
    return _damped_inv(G, damp) @ grad @ _damped_inv(A, damp)
