"""Predictive-coding network (Whittington & Bogacz 2017) — the best-developed concrete
biologically-motivated (NGRAD) backprop-free rule.

Value nodes x^(ℓ); top-down prediction μ^(ℓ) = W^(ℓ) f(x^(ℓ-1)); error ε^(ℓ) = x^(ℓ) − μ^(ℓ).
Free energy F = Σ ½‖ε‖². Learning:
  1. feedforward init x = μ (so ε = 0 except the clamped output),
  2. clamp output x^(L) = target,
  3. INFERENCE: settle hidden x for T steps   ẋ^(ℓ) = −ε^(ℓ) + f'(x^(ℓ))⊙(W^(ℓ+1)ᵀ ε^(ℓ+1)),
  4. local Hebbian update   ΔW^(ℓ) = ε^(ℓ) f(x^(ℓ-1))ᵀ   (pre × post error, no backprop).
As settling T→∞ the update converges to the backprop gradient — settling IS the cross-layer
feedback. W[i] indexes W^(i+1); lists returned align to W[i].
"""
from __future__ import annotations
import math
import torch


def f(x):
    return torch.tanh(x)


def fp(x):
    return 1.0 - torch.tanh(x) ** 2


class PCNet:
    def __init__(self, dims, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.dims = list(dims)
        self.L = len(dims) - 1
        self.W = [torch.randn(dims[l + 1], dims[l], generator=g) / math.sqrt(dims[l])
                  for l in range(self.L)]

    def feedforward(self, x0):
        """x^(ℓ) = μ^(ℓ) = W^(ℓ) f(x^(ℓ-1)); returns [x0, x1, ..., xL]."""
        xs = [x0]
        for l in range(self.L):
            xs.append(f(xs[-1]) @ self.W[l].t())
        return xs

    def bp_descent(self, x0, target):
        """Backprop descent direction −∇_{W} ½‖x^(L) − target‖² (autograd), per W[i]."""
        Wv = [w.clone().requires_grad_(True) for w in self.W]
        x = x0
        for l in range(self.L):
            x = f(x) @ Wv[l].t()
        loss = 0.5 * ((x - target) ** 2).sum(1).mean()
        return [-g for g in torch.autograd.grad(loss, Wv)]

    def _errors(self, xs):
        return [None] + [xs[l] - f(xs[l - 1]) @ self.W[l - 1].t()
                         for l in range(1, self.L + 1)]

    def pc_update(self, x0, target, T, beta=0.1):
        """PC weight change ΔW^(ℓ) = ε^(ℓ) f(x^(ℓ-1))ᵀ after T inference steps. Per W[i]."""
        xs = [x.clone() for x in self.feedforward(x0)]
        xs[-1] = target.clone()                       # clamp output
        for _ in range(T):
            eps = self._errors(xs)
            new = [x.clone() for x in xs]
            for l in range(1, self.L):                # hidden nodes only
                topdown = eps[l + 1] @ self.W[l]      # W^(ℓ+1)ᵀ ε^(ℓ+1)
                new[l] = xs[l] + beta * (-eps[l] + fp(xs[l]) * topdown)
            xs = new
            xs[-1] = target                           # keep output clamped
        eps = self._errors(xs)
        B = x0.shape[0]
        return [eps[l].t() @ f(xs[l - 1]) / B for l in range(1, self.L + 1)]
