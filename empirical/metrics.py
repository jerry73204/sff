"""Per-layer diagnostics (THEORY.md sec 4): A^(l), Delta_Gram^(l), Aniso^(l), d_V.

These ARE the empirical result. A^(l) is computed in gradients.alignment_cosine; the
rest live here.
"""
from __future__ import annotations
import torch
from gradients import signal


def gram(z: torch.Tensor) -> torch.Tensor:
    """K_ij = <z_i, z_j>.  [B,B]."""
    return z @ z.t()


def delta_gram(z_l: torch.Tensor, z_L: torch.Tensor) -> float:
    """Delta_Gram^(l) = ||K^(l) - K^(L)||_F / ||K^(L)||_F  (normalized)."""
    Kl, KL = gram(z_l), gram(z_L)
    return (Kl - KL).norm().item() / (KL.norm().item() + 1e-12)


def contrastive_subspace(s: torch.Tensor, thresh: float = 1e-3):
    """V = top-d_V right singular vectors of the stacked signal matrix s [B,n];
    d_V = numerical rank (sigma > thresh * sigma_max).  Returns (V [n,d_V], d_V)."""
    U, S, Vh = torch.linalg.svd(s, full_matrices=False)
    if S.numel() == 0 or S[0].item() == 0.0:
        return s.new_zeros(s.shape[1], 0), 0
    dV = int((S > thresh * S[0]).sum().item())
    V = Vh[:dV].t().contiguous()          # [n, d_V]
    return V, dV


def downstream_jacobian_linear(model, layer: int) -> torch.Tensor:
    """M^(l+1->L) for linear mode: product of downstream weights, [n_L, n].
    Identity if `layer` is the last weight."""
    n = model.W[layer].shape[0]
    M = torch.eye(n)
    for k in range(layer + 1, model.n_layers):
        M = model.W[k] @ M
    return M


def downstream_jacobian_relu(model, x, layer: int) -> torch.Tensor:
    """M^(l+1->L) for ReLU, averaged over the batch: mean_i prod_k W^(k) D^(k)_i.
    Per-sample masks from the forward pass (THEORY.md sec 2.3)."""
    ys = model(x)
    n = model.W[layer].shape[0]
    B = x.shape[0]
    Msum = torch.zeros(model.W[-1].shape[0], n)
    for i in range(B):
        Mi = torch.eye(n)
        for k in range(layer + 1, model.n_layers):
            pre = ys[k] @ model.W[k].t()           # [B, out]
            D = (pre[i] > 0).float()               # [out]
            Mi = (model.W[k] * D.unsqueeze(1)) @ Mi
        Msum += Mi
    return Msum / B


def aniso(M: torch.Tensor, V: torch.Tensor) -> float:
    """Aniso^(l) = ||V^T R V - (tr/dim) I||_F / ||V^T R V||_F,  R = M^T M restricted to V.
    Zero deviation from a scalar multiple of identity = perfectly isotropic on V."""
    if V.shape[1] == 0:
        return 0.0
    R = M.t() @ M
    VRV = V.t() @ R @ V
    dim = VRV.shape[0]
    iso = (torch.trace(VRV) / dim) * torch.eye(dim)
    den = VRV.norm().item()
    return (VRV - iso).norm().item() / den if den > 0 else 0.0
