"""Local goodness gradient and global probe (THEORY.md sec 3).

The local layer-wise InfoNCE gradient is computed BOTH ways:
  - closed form   `local_grad_formula`   (the THEORY.md expression)
  - autograd      `local_grad_autograd`  (gold standard)
tests/ asserts they agree to 1e-5 in linear mode (the math<->code anchor, M1).

The global gradient `global_grad` is a MEASUREMENT PROBE ONLY: one backward pass
through the final-layer goodness, grad w.r.t. an earlier layer. Never fed to an
optimizer (assert_probe_only guards this).

Sign convention: both are *goodness* gradients (ascent direction). SCFF ascends the
local goodness g^(l); BP-through-L_con ascends the final goodness. The alignment
cosine compares these two ascent directions -- THEORY.md A^(l).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from model import normalize


def softmax_weights(z: torch.Tensor, tau: float) -> torch.Tensor:
    """p_ij = softmax_j(<z_i,z_j>/tau), in-batch keys.  [B,B]  (THEORY.md p^(l))."""
    return F.softmax((z @ z.t()) / tau, dim=1)


def signal(z: torch.Tensor, z_pos: torch.Tensor, tau: float):
    """Contrastive-signal s_i = z_{i+} - sum_j p_ij z_j.  Returns (s [B,n], p [B,B])."""
    p = softmax_weights(z, tau)
    s = z_pos - p @ z
    return s, p


def local_goodness(z: torch.Tensor, z_pos: torch.Tensor, tau: float) -> torch.Tensor:
    """Scalar goodness g = sum_i [ <z_i,z_{i+}>/tau - logsumexp_j <z_i,z_j>/tau ].

    Keys/positives are detached so the autograd of g w.r.t. the query reproduces the
    THEORY.md query-side gradient dg/dz_i = s_i / tau exactly.
    """
    zk = z.detach()
    zp = z_pos.detach()
    scores = (z @ zk.t()) / tau
    pos = (z * zp).sum(1) / tau
    lse = torch.logsumexp(scores, dim=1)
    return (pos - lse).sum()


def local_grad_formula(z, z_pos, y_prev, y_norm, tau, mask=None):
    """Closed-form  grad_{W^(l)} g^(l) = (1/tau) sum_i Pperp_{z_i} s_i (y^(l-1)_i)^T.

    Pperp_z s = (s - z <z,s>) / ||y||.  `mask` is the ReLU activation mask D^(l)
    (1[pre>0]); None in linear mode.  Shapes: z,z_pos,y [B,n]; y_prev [B,in];
    y_norm [B,1]; returns [n, in].
    """
    s, _ = signal(z, z_pos, tau)
    proj = (s - z * (z * s).sum(1, keepdim=True)) / y_norm
    if mask is not None:
        proj = proj * mask
    return (proj.t() @ y_prev) / tau


def _layer_reps(model, x, layer):
    """Recompute z^(layer) as a differentiable function of W[layer] only, with the
    layer input y^(layer-1) detached (layer-local, stop-grad between layers).
    Returns (z, y, y_prev, pre) all [B, .]."""
    ys = model(x)
    y_prev = ys[layer].detach()           # y^(layer-1)
    pre = y_prev @ model.W[layer].t()     # pre-activation
    y = model.act(pre)
    z = normalize(y)
    return z, y, y_prev, pre


def local_grad_autograd(model, x, x_pos, layer, tau):
    """grad_{W[layer]} g^(layer) via autograd of the detached-key goodness."""
    z, _, _, _ = _layer_reps(model, x, layer)
    zp, _, _, _ = _layer_reps(model, x_pos, layer)
    g = local_goodness(z, zp.detach(), tau)
    return torch.autograd.grad(g, model.W[layer])[0]


def local_grad(model, x, x_pos, layer, tau):
    """Closed-form local goodness gradient for `layer` (uses ReLU mask if needed)."""
    z, y, y_prev, pre = _layer_reps(model, x, layer)
    zp, _, _, _ = _layer_reps(model, x_pos, layer)
    y_norm = y.norm(dim=1, keepdim=True)
    mask = None if model.act_name == "linear" else (pre > 0).float()
    return local_grad_formula(z, zp.detach(), y_prev, y_norm, tau, mask=mask)


def global_grad(model, x, x_pos, layer, tau):
    """PROBE ONLY.  grad_{W[layer]} g^(L) -- final-layer goodness backprop'd to `layer`.
    One real backward through the whole network; never used to update weights."""
    ys = model(x)
    ys_pos = model(x_pos)
    zL = normalize(ys[-1])
    zLp = normalize(ys_pos[-1])
    g = local_goodness(zL, zLp, tau)
    return torch.autograd.grad(g, model.W[layer])[0]


def alignment_cosine(gl: torch.Tensor, gg: torch.Tensor) -> float:
    """A^(l) = cos angle between two gradient matrices (Frobenius)."""
    a = gl.flatten()
    b = gg.flatten()
    denom = a.norm() * b.norm()
    if denom.item() == 0.0:
        return 0.0
    return (a @ b / denom).item()
