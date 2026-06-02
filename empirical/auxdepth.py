"""LoCo-style auxiliary-depth (look-ahead) local goodness for SCFF.

Vanilla SCFF trains block `layer` to maximize goodness on its OWN normalized output
z^(layer+1). Auxiliary-depth pushes block `layer`'s raw output y^(layer+1) through the
next `j` blocks (using their CURRENT weights, STOP-GRAD on those downstream weights)
then normalizes and computes goodness there. Only W[layer] receives a gradient, so each
weight's update is still local; but the objective "sees" `j` extra layers of downstream
context (the LoCo "overlap adds depth + implicit feedback" idea).

`j=0` recovers vanilla SCFF (arch.local_grad / gradients.local_grad).

Correctness contract:
  - layer input y^(layer) detached (layer-local, stop-grad between blocks; mirrors
    ArchMLP.block_output).
  - downstream block weights W[layer+1 .. layer+j] detached: they propagate y forward
    but receive NO gradient (verified in tests / experiment sanity check).
  - only W[layer] is differentiable; autograd wrt W[layer] gives the update.
"""
from __future__ import annotations
import torch

from model import normalize


def _block_apply(model, ys, l, W_l):
    """Recompute the l-th block output given the prefix list `ys` (ys[-1] = y^(l)) and an
    explicit weight matrix `W_l` for layer l. Mirrors ArchMLP._block but takes W_l so we
    can pass either the live (differentiable) W[layer] or a detached downstream weight."""
    if model.arch == "plain":
        return model.act(ys[-1] @ W_l.t())
    if model.arch == "residual":
        return ys[-1] + model.alpha * model.act(ys[-1] @ W_l.t())
    # dense: concat full prefix
    return model.act(torch.cat(ys, dim=1) @ W_l.t())


def aux_block_output(model, x, layer, j):
    """z measured `j` blocks downstream of `layer`, differentiable wrt W[layer] ONLY.

    y^(layer) and all earlier activations are detached. W[layer] is live (carries grad);
    downstream weights W[layer+1..layer+j] are detached (stop-grad). The result is
    normalize(y^(layer+1+j)).  j=0 reduces to model.block_output(x, layer).
    """
    j = min(j, model.n_layers - 1 - layer)   # clamp: cannot look past the last block
    with torch.no_grad():
        frozen = [y.detach() for y in model.forward(x)]   # full forward, all detached
    # prefix activations up to and including y^(layer), all detached
    ys = [f for f in frozen[:layer + 1]]
    # block `layer` with the LIVE weight -> differentiable wrt W[layer]
    out = _block_apply(model, ys, layer, model.W[layer])
    ys = ys + [out]
    # push through j downstream blocks with DETACHED weights (stop-grad on them)
    for k in range(layer + 1, layer + 1 + j):
        out = _block_apply(model, ys, k, model.W[k].detach())
        ys = ys + [out]
    return normalize(out)


def aux_local_grad(model, x, x_pos, layer, tau, j):
    """grad_{W[layer]} of the auxiliary-depth (look-ahead `j`) local goodness.

    Goodness is the InfoNCE goodness measured on z_lookahead = normalize(y pushed j blocks
    downstream). Keys/positive detached (same convention as gradients.local_goodness), so
    only W[layer] gets a gradient and only via its own block. j=0 == vanilla SCFF.
    """
    from gradients import local_goodness
    z = aux_block_output(model, x, layer, j)
    zp = aux_block_output(model, x_pos, layer, j)
    g = local_goodness(z, zp.detach(), tau)
    return torch.autograd.grad(g, model.W[layer])[0]


def aux_scff_step(model, x, x_pos, tau, lr, j):
    """One SCFF training step using auxiliary-depth look-ahead `j` for every block.

    Each W[layer] ascends ITS OWN auxiliary-depth goodness gradient (downstream weights
    stop-grad), so the update stays local. Gradients computed first (autograd needs grad
    enabled), then applied under no_grad. Mirrors e_arch_depth.scff_step_arch.
    """
    grads = [aux_local_grad(model, x, x_pos, l, tau, j) for l in range(model.n_layers)]
    with torch.no_grad():
        for l in range(model.n_layers):
            model.W[l].add_(lr * grads[l])
