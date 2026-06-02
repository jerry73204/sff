"""Correctness contract for the LoCo-style auxiliary-depth look-ahead goodness:
  - j=0 reproduces vanilla SCFF local_grad exactly.
  - downstream block weights receive NO gradient (stop-grad); only W[layer] updates.
"""
import torch
import pytest

from arch import ArchMLP, local_grad
from auxdepth import aux_local_grad, aux_block_output
from gradients import local_goodness

torch.set_default_dtype(torch.float64)


def _data(B=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, d, generator=g)
    xp = x + 0.1 * torch.randn(x.shape, generator=g)
    return x, xp


@pytest.mark.parametrize("arch", ["plain", "residual", "dense"])
def test_j0_reproduces_vanilla(arch):
    m = ArchMLP(8, 16, 5, arch, "linear", alpha=0.1, seed=1)
    x, xp = _data(d=8)
    for l in range(m.n_layers):
        a = aux_local_grad(m, x, xp, l, 0.5, 0)
        b = local_grad(m, x, xp, l, 0.5)
        assert torch.allclose(a, b, atol=1e-10), f"{arch} layer {l}"


@pytest.mark.parametrize("arch", ["plain", "residual"])
@pytest.mark.parametrize("j", [1, 2])
def test_downstream_weights_get_no_grad(arch, j):
    m = ArchMLP(8, 16, 6, arch, "linear", alpha=0.1, seed=2)
    x, xp = _data(d=8)
    layer = 1
    z = aux_block_output(m, x, layer, j)
    zp = aux_block_output(m, xp, layer, j)
    g = local_goodness(z, zp.detach(), 0.5)
    grads = torch.autograd.grad(g, list(m.W), allow_unused=True)
    for l, gr in enumerate(grads):
        norm = 0.0 if gr is None else gr.norm().item()
        if l == layer:
            assert norm > 0, "target layer must receive gradient"
        else:
            assert norm == 0.0, f"downstream/other W[{l}] must get NO gradient"


def test_clamp_past_last_block():
    # look-ahead beyond the last block is clamped, not an error
    m = ArchMLP(8, 16, 4, "plain", "linear", seed=3)
    x, _ = _data(d=8)
    z_clamped = aux_block_output(m, x, m.n_layers - 1, 5)   # j huge at last block
    z_vanilla = m.block_output(x, m.n_layers - 1)
    assert torch.allclose(z_clamped, z_vanilla, atol=1e-10)
