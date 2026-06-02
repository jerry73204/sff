import math
import torch
import pytest

from arch import ArchMLP
from gradients import global_grad, alignment_cosine
import arch as A

torch.set_default_dtype(torch.float64)


def _x(B=4, d=8, seed=0):
    return torch.randn(B, d, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("arch", ["plain", "residual", "dense"])
@pytest.mark.parametrize("L", [4, 8])
def test_forward_shapes(arch, L):
    m = ArchMLP(d_in=8, width=16, n_layers=L, arch=arch, act="linear", seed=1)
    ys = m(_x())
    assert len(ys) == L + 1
    for y in ys:
        assert y.shape == (4, 16)


def test_residual_alpha_zero_is_identity():
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch="residual", act="linear",
                alpha=0.0, seed=2)
    ys = m(_x())
    for l in range(1, len(ys)):
        assert torch.allclose(ys[l], ys[l - 1], atol=1e-12)


def test_forward_from_reconstructs():
    m = ArchMLP(d_in=8, width=16, n_layers=6, arch="dense", act="relu", seed=3)
    frozen = [y.detach() for y in m(_x())]
    for start in range(1, m.n_layers + 1):
        yL = m.forward_from(start, frozen[start], frozen)
        assert torch.allclose(yL, frozen[-1], atol=1e-10)


def test_block_output_differentiable_wrt_W():
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch="residual", act="linear", seed=4)
    z = m.block_output(_x(), layer=1)
    assert z.shape == (4, 16)
    g = torch.autograd.grad(z.sum(), m.W[1])[0]
    assert g.shape == m.W[1].shape and torch.isfinite(g).all()
    g0 = torch.autograd.grad(z.sum(), m.W[0], allow_unused=True)[0]
    assert g0 is None


@pytest.mark.parametrize("a", ["plain", "residual", "dense"])
def test_local_and_global_grad_shapes(a):
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch=a, act="linear", seed=5)
    x, xp = _x(), _x(seed=6)
    gl = A.local_grad(m, x, xp, layer=1, tau=0.5)
    gg = global_grad(m, x, xp, layer=1, tau=0.5)
    assert gl.shape == m.W[1].shape == gg.shape
    c = alignment_cosine(gl, gg)
    assert -1.0 - 1e-9 <= c <= 1.0 + 1e-9


def test_last_block_is_self_aligned():
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch="plain", act="linear", seed=7)
    x, xp = _x(), _x(seed=8)
    last = m.n_layers - 1
    gl = A.local_grad(m, x, xp, layer=last, tau=0.5)
    gg = global_grad(m, x, xp, layer=last, tau=0.5)
    assert alignment_cosine(gl, gg) == pytest.approx(1.0, abs=1e-6)


def test_downstream_jacobian_matches_linear_product_plain():
    m = ArchMLP(d_in=8, width=10, n_layers=4, arch="plain", act="linear", seed=9)
    x = _x()
    for layer in range(m.n_layers):
        M = A.downstream_jacobian(m, x, layer)
        expect = torch.eye(m.width)
        for k in range(layer + 1, m.n_layers):
            expect = m.W[k] @ expect
        assert torch.allclose(M, expect, atol=1e-8), f"layer {layer}"


def test_downstream_jacobian_identity_last_block():
    m = ArchMLP(d_in=8, width=10, n_layers=4, arch="dense", act="relu", seed=10)
    M = A.downstream_jacobian(m, _x(), layer=m.n_layers - 1)
    assert torch.allclose(M, torch.eye(m.width), atol=1e-10)


@pytest.mark.parametrize("a", ["plain", "residual", "dense"])
def test_measure_blocks(a):
    m = ArchMLP(d_in=8, width=32, n_layers=4, arch=a, act="linear", seed=11)
    rows = A.measure_blocks(m, _x(), _x(seed=12), tau=0.5)
    assert len(rows) == m.n_layers
    for (l, Aval, dg, an, dV) in rows:
        assert -1.0 - 1e-9 <= Aval <= 1.0 + 1e-9
        assert dg >= 0.0 and an >= 0.0 and dV >= 0


def test_norm_default_off_is_unchanged():
    """norm defaults to False and reproduces the un-normalized forward exactly."""
    m0 = ArchMLP(d_in=8, width=16, n_layers=4, arch="plain", act="linear", seed=1)
    m1 = ArchMLP(d_in=8, width=16, n_layers=4, arch="plain", act="linear", seed=1,
                 norm=False)
    assert m0.norm is False
    for y0, y1 in zip(m0(_x()), m1(_x())):
        assert torch.allclose(y0, y1, atol=1e-12)


def test_norm_layernorm_standardizes_block_output():
    """With norm=True each block output is mean~0, std~1 per sample (parameter-free LN)."""
    m = ArchMLP(d_in=8, width=64, n_layers=4, arch="plain", act="linear", seed=2,
                norm=True)
    ys = m(_x())
    for y in ys[1:]:                       # blocks (stem ys[0] is not LN'd)
        assert y.mean(dim=1).abs().max().item() < 1e-6
        assert (y.std(dim=1, unbiased=False) - 1.0).abs().max().item() < 1e-3


def test_norm_local_grad_finite_and_shaped():
    """LayerNorm local goodness grad (autograd) is finite and W-shaped, all archs/acts."""
    x, xp = _x(), _x(seed=20)
    for arch in ("plain", "residual", "dense"):
        for act in ("linear", "relu"):
            m = ArchMLP(d_in=8, width=16, n_layers=4, arch=arch, act=act, seed=3,
                        norm=True)
            gl = A.local_grad(m, x, xp, layer=1, tau=0.5)
            assert gl.shape == m.W[1].shape and torch.isfinite(gl).all()


def test_norm_local_grad_matches_independent_autograd():
    """Anchor: harness autograd local grad == an independently-built autograd grad of
    the same LN block-local goodness, to 1e-10 (linear)."""
    from gradients import local_goodness
    from model import normalize
    x, xp = _x(), _x(seed=21)
    m = ArchMLP(d_in=8, width=16, n_layers=4, arch="plain", act="linear", seed=4,
                norm=True)
    layer, tau = 1, 0.5
    with torch.no_grad():
        ys = [y.detach() for y in m(x)]
        ysp = [y.detach() for y in m(xp)]
    W = m.W[layer]
    z = normalize(m._ln(ys[layer] @ W.t()))
    with torch.no_grad():
        zp = normalize(m._ln(ysp[layer] @ W.t()))
    gref = torch.autograd.grad(local_goodness(z, zp.detach(), tau), W)[0]
    gl = A.local_grad(m, x, xp, layer, tau)
    assert torch.allclose(gl, gref, atol=1e-10)
