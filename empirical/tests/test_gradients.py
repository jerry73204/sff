"""math<->code anchor (design.md M1): closed-form local goodness gradient must match
autograd in linear mode to 1e-5; mu-P init variances; metric sanity."""
import math
import torch
import pytest

from model import MLP, normalize
from gradients import (local_grad, local_grad_autograd, global_grad,
                       alignment_cosine, signal, softmax_weights)
import metrics


torch.set_default_dtype(torch.float64)  # tight tolerance needs float64


def _batch(d_in=6, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, d_in, generator=g)
    x_pos = x + 0.1 * torch.randn(B, d_in, generator=g)
    return x, x_pos


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_local_grad_matches_autograd_linear(layer):
    """THE anchor: closed-form == autograd, linear mode, < 1e-5."""
    model = MLP(d_in=6, width=10, n_layers=3, act="linear", seed=1)
    x, x_pos = _batch()
    gf = local_grad(model, x, x_pos, layer, tau=0.5)
    ga = local_grad_autograd(model, x, x_pos, layer, tau=0.5)
    assert torch.allclose(gf, ga, atol=1e-5, rtol=1e-5), \
        f"layer {layer}: max abs diff {(gf - ga).abs().max().item():.2e}"


def test_local_grad_matches_autograd_relu():
    """ReLU mode: closed-form (with activation mask) == autograd."""
    model = MLP(d_in=6, width=10, n_layers=3, act="relu", seed=2)
    x, x_pos = _batch(seed=3)
    for layer in range(3):
        gf = local_grad(model, x, x_pos, layer, tau=0.5)
        ga = local_grad_autograd(model, x, x_pos, layer, tau=0.5)
        assert torch.allclose(gf, ga, atol=1e-5, rtol=1e-5), \
            f"relu layer {layer}: max diff {(gf - ga).abs().max().item():.2e}"


def test_mup_init_variance():
    """Hidden weights var ~ 1/fan_in; last layer var ~ 1/fan_in^2 (THEORY.md sec 1)."""
    d_in, width, L = 64, 4096, 3
    model = MLP(d_in=d_in, width=width, n_layers=L, act="linear", seed=7)
    # hidden layer 1 (fan_in = width)
    v1 = model.W[1].var().item()
    assert v1 == pytest.approx(1.0 / width, rel=0.1)
    # last layer: extra 1/fan_in -> var 1/fan_in^2
    vL = model.W[L - 1].var().item()
    assert vL == pytest.approx(1.0 / width ** 2, rel=0.1)


def test_global_grad_is_probe():
    """Global probe returns a grad of the right shape and does not mutate weights."""
    model = MLP(d_in=6, width=10, n_layers=3, act="linear", seed=4)
    x, x_pos = _batch(seed=5)
    before = [w.detach().clone() for w in model.W]
    gg = global_grad(model, x, x_pos, layer=0, tau=0.5)
    assert gg.shape == model.W[0].shape
    for w, b in zip(model.W, before):
        assert torch.equal(w.detach(), b), "probe must not change weights"


def test_alignment_cosine_range_and_parallel():
    a = torch.randn(5, 7)
    assert alignment_cosine(a, a) == pytest.approx(1.0, abs=1e-9)
    assert alignment_cosine(a, -a) == pytest.approx(-1.0, abs=1e-9)
    b = torch.randn(5, 7)
    c = alignment_cosine(a, b)
    assert -1.0 - 1e-9 <= c <= 1.0 + 1e-9


def test_softmax_weights_rows_sum_to_one():
    z = normalize(torch.randn(8, 10))
    p = softmax_weights(z, tau=0.5)
    assert torch.allclose(p.sum(1), torch.ones(8), atol=1e-10)


def test_delta_gram_zero_for_equal_reps():
    z = normalize(torch.randn(8, 10))
    assert metrics.delta_gram(z, z) == pytest.approx(0.0, abs=1e-9)


def test_aniso_zero_for_isotropic():
    """Aniso = 0 when M^T M is a scalar multiple of identity on V."""
    n, dV = 12, 4
    M = math.sqrt(3.0) * torch.eye(n)        # M^T M = 3 I
    V, _ = metrics.contrastive_subspace(torch.randn(dV, n))
    assert metrics.aniso(M, V) == pytest.approx(0.0, abs=1e-9)
