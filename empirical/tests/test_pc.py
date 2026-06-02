"""Predictive-coding net: shapes + the key alignment-vs-settling behaviour."""
import torch
import pytest
from pc import PCNet

torch.set_default_dtype(torch.float64)


def _io(dims, B=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, dims[0], generator=g),
            torch.randn(B, dims[-1], generator=g))


def _cos(a, b):
    a, b = a.flatten(), b.flatten()
    return (a @ b / (a.norm() * b.norm())).item()


def test_shapes():
    dims = [6, 10, 10, 4]
    net = PCNet(dims, seed=1)
    x0, tgt = _io(dims)
    assert len(net.feedforward(x0)) == len(dims)
    for g, w in zip(net.bp_descent(x0, tgt), net.W):
        assert g.shape == w.shape
    for d, w in zip(net.pc_update(x0, tgt, T=5), net.W):
        assert d.shape == w.shape


def test_T0_only_output_aligns():
    """At T=0 (no settling) only the output layer carries error => aligns with BP;
    deeper layers get ~zero update."""
    dims = [6, 10, 10, 4]
    net = PCNet(dims, seed=2)
    x0, tgt = _io(dims, seed=3)
    bp = net.bp_descent(x0, tgt)
    pc = net.pc_update(x0, tgt, T=0)
    assert _cos(pc[-1], bp[-1]) > 0.99             # output layer aligned
    assert pc[0].abs().max().item() < 1e-9         # deepest layer: no update yet


def test_settling_propagates_alignment_to_deep_layer():
    """The deepest (furthest-from-output) layer is unaligned at T=0 but recovers the BP
    gradient once settling reaches it — settling = the cross-layer feedback propagating down."""
    dims = [8, 16, 16, 16, 6]                       # 4 weight layers
    net = PCNet(dims, seed=4)
    x0, tgt = _io(dims, seed=5)
    bp = net.bp_descent(x0, tgt)
    assert net.pc_update(x0, tgt, T=0)[0].abs().max().item() < 1e-9   # T=0: no update
    assert _cos(net.pc_update(x0, tgt, T=5, beta=0.1)[0], bp[0]) > 0.9  # settled: ≈ BP
