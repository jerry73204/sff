import torch
from genff import GenFFMLP, train_early_denoise

def test_backbone_shapes():
    torch.manual_seed(0)
    m = GenFFMLP(d_in=20, width=16, n_layers=3)
    hs = m(torch.randn(8, 20))
    assert len(hs) == 3 and hs[-1].shape == (8, 16)
    assert m.features(torch.randn(8, 20)).shape == (8, 16 * 3)

def test_denoise_increases_manifold_gap():
    torch.manual_seed(0)
    m = GenFFMLP(d_in=20, width=16, n_layers=3)
    X = torch.randn(64, 20)
    def gap(model):
        hs_r = model(X); hs_n = model(X + 0.5 * torch.randn_like(X))
        gr = sum(h.pow(2).mean().item() for h in hs_r)
        gn = sum(h.pow(2).mean().item() for h in hs_n)
        return gr - gn
    g0 = gap(m)
    train_early_denoise(m, X, dict(epochs=30, batch=32, lr=0.02, sigma=0.5, theta=1.0, seed=0))
    assert gap(m) > g0

def test_denoise_is_layer_local():
    torch.manual_seed(0)
    m = GenFFMLP(d_in=20, width=16, n_layers=3)
    before = m.W[2].detach().clone()
    from genff import _layer_inputs
    hs = _layer_inputs(m, torch.randn(8, 20))
    h0 = torch.relu(hs[0] @ m.W[0].t())
    grads = torch.autograd.grad(h0.pow(2).mean(), [m.W[1], m.W[2]], allow_unused=True)
    assert all(g is None for g in grads)
