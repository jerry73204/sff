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

def test_energy_head_and_predict():
    from genff import GenFFMLP, EnergyHead, predict
    torch.manual_seed(0)
    m = GenFFMLP(d_in=20, width=16, n_layers=3)
    head = EnergyHead(16 * 3, n_classes=5)
    X = torch.randn(8, 20)
    logits = head(m.features(X))
    assert logits.shape == (8, 5)
    pred = predict(m, head, X)
    assert pred.shape == (8,) and int(pred.min()) >= 0 and int(pred.max()) < 5

def test_head_training_separates_ood_energy():
    from genff import GenFFMLP, EnergyHead, train_head, train_early_denoise, free_energy
    torch.manual_seed(0)
    m = GenFFMLP(d_in=20, width=16, n_layers=3)
    X = torch.randn(256, 20); y = torch.randint(0, 5, (256,))
    train_early_denoise(m, X, dict(epochs=20, batch=64, lr=0.02, sigma=0.5, theta=1.0, seed=0))
    head = EnergyHead(16 * 3, 5)
    train_head(m, head, X, y, dict(epochs=40, batch=64, lr=1e-2, lam=0.1, sigma=0.5, seed=0))
    fe_real = free_energy(m, head, X).mean().item()
    fe_ood = free_energy(m, head, 3.0 * torch.randn(256, 20)).mean().item()
    assert fe_ood > fe_real
