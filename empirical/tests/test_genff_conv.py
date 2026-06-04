import torch, pytest
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

@cuda_only
def test_conv_features_shape():
    from gpu_arch import ConvSCFF
    m = ConvSCFF(C=32, n_blocks=4, arch="residual", alpha=0.3).cuda()
    f = m.features(torch.randn(8, 3, 32, 32, device="cuda"))
    assert f.shape == (8, 32 * 4)

@cuda_only
def test_conv_denoise_raises_energy_gap():
    from gpu_arch import ConvSCFF
    from genff_conv import conv_denoise_step, block_energy_gap
    torch.manual_seed(0)
    m = ConvSCFF(C=32, n_blocks=3, arch="residual", alpha=0.3).cuda()
    X = torch.randn(64, 3, 32, 32, device="cuda")
    def gap():
        return block_energy_gap(m, X, X + 0.3 * torch.randn_like(X))
    g0 = gap()
    for _ in range(40):
        conv_denoise_step(m, X, dict(sigma=0.3, lr=1.0))
    assert gap() > g0

@cuda_only
def test_conv_denoise_is_layer_local():
    from gpu_arch import ConvSCFF
    torch.manual_seed(0)
    m = ConvSCFF(C=32, n_blocks=3, arch="residual", alpha=0.3).cuda()
    ys = [y.detach() for y in m(torch.randn(8, 3, 32, 32, device="cuda"))]
    g = m.apply_block(ys[0], 0).pow(2).mean()
    grads = torch.autograd.grad(g, list(m.blocks[1].parameters()), allow_unused=True)
    assert all(gp is None for gp in grads)
