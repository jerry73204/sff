import torch, pytest
from cuda.reference import signal_ref

def test_reference_matches_gradients_module():
    from gradients import signal
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)
    zp = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)
    s_ref, g_ref = signal_ref(z, zp, 0.5)
    s_grad, _ = signal(z, zp, 0.5)
    s_grad_perp = s_grad - z * (z * s_grad).sum(1, keepdim=True)
    assert torch.allclose(s_ref, s_grad_perp, atol=1e-5)
    assert s_ref.shape == (16, 32)
    assert torch.allclose((s_ref * z).sum(1), torch.zeros(16), atol=1e-5)


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

@cuda_only
def test_kernel_matches_reference():
    from cuda.scff_ext import scff_signal
    from cuda.reference import signal_ref
    torch.manual_seed(1)
    for B, C in [(16, 32), (64, 128), (128, 256)]:
        z = torch.nn.functional.normalize(torch.randn(B, C, device="cuda"), dim=1)
        zp = torch.nn.functional.normalize(torch.randn(B, C, device="cuda"), dim=1)
        s_ker, g_ker = scff_signal(z, zp, 0.5)
        s_ref, g_ref = signal_ref(z, zp, 0.5)
        assert torch.allclose(s_ker, s_ref, atol=1e-4), (B, C, (s_ker - s_ref).abs().max())
        assert abs(float(g_ker) - float(g_ref)) < 1e-2 * max(1.0, abs(float(g_ref)))


@cuda_only
def test_gpu_train_step_improves_goodness():
    from gpu_arch import ConvSCFF, scff_local_step, block_goodness
    torch.manual_seed(0)
    m = ConvSCFF(C=32, n_blocks=3, arch="residual", alpha=0.2).cuda()
    x = torch.randn(64, 3, 32, 32, device="cuda")
    xp = x + 0.1 * torch.randn_like(x)
    g0 = block_goodness(m, x, xp, tau=0.5)
    for _ in range(20):
        scff_local_step(m, x, xp, tau=0.5, lr=0.1)
    g1 = block_goodness(m, x, xp, tau=0.5)
    assert g1 > g0
