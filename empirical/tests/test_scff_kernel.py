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
