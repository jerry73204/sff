"""JIT-compiled CUDA scff_signal kernel."""
import os, torch
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_ext = None

def _get_ext():
    global _ext
    if _ext is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        _ext = load(name="scff_cuda", sources=[os.path.join(_HERE, "scff_signal.cu")], verbose=False)
    return _ext

def scff_signal(z, z_pos, tau):
    """s_perp [B,C] = P_perp_z(z_pos - softmax(z z^T/tau) z); goodness scalar. float32 CUDA only.
    Requires B <= 256 and C <= 512 (shared-mem budget)."""
    assert z.is_cuda and z.dtype == torch.float32, "float32 CUDA tensor required"
    z = z.contiguous(); z_pos = z_pos.contiguous()
    B, C = z.shape
    assert B <= 256 and C <= 512, f"kernel shared-mem budget exceeded (B={B},C={C}); use reference"
    s_perp = torch.empty_like(z)
    goodness = torch.zeros(1, device=z.device, dtype=torch.float32)
    _get_ext().scff_signal(z, z_pos, s_perp, goodness, float(tau), B, C)
    return s_perp, goodness[0]
