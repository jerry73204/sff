"""Pure-torch reference for the scff_signal CUDA kernel (correctness oracle)."""
import torch

def signal_ref(z, z_pos, tau):
    """s_perp = P_perp_z( z_pos - softmax(z z^T / tau) @ z ); goodness scalar. Matches
    gradients.signal + tangent projection and gradients.local_goodness."""
    scores = (z @ z.t()) / tau
    p = torch.softmax(scores, dim=1)
    s = z_pos - p @ z
    s_perp = s - z * (z * s).sum(1, keepdim=True)
    good = ((z * z_pos).sum(1) / tau - torch.logsumexp(scores, dim=1)).sum()
    return s_perp, good
