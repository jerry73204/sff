"""Device-aware conv SCFF with a forward-only local step driven by the scff_signal kernel."""
import torch, torch.nn as nn, torch.nn.functional as F

def _pooled(y):
    return F.normalize(y.mean(dim=(2, 3)), dim=1)        # [B,C] on the unit sphere

class ConvSCFF(nn.Module):
    def __init__(self, C=64, n_blocks=8, arch="residual", alpha=0.2, in_ch=3):
        super().__init__()
        assert arch in ("plain", "residual")
        self.arch, self.alpha, self.C, self.n_blocks = arch, alpha, C, n_blocks
        self.stem = nn.Conv2d(in_ch, C, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([nn.Conv2d(C, C, 3, padding=1) for _ in range(n_blocks)])

    def apply_block(self, y, l):
        h = F.relu(self.blocks[l](y))
        return y + self.alpha * h if self.arch == "residual" else h

    def forward(self, x):
        ys = [F.relu(self.stem(x))]
        for l in range(self.n_blocks):
            ys.append(self.apply_block(ys[-1], l))
        return ys

    def pooled(self, y):
        return _pooled(y)

def block_goodness(model, x, xp, tau):
    """Sum of per-block local goodness (kernel's scalar), for tests/logging."""
    from cuda.scff_ext import scff_signal
    ys, ysp = model(x), model(xp)
    tot = 0.0
    for l in range(model.n_blocks):
        _, g = scff_signal(_pooled(ys[l + 1]), _pooled(ysp[l + 1]), tau)
        tot += float(g)
    return tot

def scff_local_step(model, x, xp, tau, lr):
    """One forward-only local update for every block. Per block: kernel emits s_perp on the pooled
    rep; backprop s_perp through THIS block only (input detached) -> grad of conv params; ascend."""
    from cuda.scff_ext import scff_signal
    with torch.no_grad():
        ys = [y.detach() for y in model(x)]
        ysp = [y.detach() for y in model(xp)]
    for l in range(model.n_blocks):
        y_in = ys[l].requires_grad_(False)
        out = model.apply_block(y_in, l)                 # differentiable wrt block l only
        z = _pooled(out)
        s_perp, _ = scff_signal(z.detach(), _pooled(ysp[l + 1]).detach(), tau)
        grads = torch.autograd.grad(z, model.blocks[l].parameters(), grad_outputs=s_perp)
        with torch.no_grad():
            for p, gp in zip(model.blocks[l].parameters(), grads):
                p.add_(lr * gp)                          # + : ascend (s_perp is +dg/dz direction)
