"""Conv gen-FF: per-location denoising-energy early objective + GPU-clean EBM head trainer.
Spec: docs/superpowers/specs/2026-06-04-genff-conv-gpu-design.md."""
import torch, torch.nn as nn, torch.nn.functional as F


def _conv_branch(model, y_in, l):
    """The learnable part the block ADDS: relu(conv_l(y_in)). For a residual block the energy must be
    measured here, NOT on the residual output y + alpha*relu(conv(y)) — the latter is dominated by the
    passthrough y (which carries no gradient to conv_l), so the conv-weight signal would be ~O(alpha)
    tiny. Measuring the conv branch gives the filters a strong real-vs-noised gradient."""
    return F.relu(model.blocks[l](y_in))


def conv_denoise_step(model, x, cfg):
    """One forward-only local update per block: per-location energy G_{h,w}=mean_C (relu(conv y))^2
    of the CONV BRANCH trained HIGH on real x, LOW on noised x via the paired contrast
    -logsigmoid(G_real - G_noised). Layer-local (block input detached)."""
    sig, lr = cfg["sigma"], cfg["lr"]
    xn = x + sig * torch.randn_like(x)
    with torch.no_grad():
        ys = [y.detach() for y in model(x)]
        ysn = [y.detach() for y in model(xn)]
    for l in range(model.n_blocks):
        Gr = _conv_branch(model, ys[l], l).pow(2).mean(1)     # [B,H,W] conv-branch per-location energy
        Gn = _conv_branch(model, ysn[l], l).pow(2).mean(1)
        loss = -F.logsigmoid(Gr - Gn).mean()
        grads = torch.autograd.grad(loss, model.blocks[l].parameters())
        with torch.no_grad():
            for p, g in zip(model.blocks[l].parameters(), grads):
                p.add_(-lr * g)                               # descend the paired-contrast loss


def block_energy_gap(model, x, xn):
    """Sum over blocks of (mean real - mean noised) CONV-BRANCH energy (what the objective optimizes).
    Rises as denoising trains."""
    ys, ysn = model(x), model(xn)
    return sum((_conv_branch(model, ys[l], l).pow(2).mean()
                - _conv_branch(model, ysn[l], l).pow(2).mean()).item()
               for l in range(model.n_blocks))


def train_head_conv(model, head, X, y, cfg):
    """GPU-clean EBM head trainer (device-correct noise via randn_like). Loss = CE + lam * softplus(
    lse_noised - lse_real). Backbone frozen."""
    opt = torch.optim.Adam(head.parameters(), lr=cfg["lr"])
    ce = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(cfg["seed"])
    sig, lam = cfg["sigma"], cfg["lam"]
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(X), generator=g)
        for i in range(0, len(X) - cfg["batch"] + 1, cfg["batch"]):
            b = idx[i:i + cfg["batch"]]
            xb, yb = X[b], y[b]
            with torch.no_grad():
                fr = model.features(xb)
                fn = model.features(xb + sig * torch.randn_like(xb))
            logits = head(fr)
            lse_r = torch.logsumexp(logits, dim=1)
            lse_n = torch.logsumexp(head(fn), dim=1)
            loss = ce(logits, yb) + lam * F.softplus(lse_n - lse_r).mean()
            opt.zero_grad(); loss.backward(); opt.step()
