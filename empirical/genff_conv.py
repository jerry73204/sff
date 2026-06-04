"""Conv gen-FF: per-location denoising-energy early objective + GPU-clean EBM head trainer.
Spec: docs/superpowers/specs/2026-06-04-genff-conv-gpu-design.md."""
import torch, torch.nn as nn, torch.nn.functional as F


def conv_denoise_step(model, x, cfg):
    """One forward-only local update per block: per-location energy G_{h,w}=mean_C h^2 trained HIGH
    on real x, LOW on noised x via the paired contrast -logsigmoid(G_real - G_noised). Layer-local
    (block input detached). Both G_real and G_noised carry grad wrt block l."""
    sig, lr = cfg["sigma"], cfg["lr"]
    xn = x + sig * torch.randn_like(x)
    with torch.no_grad():
        ys = [y.detach() for y in model(x)]
        ysn = [y.detach() for y in model(xn)]
    for l in range(model.n_blocks):
        Gr = model.apply_block(ys[l], l).pow(2).mean(1)
        Gn = model.apply_block(ysn[l], l).pow(2).mean(1)
        loss = -F.logsigmoid(Gr - Gn).mean()
        grads = torch.autograd.grad(loss, model.blocks[l].parameters())
        with torch.no_grad():
            for p, g in zip(model.blocks[l].parameters(), grads):
                p.add_(-lr * g)


def block_energy_gap(model, x, xn):
    """Sum over blocks of (mean real energy - mean noised energy). Rises as denoising trains."""
    ys, ysn = model(x), model(xn)
    return sum((ys[l].pow(2).mean() - ysn[l].pow(2).mean()).item()
               for l in range(1, model.n_blocks + 1))


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
