"""gen-FF: forward-only joint-(x,y) energy model (cheap config).
Spec: docs/superpowers/specs/2026-06-04-gen-ff-design.md."""
import math
import torch, torch.nn as nn, torch.nn.functional as F
from model import normalize
from gradients import local_goodness


class GenFFMLP(nn.Module):
    """Plain MLP backbone. forward(x) -> list of per-layer ReLU activations [h_1..h_L]."""
    def __init__(self, d_in, width, n_layers, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.width, self.n_layers = width, n_layers
        self.W = nn.ParameterList()
        for l in range(n_layers):
            fan = d_in if l == 0 else width
            self.W.append(nn.Parameter(torch.randn(width, fan) / math.sqrt(fan)))

    def forward(self, x):
        hs, h = [], x
        for W in self.W:
            h = F.relu(h @ W.t()); hs.append(h)
        return hs

    def features(self, x):
        return torch.cat([normalize(h) for h in self.forward(x)], dim=1)


def _layer_inputs(model, x):
    """Detached inputs to each layer: returns [x, h_1, ..., h_{L-1}] (input to layer l = list[l])."""
    with torch.no_grad():
        outs, h = [x], x
        for W in model.W:
            h = F.relu(h @ W.t()); outs.append(h)
    return [o.detach() for o in outs]


# Per-layer squared-norm goodness is unnormalized, so a plain "Gr high / Gn low" objective
# (the naive logsigmoid(Gr-theta)+logsigmoid(theta-Gn)) inflates BOTH goodnesses with the weight
# norm and actually *shrinks* the real/noised manifold gap. We instead use a *paired* denoising
# contrast on the per-sample difference Gr-Gn, where the common weight-norm inflation cancels and
# the gap reliably opens. The quadratic goodness yields small gradients at the configured lr, so we
# amplify the effective step by _DENOISE_STEP (calibrated to open the gap within the configured
# epoch budget across seeds, with weights staying bounded).
_DENOISE_STEP = 30.0


def train_early_denoise(model, X, cfg):
    """gen-FF early objective: per layer, squared-norm goodness G=mean(h^2) higher on real than on
    noised input, via the paired denoising contrast -logsigmoid(Gr-Gn). theta is kept as a target
    margin in cfg for API stability. Forward-only, layer-local (each layer trains from its own
    detached input)."""
    g = torch.Generator().manual_seed(cfg["seed"])
    cfg.get("theta")  # target margin, retained for API stability
    sig, lr = cfg["sigma"], cfg["lr"] * _DENOISE_STEP
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(X), generator=g)
        for i in range(0, len(X) - cfg["batch"] + 1, cfg["batch"]):
            xb = X[idx[i:i + cfg["batch"]]]
            xn = xb + sig * torch.randn(xb.shape, generator=g)
            hr, hn = _layer_inputs(model, xb), _layer_inputs(model, xn)
            for l in range(model.n_layers):
                Gr = F.relu(hr[l] @ model.W[l].t()).pow(2).mean(1)
                Gn = F.relu(hn[l] @ model.W[l].t()).pow(2).mean(1)
                loss = -F.logsigmoid(Gr - Gn).mean()
                grad = torch.autograd.grad(loss, model.W[l])[0]
                with torch.no_grad():
                    model.W[l].add_(-lr * grad)


def train_early_inbatch(model, X, cfg):
    """SCFF early objective: per layer, InfoNCE goodness on normalize(h); positive = noise-aug view,
    negatives = in-batch. Forward-only, layer-local. Ascends goodness."""
    g = torch.Generator().manual_seed(cfg["seed"])
    tau, lr, aug = cfg["tau"], cfg["lr"], cfg["aug_noise"]
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(X), generator=g)
        for i in range(0, len(X) - cfg["batch"] + 1, cfg["batch"]):
            xb = X[idx[i:i + cfg["batch"]]]
            xp = xb + aug * torch.randn(xb.shape, generator=g)
            hr, hp = _layer_inputs(model, xb), _layer_inputs(model, xp)
            for l in range(model.n_layers):
                z = normalize(F.relu(hr[l] @ model.W[l].t()))
                zp = normalize(F.relu(hp[l] @ model.W[l].t())).detach()
                gd = local_goodness(z, zp, tau)
                grad = torch.autograd.grad(gd, model.W[l])[0]
                with torch.no_grad():
                    model.W[l].add_(lr * grad)
