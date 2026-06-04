"""GPU CIFAR data module + metrics for the conv gen-FF pipeline."""
import os, sys
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.cifar_conv import load_cifar


def to_gpu(cfg, dev="cuda"):
    """Load CIFAR-10 and move all splits to `dev`. Returns (Xtr,ytr,Xte,yte)."""
    Xtr, ytr, Xte, yte = load_cifar(cfg)
    return Xtr.to(dev), ytr.to(dev), Xte.to(dev), yte.to(dev)


def augment_batch(x):
    """On-device batch augmentation: reflect-pad 4 + a random 32x32 crop + random h-flip
    (batch-level offsets)."""
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = int(torch.randint(0, 9, (1,))), int(torch.randint(0, 9, (1,)))
    out = xp[:, :, i:i + 32, j:j + 32]
    if torch.rand(1).item() < 0.5:
        out = torch.flip(out, dims=[3])
    return out.contiguous()


def ece(probs, y, bins=15):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    e, edges = 0.0, torch.linspace(0, 1, bins + 1)
    for j in range(bins):
        m = (conf > edges[j]) & (conf <= edges[j + 1])
        if m.any():
            e += (m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()).item()
    return e


def ood_auroc(fe_in, fe_ood):
    from sklearn.metrics import roc_auc_score
    s = torch.cat([fe_in.cpu(), fe_ood.cpu()]).numpy()
    lab = np.r_[np.zeros(len(fe_in)), np.ones(len(fe_ood))]
    return float(roc_auc_score(lab, s))
