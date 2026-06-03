"""CIFAR-10 conv credibility test: does the residual-SCFF alignment win transfer to convolutions?

Replicates the MLP/MNIST story (task_accuracy.py) at conv scale:
  supervised-BP   conv net + linear head, cross-entropy, backprop (upper bound)
  plain-SCFF      conv, per-block local InfoNCE goodness, forward-only (the gap)
  residual-SCFF   conv + residual blocks, same local rule (the alignment fix)
Linear probe on concatenated pooled block features. Reports accuracy + mean alignment A.

Prediction (from MLP + depth-scaling): residual-SCFF >> plain-SCFF, approaching supervised-BP.

Data: CIFAR-10 via direct download (torchvision ABI-broken in this env). float32 (conv on CPU).
Run: python experiments/cifar_conv.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convarch import ConvSCFF, local_grad, global_grad, flat_cos
from gradients import local_goodness

CFG = dict(C=64, n_blocks=4, arch_alpha=0.2, tau=0.5, batch=64, epochs=8,
           lr_scff=0.05, lr_bp=1e-3, aug_noise=0.1, n_train=8000, n_test=2000, seed=0)


def load_cifar(cfg):
    import urllib.request, tarfile, pickle
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    d = "/tmp/cifar"; os.makedirs(d, exist_ok=True)
    base = d + "/cifar-10-batches-py"
    if not os.path.exists(base + "/test_batch"):
        tgz = d + "/cifar.tar.gz"
        if not os.path.exists(tgz):
            urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(d)

    def load(f):
        with open(f, "rb") as fo:
            dd = pickle.load(fo, encoding="bytes")
        return np.asarray(dd[b"data"]), np.asarray(dd[b"labels"])

    Xtr, Ytr = [], []
    for i in range(1, 6):
        X, Y = load(f"{base}/data_batch_{i}"); Xtr.append(X); Ytr.append(Y)
    Xtr = np.concatenate(Xtr); Ytr = np.concatenate(Ytr)
    Xte, Yte = load(f"{base}/test_batch")
    rng = np.random.default_rng(cfg["seed"])
    itr = rng.permutation(len(Xtr))[:cfg["n_train"]]
    ite = rng.permutation(len(Xte))[:cfg["n_test"]]

    def to(X):
        return torch.tensor(X.reshape(-1, 3, 32, 32).astype("float32") / 255.0)

    Xtr_t, Xte_t = to(Xtr[itr]), to(Xte[ite])
    mean = Xtr_t.mean(dim=(0, 2, 3), keepdim=True)
    std = Xtr_t.std(dim=(0, 2, 3), keepdim=True)
    Xtr_t = (Xtr_t - mean) / std; Xte_t = (Xte_t - mean) / std
    return (Xtr_t, torch.tensor(Ytr[itr]).long(), Xte_t, torch.tensor(Yte[ite]).long())


def augment(x, noise):
    """Positive view: per-sample random hflip + Gaussian noise."""
    flip = torch.rand(x.shape[0]) < 0.5
    x = x.clone()
    x[flip] = torch.flip(x[flip], dims=[3])
    return x + noise * torch.randn_like(x)


def features(model, X, bs=500):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ys = model(X[i:i + bs])
            outs.append(torch.cat([model.pooled(ys[l]) for l in range(1, model.n_blocks + 1)], 1))
    return torch.cat(outs).numpy()


def probe(model, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300, C=1.0).fit(features(model, Xtr), ytr.numpy())
    return float((clf.predict(features(model, Xte)) == yte.numpy()).mean())


def batches(n, bs, gen):
    idx = torch.randperm(n, generator=gen)
    for i in range(0, n - bs + 1, bs):
        yield idx[i:i + bs]


def train_scff(model, Xtr, cfg, gen):
    for _ in range(cfg["epochs"]):
        for b in batches(len(Xtr), cfg["batch"], gen):
            xb = Xtr[b]; xp = augment(xb, cfg["aug_noise"])
            grads = [local_grad(model, xb, xp, l, cfg["tau"]) for l in range(model.n_blocks)]
            with torch.no_grad():
                for l in range(model.n_blocks):
                    for p, g in zip(model.blocks[l].parameters(), grads[l]):
                        p.add_(cfg["lr_scff"] * g)               # ascend local goodness


def train_bp(model, Xtr, ytr, cfg, gen, n_classes):
    head = torch.nn.Linear(model.C, n_classes)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=cfg["lr_bp"])
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(cfg["epochs"]):
        for b in batches(len(Xtr), cfg["batch"], gen):
            logits = head(model.pooled(model(Xtr[b])[-1]))
            loss = lossf(logits, ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(model.pooled(model(Xte_cache[0])[-1])).argmax(1)
    return float((pred == Xte_cache[1]).float().mean())


def mean_A(model, Xte, cfg):
    x = Xte[:cfg["batch"]]; xp = augment(x, cfg["aug_noise"])
    vals = [flat_cos(local_grad(model, x, xp, l, cfg["tau"]),
                     global_grad(model, x, xp, l, cfg["tau"]))
            for l in range(model.n_blocks - 1)]
    return sum(vals) / len(vals)


Xte_cache = (None, None)


def main():
    global Xte_cache
    torch.manual_seed(0)
    Xtr, ytr, Xte, yte = load_cifar(CFG)
    Xte_cache = (Xte, yte)
    n_classes = 10
    C, L = CFG["C"], CFG["n_blocks"]
    print(f"data=CIFAR-10  train={len(Xtr)} test={len(Xte)}  conv C={C} blocks={L} "
          f"(16x16 after stem)  epochs={CFG['epochs']}\n")

    def g():
        return torch.Generator().manual_seed(CFG["seed"])

    rows = []
    # supervised-BP (upper bound)
    m = ConvSCFF(C, L, "plain", seed=CFG["seed"])
    head_acc = train_bp(m, Xtr, ytr, CFG, g(), n_classes)
    rows.append(("supervised-BP", probe(m, Xtr, ytr, Xte, yte), float("nan")))
    print(f"supervised-BP   probe={rows[-1][1]:.4f}  head={head_acc:.4f}", flush=True)
    # plain-SCFF
    m = ConvSCFF(C, L, "plain", seed=CFG["seed"])
    train_scff(m, Xtr, CFG, g())
    rows.append(("plain-SCFF", probe(m, Xtr, ytr, Xte, yte), mean_A(m, Xte, CFG)))
    print(f"plain-SCFF      probe={rows[-1][1]:.4f}  A={rows[-1][2]:.3f}", flush=True)
    # residual-SCFF
    m = ConvSCFF(C, L, "residual", alpha=CFG["arch_alpha"], seed=CFG["seed"])
    train_scff(m, Xtr, CFG, g())
    rows.append(("residual-SCFF", probe(m, Xtr, ytr, Xte, yte), mean_A(m, Xte, CFG)))
    print(f"residual-SCFF   probe={rows[-1][1]:.4f}  A={rows[-1][2]:.3f}", flush=True)

    print("\n=== VERDICT (CIFAR-10 linear probe) ===")
    acc = dict((r[0], r[1]) for r in rows)
    for name, a, al in rows:
        extra = f"  A={al:.3f}" if al == al else ""
        print(f"  {name:14s} probe={a:.4f}{extra}")
    print(f"\nalignment fix (residual-SCFF - plain-SCFF): {acc['residual-SCFF']-acc['plain-SCFF']:+.4f}")
    print(f"gap to supervised-BP: {acc['supervised-BP']-acc['residual-SCFF']:+.4f}")
    print("=> " + ("conv: alignment fix TRANSFERS (residual >> plain, near BP)"
                   if acc['residual-SCFF'] - acc['plain-SCFF'] > 0.03 else
                   "conv: alignment fix weaker than MLP -- investigate"))


if __name__ == "__main__":
    main()
