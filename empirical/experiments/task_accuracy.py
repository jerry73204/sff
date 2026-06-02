"""Practical task test (FINDINGS gap #1): does the SCFF↔BP alignment win translate into a
TRAINED-MODEL win on real data?

Same architecture, three training rules, self-supervised contrastive pretraining, then a linear
probe on labels:
  - plain-BP      : backprop the global contrastive loss end-to-end (gold standard, plain arch)
  - plain-SCFF    : per-layer local goodness, no backprop (the gap)
  - residual-SCFF : per-layer local goodness, residual arch (the alignment fix)

Positives = noise-augmented views; negatives = in-batch (InfoNCE on normalized reps). Probe =
logistic regression on the frozen concatenated block features. We also report the mean
gradient-alignment A so we can correlate alignment → accuracy.

Prediction: plain-SCFF << plain-BP (the cross-layer gap); residual-SCFF closes toward BP.

Run: python experiments/task_accuracy.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import MLP, normalize
from arch import ArchMLP
from gradients import local_goodness, global_grad, alignment_cosine
import arch as A

CFG = dict(width=256, n_layers=4, act="relu", tau=0.5, batch=64, epochs=12,
           lr_bp=0.05, lr_scff=0.05, alpha=0.1, aug_noise=0.3,
           n_train=6000, n_test=1000, seed=0)


def _load_raw():
    """MNIST via direct idx download (S3 mirror); fall back to sklearn digits."""
    import urllib.request, gzip
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    try:
        os.makedirs("/tmp/mnistraw", exist_ok=True)
        def grab(fn):
            p = "/tmp/mnistraw/" + fn
            if not os.path.exists(p):
                urllib.request.urlretrieve(base + fn, p)
            return p
        with gzip.open(grab("train-images-idx3-ubyte.gz")) as f:
            X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784).astype("float64") / 255.0
        with gzip.open(grab("train-labels-idx1-ubyte.gz")) as f:
            y = np.frombuffer(f.read(), np.uint8, offset=8).astype(int)
        return "MNIST", X, y
    except Exception as e:
        print("MNIST download failed:", repr(e)[:120])
        from sklearn.datasets import load_digits
        dd = load_digits()
        return "digits", dd.data.astype("float64") / 16.0, dd.target.astype(int)


def load_data(cfg):
    from sklearn.preprocessing import StandardScaler
    name, X, y = _load_raw()
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(X))
    nte = min(cfg["n_test"], len(X) // 3)
    ntr = min(cfg["n_train"], len(X) - nte)
    tr, te = idx[:ntr], idx[ntr:ntr + nte]
    sc = StandardScaler().fit(X[tr])
    Xtr = torch.tensor(sc.transform(X[tr])); Xte = torch.tensor(sc.transform(X[te]))
    return name, Xtr, torch.tensor(np.asarray(y)[tr]), Xte, torch.tensor(np.asarray(y)[te])


def features(model, X):
    """Concatenated normalized block reps (the SCFF/FF probe feature)."""
    with torch.no_grad():
        ys = model(X)
        return torch.cat([normalize(ys[l]) for l in range(1, model.n_layers + 1)], dim=1)


def probe(model, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    Ftr = features(model, Xtr).numpy(); Fte = features(model, Xte).numpy()
    clf = LogisticRegression(max_iter=300, C=1.0).fit(Ftr, ytr.numpy())
    return float((clf.predict(Fte) == yte.numpy()).mean())


def epochs_iter(X, batch, gen):
    idx = torch.randperm(X.shape[0], generator=gen)
    for i in range(0, X.shape[0] - batch + 1, batch):
        yield X[idx[i:i + batch]]


def train_bp(model, Xtr, cfg, gen):
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr_bp"])
    for _ in range(cfg["epochs"]):
        for xb in epochs_iter(Xtr, cfg["batch"], gen):
            xp = xb + cfg["aug_noise"] * torch.randn(xb.shape)
            ys, ysp = model(xb), model(xp)
            loss = -local_goodness(normalize(ys[-1]), normalize(ysp[-1]).detach(), cfg["tau"])
            opt.zero_grad(); loss.backward(); opt.step()


def train_bp_supervised(model, Xtr, ytr, cfg, gen, n_classes):
    """End-to-end backprop with cross-entropy on a readout head (the true task upper bound).
    Returns test-head accuracy via a fresh head; the model is also probed on features after."""
    head = torch.nn.Linear(model.width, n_classes).to(torch.float64)
    opt = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr) - cfg["batch"] + 1, cfg["batch"]):
            b = idx[i:i + cfg["batch"]]
            logits = head(normalize(model(Xtr[b])[-1]))
            loss = lossf(logits, ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return head


def head_accuracy(model, head, Xte, yte):
    with torch.no_grad():
        pred = head(normalize(model(Xte)[-1])).argmax(1)
    return float((pred == yte).double().mean())


def train_scff(model, Xtr, cfg, gen):
    for _ in range(cfg["epochs"]):
        for xb in epochs_iter(Xtr, cfg["batch"], gen):
            xp = xb + cfg["aug_noise"] * torch.randn(xb.shape)
            grads = [A.local_grad(model, xb, xp, l, cfg["tau"]) for l in range(model.n_layers)]
            with torch.no_grad():
                for l in range(model.n_layers):
                    model.W[l].add_(cfg["lr_scff"] * grads[l])   # ascend local goodness


def mean_alignment(model, Xte, cfg):
    x = Xte[:cfg["batch"]]; xp = x + cfg["aug_noise"] * torch.randn(x.shape)
    vals = [alignment_cosine(A.local_grad(model, x, xp, l, cfg["tau"]),
                             global_grad(model, x, xp, l, cfg["tau"]))
            for l in range(model.n_layers - 1)]
    return sum(vals) / len(vals)


def main():
    torch.set_default_dtype(torch.float64)
    name, Xtr, ytr, Xte, yte = load_data(CFG)
    d_in = Xtr.shape[1]
    print(f"data={name}  d_in={d_in}  train={len(Xtr)} test={len(Xte)}  "
          f"arch width={CFG['width']} L={CFG['n_layers']} {CFG['act']}\n")
    gen = torch.Generator().manual_seed(CFG["seed"])

    n_classes = int(ytr.max().item()) + 1
    results = []
    # supervised-BP (cross-entropy end-to-end) = true task upper bound; report head acc + probe.
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "plain", CFG["act"], seed=CFG["seed"])
    head = train_bp_supervised(m, Xtr, ytr, CFG, gen, n_classes)
    results.append(("supervised-BP", probe(m, Xtr, ytr, Xte, yte), mean_alignment(m, Xte, CFG)))
    sup_head = head_accuracy(m, head, Xte, yte)
    print(f"supervised-BP   probe={results[-1][1]:.4f}  head={sup_head:.4f}  A={results[-1][2]:.3f}", flush=True)
    # BP-contrastive (same self-supervised objective as SCFF, end-to-end)
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "plain", CFG["act"], seed=CFG["seed"])
    train_bp(m, Xtr, CFG, gen)
    results.append(("BP-contrastive", probe(m, Xtr, ytr, Xte, yte), mean_alignment(m, Xte, CFG)))
    print(f"BP-contrastive  probe={results[-1][1]:.4f}  A={results[-1][2]:.3f}", flush=True)
    # plain-SCFF
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "plain", CFG["act"], seed=CFG["seed"])
    train_scff(m, Xtr, CFG, gen)
    results.append(("plain-SCFF", probe(m, Xtr, ytr, Xte, yte), mean_alignment(m, Xte, CFG)))
    print(f"plain-SCFF      probe={results[-1][1]:.4f}  A={results[-1][2]:.3f}", flush=True)
    # residual-SCFF
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "residual", CFG["act"],
                alpha=CFG["alpha"], seed=CFG["seed"])
    train_scff(m, Xtr, CFG, gen)
    results.append(("residual-SCFF", probe(m, Xtr, ytr, Xte, yte), mean_alignment(m, Xte, CFG)))
    print(f"residual-SCFF   probe={results[-1][1]:.4f}  A={results[-1][2]:.3f}", flush=True)

    print("\n=== RESULT (test linear-probe accuracy on concat features) ===")
    for name_, acc, al in results:
        print(f"  {name_:14s} probe acc={acc:.4f}  align A={al:.3f}")
    print(f"  supervised-BP head (direct) acc={sup_head:.4f}")
    acc = dict((r[0], r[1]) for r in results)
    print(f"\nresidual-SCFF {acc['residual-SCFF']:.4f} vs plain-SCFF {acc['plain-SCFF']:.4f}: "
          f"the alignment fix changes probe accuracy by {acc['residual-SCFF']-acc['plain-SCFF']:+.4f}.")


if __name__ == "__main__":
    main()
