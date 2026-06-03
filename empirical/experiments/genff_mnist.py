"""gen-FF MNIST 4-arm test (spec docs/superpowers/specs/2026-06-04-gen-ff-design.md):
BP | SCFF+probe | SCFF-feat+head | denoise-feat+head. Accuracy, 1-forward cost, ECE, OOD-AUROC.
Run: python experiments/genff_mnist.py"""
import os, sys, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genff import (GenFFMLP, EnergyHead, train_early_denoise, train_early_inbatch, train_head,
                   predict, free_energy)
from experiments.task_accuracy import load_data

CFG = dict(width=256, n_layers=4, n_train=8000, n_test=2000, seed=0,
           early_epochs=15, head_epochs=40, batch=64,
           lr_denoise=0.6, sigma=0.5, theta=1.0,
           lr_inbatch=0.05, tau=0.5, aug_noise=0.3,
           lr_head=1e-3, lam=0.2, lr_bp=1e-3, bp_epochs=15)

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
    s = torch.cat([fe_in, fe_ood]).numpy()
    lab = np.r_[np.zeros(len(fe_in)), np.ones(len(fe_ood))]
    return float(roc_auc_score(lab, s))

def probe_arm(m, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    Ftr = m.features(Xtr).detach().numpy(); Fte = m.features(Xte).detach().numpy()
    clf = LogisticRegression(max_iter=300).fit(Ftr, ytr.numpy())
    acc = float((clf.predict(Fte) == yte.numpy()).mean())
    probs = torch.tensor(clf.predict_proba(Fte))
    return acc, ece(probs, yte)

def head_metrics(m, head, Xte, yte, Xood):
    with torch.no_grad():
        logits = head(m.features(Xte))
        acc = float((logits.argmax(1) == yte).float().mean())
        e = ece(torch.softmax(logits, 1), yte)
    auroc = ood_auroc(free_energy(m, head, Xte), free_energy(m, head, Xood))
    return acc, e, auroc

def train_bp(d_in, C, Xtr, ytr, cfg):
    m = GenFFMLP(d_in, cfg["width"], cfg["n_layers"], seed=cfg["seed"])
    head = EnergyHead(cfg["width"] * cfg["n_layers"], C)
    opt = torch.optim.Adam(list(m.parameters()) + list(head.parameters()), lr=cfg["lr_bp"])
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["bp_epochs"]):
        idx = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr) - cfg["batch"] + 1, cfg["batch"]):
            b = idx[i:i + cfg["batch"]]
            loss = lossf(head(m.features(Xtr[b])), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m, head

def main():
    torch.set_default_dtype(torch.float64)
    name, Xtr, ytr, Xte, yte = load_data(CFG)
    d_in, C = Xtr.shape[1], int(ytr.max()) + 1
    Xood = 3.0 * torch.randn(len(Xte), d_in)
    print(f"data={name} train={len(Xtr)} test={len(Xte)} d_in={d_in} C={C} "
          f"width={CFG['width']} L={CFG['n_layers']}\n")
    ec = dict(epochs=CFG["early_epochs"], batch=CFG["batch"], seed=CFG["seed"])
    hc = dict(epochs=CFG["head_epochs"], batch=CFG["batch"], lr=CFG["lr_head"],
              lam=CFG["lam"], sigma=CFG["sigma"], seed=CFG["seed"])

    m, head = train_bp(d_in, C, Xtr, ytr, CFG)
    a, e, au = head_metrics(m, head, Xte, yte, Xood)
    print(f"  supervised-BP      acc={a:.4f}  ECE={e:.4f}  OOD-AUROC={au:.3f}  (1 fwd)", flush=True)

    m = GenFFMLP(d_in, CFG["width"], CFG["n_layers"], seed=CFG["seed"])
    train_early_inbatch(m, Xtr, dict(**ec, lr=CFG["lr_inbatch"], tau=CFG["tau"], aug_noise=CFG["aug_noise"]))
    a2, e2 = probe_arm(m, Xtr, ytr, Xte, yte)
    print(f"  SCFF+probe         acc={a2:.4f}  ECE={e2:.4f}  (probe, 1 fwd)", flush=True)

    head = EnergyHead(CFG["width"] * CFG["n_layers"], C)
    train_head(m, head, Xtr, ytr, hc)
    a3, e3, au3 = head_metrics(m, head, Xte, yte, Xood)
    print(f"  SCFF-feat+head     acc={a3:.4f}  ECE={e3:.4f}  OOD-AUROC={au3:.3f}  (1 fwd)", flush=True)

    m = GenFFMLP(d_in, CFG["width"], CFG["n_layers"], seed=CFG["seed"])
    train_early_denoise(m, Xtr, dict(**ec, lr=CFG["lr_denoise"], sigma=CFG["sigma"], theta=CFG["theta"]))
    head = EnergyHead(CFG["width"] * CFG["n_layers"], C)
    train_head(m, head, Xtr, ytr, hc)
    a4, e4, au4 = head_metrics(m, head, Xte, yte, Xood)
    print(f"  denoise-feat+head  acc={a4:.4f}  ECE={e4:.4f}  OOD-AUROC={au4:.3f}  (1 fwd)", flush=True)

    print("\n=== VERDICT ===")
    print(f"  accuracy:  BP {a:.3f} | denoise+head {a4:.3f} | SCFF+head {a3:.3f} | SCFF+probe {a2:.3f}")
    print(f"  head vs probe (a3-a2): {a3-a2:+.4f}   denoise vs in-batch (a4-a3): {a4-a3:+.4f}")
    print(f"  calibration (lower ECE better): probe {e2:.3f} vs heads {e3:.3f}/{e4:.3f}")
    print(f"  OOD-AUROC (higher better): heads {au3:.3f}/{au4:.3f}")
    print("all arms classify in 1 forward; early training forward-only (<= BP cost).")

if __name__ == "__main__":
    main()
