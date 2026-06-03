"""Spatial SCFF (spec docs/superpowers/specs/2026-06-03-spatial-scff-design.md): does moving the
local objective from global-pool to per-location fix the conv bottleneck?
Compares supervised-BP vs global-pool-SCFF vs per-location-SCFF on CIFAR-10 (residual conv).
Run: python experiments/cifar_spatial.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_arch import (ConvSCFF, _pooled, scff_local_step, scff_local_step_spatial,
                      augment_appearance)
from experiments.cifar_conv import load_cifar

DEV = "cuda"
CFG = dict(C=64, n_blocks=8, alpha=1.0/math.sqrt(8), tau=0.5, batch=128, epochs=12,
           lr_scff=0.05, lr_bp=1e-3, aug_noise=0.06, n_train=20000, n_test=5000, seed=0)

def features(model, X, bs=1000):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ys = model(X[i:i+bs].to(DEV))
            outs.append(torch.cat([_pooled(ys[l]) for l in range(1, model.n_blocks+1)], 1).cpu())
    return torch.cat(outs).numpy()

def probe(model, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=300, C=1.0).fit(features(model, Xtr), ytr.numpy())
    return float((clf.predict(features(model, Xte)) == yte.numpy()).mean())

def batches(n, bs, gen):
    idx = torch.randperm(n, generator=gen)
    for i in range(0, n-bs+1, bs):
        yield idx[i:i+bs]

def train_scff(model, Xtr, cfg, step_fn):
    gen = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        for b in batches(len(Xtr), cfg["batch"], gen):
            xb = Xtr[b].to(DEV); xp = augment_appearance(xb, cfg["aug_noise"])
            step_fn(model, xb, xp, cfg["tau"], cfg["lr_scff"])

def train_bp(model, Xtr, ytr, cfg, n_classes=10):
    head = torch.nn.Linear(model.C, n_classes).to(DEV)
    opt = torch.optim.Adam(list(model.parameters())+list(head.parameters()), lr=cfg["lr_bp"])
    lossf = torch.nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        for b in batches(len(Xtr), cfg["batch"], gen):
            logits = head(model.pooled(model(Xtr[b].to(DEV))[-1]))
            loss = lossf(logits, ytr[b].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()

def run(name, make, train, Xtr, ytr, Xte, yte):
    m = make().to(DEV); train(m)
    acc = probe(m, Xtr, ytr, Xte, yte)
    print(f"  {name:20s} acc={acc:.4f}", flush=True)
    return acc

def main():
    assert torch.cuda.is_available(), "GPU required"
    Xtr, ytr, Xte, yte = load_cifar(CFG)
    L = CFG["n_blocks"]
    print(f"CIFAR-10 train={len(Xtr)} test={len(Xte)} residual conv C={CFG['C']} L={L} "
          f"on {torch.cuda.get_device_name(0)}\n")
    def mk(): return ConvSCFF(CFG["C"], L, "residual", CFG["alpha"])
    a_bp  = run("supervised-BP",      mk, lambda m: train_bp(m, Xtr, ytr, CFG), Xtr, ytr, Xte, yte)
    a_pool= run("global-pool-SCFF",   mk, lambda m: train_scff(m, Xtr, CFG, scff_local_step),
                Xtr, ytr, Xte, yte)
    a_loc = run("per-location-SCFF",  mk, lambda m: train_scff(m, Xtr, CFG, scff_local_step_spatial),
                Xtr, ytr, Xte, yte)
    print("\n=== VERDICT (CIFAR-10 linear probe, residual conv) ===")
    print(f"  supervised-BP      {a_bp:.4f}")
    print(f"  per-location-SCFF  {a_loc:.4f}")
    print(f"  global-pool-SCFF   {a_pool:.4f}")
    print(f"\nspatial fix (per-location - global-pool): {a_loc-a_pool:+.4f}")
    print(f"remaining gap to BP: {a_bp-a_loc:+.4f}")
    print("=> " + ("per-location FIXES the conv bottleneck (>> global-pool, toward BP)"
                   if a_loc - a_pool > 0.05 else
                   "per-location did not clearly beat global-pool -- investigate"))

if __name__ == "__main__":
    main()
