"""GPU depth-stress (spec docs/superpowers/specs/2026-06-03-cuda-scff-gpu-design.md).
Does residual hold accuracy/alignment as depth grows while plain collapses? Memory flat vs BP?
Run: python experiments/gpu_depth_stress.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_arch import ConvSCFF, _pooled, scff_local_step
from cuda.scff_ext import scff_signal
from experiments.cifar_conv import load_cifar, augment

DEV = "cuda"
DEPTHS = [4, 8, 16, 32]
CFG = dict(C=64, tau=0.5, batch=128, epochs=15, lr_scff=0.05, lr_bp=1e-3,
           aug_noise=0.06, n_train=50000, n_test=10000, seed=0)

def features(model, X, bs=1000):
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            ys = model(X[i:i+bs].to(DEV))
            outs.append(torch.cat([_pooled(ys[l]) for l in range(1, model.n_blocks+1)], 1).cpu())
    return torch.cat(outs).numpy()

def probe(model, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=200, C=1.0).fit(features(model, Xtr), ytr.numpy())
    return float((clf.predict(features(model, Xte)) == yte.numpy()).mean())

def _pooled_grad_proj(g, y):
    gp = g.mean(dim=(2,3))
    z = _pooled(y)
    return gp - z * (z*gp).sum(1, keepdim=True)

def mean_A(model, Xte, cfg):
    x = Xte[:cfg["batch"]].to(DEV); xp = augment(x.cpu(), cfg["aug_noise"]).to(DEV)
    ys, ysp = model(x), model(xp)
    vals = []
    for l in range(model.n_blocks - 1):
        z, zp = _pooled(ys[l+1]).detach(), _pooled(ysp[l+1]).detach()
        s_loc, _ = scff_signal(z, zp, cfg["tau"])
        sL, _ = scff_signal(_pooled(ys[-1]).detach(), _pooled(ysp[-1]).detach(), cfg["tau"])
        with torch.enable_grad():
            yin = ys[l+1].detach().requires_grad_(True)
            ytmp = yin
            for k in range(l+1, model.n_blocks):
                ytmp = model.apply_block(ytmp, k)
            g = torch.autograd.grad(_pooled(ytmp), yin, grad_outputs=sL)[0]
        s_bp = _pooled_grad_proj(g, ys[l+1])
        vals.append(float((s_loc.flatten() @ s_bp.flatten()) /
                          (s_loc.norm() * s_bp.norm() + 1e-9)))
    return sum(vals) / len(vals)

def train_scff(model, Xtr, cfg):
    g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr)-cfg["batch"]+1, cfg["batch"]):
            b = idx[i:i+cfg["batch"]]; xb = Xtr[b].to(DEV)
            xp = augment(Xtr[b], cfg["aug_noise"]).to(DEV)
            scff_local_step(model, xb, xp, cfg["tau"], cfg["lr_scff"])

def train_bp(model, Xtr, ytr, cfg, n_classes=10):
    head = torch.nn.Linear(model.C, n_classes).to(DEV)
    opt = torch.optim.Adam(list(model.parameters())+list(head.parameters()), lr=cfg["lr_bp"])
    lossf = torch.nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(cfg["seed"])
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr)-cfg["batch"]+1, cfg["batch"]):
            b = idx[i:i+cfg["batch"]]
            logits = head(model.pooled(model(Xtr[b].to(DEV))[-1]))
            loss = lossf(logits, ytr[b].to(DEV)); opt.zero_grad(); loss.backward(); opt.step()

def run_arm(name, make, train, Xtr, ytr, Xte, yte, want_A):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    m = make().to(DEV); train(m)
    mem = torch.cuda.max_memory_allocated()/1e6
    acc = probe(m, Xtr, ytr, Xte, yte)
    A = mean_A(m, Xte, CFG) if want_A else float("nan")
    print(f"  {name:16s} acc={acc:.4f}  A={A:.3f}  peakMB={mem:.0f}", flush=True)
    return acc, A, mem

def main():
    assert torch.cuda.is_available(), "GPU required"
    Xtr, ytr, Xte, yte = load_cifar(CFG)
    print(f"CIFAR-10 train={len(Xtr)} test={len(Xte)} on {torch.cuda.get_device_name(0)}\n")
    for L in DEPTHS:
        print(f"L={L}:")
        run_arm("supervised-BP", lambda: ConvSCFF(CFG["C"], L, "plain"),
                lambda m: train_bp(m, Xtr, ytr, CFG), Xtr, ytr, Xte, yte, False)
        run_arm("plain-SCFF", lambda: ConvSCFF(CFG["C"], L, "plain"),
                lambda m: train_scff(m, Xtr, CFG), Xtr, ytr, Xte, yte, True)
        run_arm("residual-SCFF", lambda: ConvSCFF(CFG["C"], L, "residual", 1.0/math.sqrt(L)),
                lambda m: train_scff(m, Xtr, CFG), Xtr, ytr, Xte, yte, True)

if __name__ == "__main__":
    main()
