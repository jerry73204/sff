"""Conv gen-FF CIFAR-10 4-arm test (spec docs/superpowers/specs/2026-06-04-genff-conv-gpu-design.md):
BP | SCFF-conv+probe | SCFF-conv+head | gen-FF-conv(denoise+head). Does denoising-energy beat SCFF's
conv wall? Accuracy, ECE, OOD-AUROC, peak GPU memory, 1-forward.
Run: python experiments/genff_cifar.py"""
import os, sys, math, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpu_arch import ConvSCFF, _pooled, scff_local_step
from genff import EnergyHead, predict, free_energy
from genff_conv import conv_denoise_step, train_head_conv
from gpu_pipeline import to_gpu, augment_batch, ece, ood_auroc

DEV = "cuda"
CFG = dict(C=128, n_blocks=6, batch=128, n_train=50000, n_test=10000, seed=0,
           early_epochs=20, head_epochs=30, sigma=0.5,
           lr_denoise=0.5, lr_inbatch=0.05, tau=0.5, aug_noise=0.1,
           lr_head=1e-3, lam=0.2, lr_bp=1e-3, bp_epochs=20)

def mk():
    return ConvSCFF(CFG["C"], CFG["n_blocks"], "residual", 1.0 / math.sqrt(CFG["n_blocks"])).to(DEV)

def batches(n, bs, gen):
    idx = torch.randperm(n, generator=gen)
    for i in range(0, n - bs + 1, bs):
        yield idx[i:i + bs]

def train_denoise(m, Xtr):
    g = torch.Generator().manual_seed(CFG["seed"])
    for _ in range(CFG["early_epochs"]):
        for b in batches(len(Xtr), CFG["batch"], g):
            conv_denoise_step(m, augment_batch(Xtr[b]), dict(sigma=CFG["sigma"], lr=CFG["lr_denoise"]))

def train_inbatch(m, Xtr):
    g = torch.Generator().manual_seed(CFG["seed"])
    for _ in range(CFG["early_epochs"]):
        for b in batches(len(Xtr), CFG["batch"], g):
            xb = augment_batch(Xtr[b]); xp = augment_batch(Xtr[b])
            scff_local_step(m, xb, xp, CFG["tau"], CFG["lr_inbatch"])

def hc():
    return dict(epochs=CFG["head_epochs"], batch=CFG["batch"], lr=CFG["lr_head"],
               lam=CFG["lam"], sigma=CFG["sigma"], seed=CFG["seed"])

def probe(m, Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    Ftr = m.features(Xtr).detach().cpu().numpy(); Fte = m.features(Xte).detach().cpu().numpy()
    clf = LogisticRegression(max_iter=200).fit(Ftr, ytr.cpu().numpy())
    acc = float((clf.predict(Fte) == yte.cpu().numpy()).mean())
    return acc, ece(torch.tensor(clf.predict_proba(Fte)), yte.cpu())

def head_eval(m, head, Xte, yte, Xood):
    with torch.no_grad():
        logits = head(m.features(Xte))
        acc = float((logits.argmax(1) == yte).float().mean())
        e = ece(torch.softmax(logits, 1).cpu(), yte.cpu())
    au = ood_auroc(free_energy(m, head, Xte), free_energy(m, head, Xood))
    return acc, e, au

def peakMB():
    return torch.cuda.max_memory_allocated() / 1e6

def main():
    assert torch.cuda.is_available(), "GPU required"
    Xtr, ytr, Xte, yte = to_gpu(CFG, DEV)
    C = int(ytr.max()) + 1; fd = CFG["C"] * CFG["n_blocks"]
    Xood = (3.0 * torch.randn(len(Xte), 3, 32, 32, device=DEV))
    print(f"CIFAR-10 train={len(Xtr)} test={len(Xte)} conv C={CFG['C']} L={CFG['n_blocks']} "
          f"on {torch.cuda.get_device_name(0)}\n")

    torch.cuda.reset_peak_memory_stats()
    m = mk(); head = EnergyHead(fd, C).to(DEV)
    opt = torch.optim.Adam(list(m.parameters()) + list(head.parameters()), lr=CFG["lr_bp"])
    lossf = torch.nn.CrossEntropyLoss(); g = torch.Generator().manual_seed(CFG["seed"])
    for _ in range(CFG["bp_epochs"]):
        for b in batches(len(Xtr), CFG["batch"], g):
            loss = lossf(head(m.features(augment_batch(Xtr[b]))), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    a1, e1, au1 = head_eval(m, head, Xte, yte, Xood); mb1 = peakMB()
    print(f"  supervised-BP        acc={a1:.4f} ECE={e1:.4f} OOD={au1:.3f} peakMB={mb1:.0f} (1fwd)", flush=True)

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    m = mk(); train_inbatch(m, Xtr); mb2 = peakMB()
    a2, e2 = probe(m, Xtr, ytr, Xte, yte)
    print(f"  SCFF-conv+probe      acc={a2:.4f} ECE={e2:.4f} peakMB={mb2:.0f} (probe,1fwd)", flush=True)

    head = EnergyHead(fd, C).to(DEV); train_head_conv(m, head, Xtr, ytr, hc())
    a3, e3, au3 = head_eval(m, head, Xte, yte, Xood)
    print(f"  SCFF-conv+head       acc={a3:.4f} ECE={e3:.4f} OOD={au3:.3f} (1fwd)", flush=True)

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    m = mk(); train_denoise(m, Xtr); mb4 = peakMB()
    head = EnergyHead(fd, C).to(DEV); train_head_conv(m, head, Xtr, ytr, hc())
    a4, e4, au4 = head_eval(m, head, Xte, yte, Xood)
    print(f"  gen-FF-conv          acc={a4:.4f} ECE={e4:.4f} OOD={au4:.3f} peakMB={mb4:.0f} (1fwd)", flush=True)

    print("\n=== VERDICT ===")
    print(f"  acc:  BP {a1:.3f} | gen-FF {a4:.3f} | SCFF+head {a3:.3f} | SCFF+probe {a2:.3f}")
    print(f"  gen-FF vs SCFF-wall (a4-a2): {a4-a2:+.4f}")
    print(f"  denoise vs in-batch (a4-a3): {a4-a3:+.4f}   head vs probe (a3-a2): {a3-a2:+.4f}")
    print(f"  ECE: probe {e2:.3f} | heads {e3:.3f}/{e4:.3f}   OOD: heads {au3:.3f}/{au4:.3f}")
    print(f"  peak MB: BP {mb1:.0f} | SCFF {mb2:.0f} | gen-FF {mb4:.0f}")
    print("=> " + ("gen-FF DODGES the conv wall (>> SCFF)" if a4 - a2 > 0.05
                   else "gen-FF ~= SCFF on conv: the conv gap is deeper than the objective"))

if __name__ == "__main__":
    main()
