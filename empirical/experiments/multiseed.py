"""Multi-seed hardening of the headline accuracy claims (lift the single-seed caveat).

Runs the key methods over N seeds (varying init + data split + augmentation noise) and reports
mean +/- std, plus the two claims that matter with error bars:
  - alignment fix:     residual-SCFF  -  plain-SCFF
  - price of locality: supervised-BP  -  best BP-free

Methods: supervised-BP (upper bound), plain-SCFF, residual-SCFF, plain local-supervised.
Reuses the task_accuracy / objective_lever building blocks verbatim.

Run: python experiments/multiseed.py
"""
from __future__ import annotations
import os, sys
import math
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP
from task_accuracy import (CFG, load_data, probe, train_scff, train_bp_supervised,
                           head_accuracy, mean_alignment)
from objective_lever import train_local_supervised, ensemble_head_accuracy

SEEDS = [0, 1, 2, 3, 4]


def mean_std(xs):
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1) if len(xs) > 1 else 0.0
    return m, math.sqrt(v)


def run_seed(seed):
    cfg = dict(CFG); cfg["seed"] = seed
    name, Xtr, ytr, Xte, yte = load_data(cfg)
    d_in = Xtr.shape[1]
    n_classes = int(ytr.max().item()) + 1

    def g():
        return torch.Generator().manual_seed(seed)

    out = {}
    # supervised-BP (upper bound)
    m = ArchMLP(d_in, cfg["width"], cfg["n_layers"], "plain", cfg["act"], seed=seed)
    head = train_bp_supervised(m, Xtr, ytr, cfg, g(), n_classes)
    out["supervised-BP"] = head_accuracy(m, head, Xte, yte)
    # plain-SCFF
    m = ArchMLP(d_in, cfg["width"], cfg["n_layers"], "plain", cfg["act"], seed=seed)
    train_scff(m, Xtr, cfg, g())
    out["plain-SCFF"] = probe(m, Xtr, ytr, Xte, yte)
    out["plain-SCFF/A"] = mean_alignment(m, Xte, cfg)
    # residual-SCFF
    m = ArchMLP(d_in, cfg["width"], cfg["n_layers"], "residual", cfg["act"], alpha=0.2, seed=seed)
    train_scff(m, Xtr, cfg, g())
    out["residual-SCFF"] = probe(m, Xtr, ytr, Xte, yte)
    out["residual-SCFF/A"] = mean_alignment(m, Xte, cfg)
    # plain local-supervised
    m = ArchMLP(d_in, cfg["width"], cfg["n_layers"], "plain", cfg["act"], seed=seed)
    heads = train_local_supervised(m, Xtr, ytr, cfg, g(), n_classes)
    out["plain-local-sup"] = probe(m, Xtr, ytr, Xte, yte)
    return name, out


def main():
    torch.set_default_dtype(torch.float64)
    print(f"multi-seed (n={len(SEEDS)}): {SEEDS}  "
          f"width={CFG['width']} L={CFG['n_layers']} {CFG['act']} epochs={CFG['epochs']}\n")
    acc = {}
    name = None
    for s in SEEDS:
        name, out = run_seed(s)
        for k, v in out.items():
            acc.setdefault(k, []).append(v)
        print(f"  seed {s}: " + "  ".join(
            f"{k}={out[k]:.3f}" for k in
            ["supervised-BP", "plain-SCFF", "residual-SCFF", "plain-local-sup"]), flush=True)

    print(f"\ndata={name}\n=== mean +/- std over {len(SEEDS)} seeds (test acc) ===")
    order = ["supervised-BP", "residual-SCFF", "plain-local-sup", "plain-SCFF"]
    for k in order:
        m, sd = mean_std(acc[k])
        extra = ""
        if k + "/A" in acc:
            am, asd = mean_std(acc[k + "/A"])
            extra = f"   A={am:.3f}+/-{asd:.3f}"
        print(f"  {k:16s} {m:.4f} +/- {sd:.4f}{extra}")

    # the two headline gaps with error bars
    fix = [r - p for r, p in zip(acc["residual-SCFF"], acc["plain-SCFF"])]
    fm, fsd = mean_std(fix)
    bp_free_best = [max(r, l) for r, l in zip(acc["residual-SCFF"], acc["plain-local-sup"])]
    price = [b - f for b, f in zip(acc["supervised-BP"], bp_free_best)]
    pm, psd = mean_std(price)
    print(f"\nalignment fix (residual-SCFF - plain-SCFF):  {fm:+.4f} +/- {fsd:.4f}")
    print(f"price of locality (supervised-BP - best BP-free): {pm:+.4f} +/- {psd:.4f}")


if __name__ == "__main__":
    main()
