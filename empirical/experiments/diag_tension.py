"""Diagnostic: the alignment <-> expressivity tension (FINDINGS revision probe).

The residual fix aligns by driving M -> I (lazy regime): great alignment, but the residual
branch barely contributes when alpha is small, so features may be under-expressive. The MNIST
result (residual-SCFF 0.887 vs supervised-BP 0.944) leaves a ~6pt gap that might be this lazy cap.

Two questions, two probes, reusing the task_accuracy harness (MNIST, concat-feature linear probe):

  (A) accuracy-vs-alpha sweep for residual-SCFF.
      If the best-*accuracy* alpha is LARGER than the best-*alignment* alpha (=small), then
      small-alpha residual is lazy-capped: the method should use a bigger (or scheduled) alpha
      for accuracy, trading some A.

  (B) aux-depth-SCFF (look-ahead j) accuracy on the PLAIN arch.
      Aux-depth aligns by letting the local objective SEE M (M stays rich/expressive) instead
      of making M trivial. If it beats residual-SCFF on accuracy, the next revision should be
      "see M" (rich + aligned), not "make M = I" (lazy), accepting the compute cost.

Run: python experiments/diag_tension.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP
from gradients import local_goodness, global_grad, alignment_cosine
import arch as A
import auxdepth as AD
# reuse the task harness verbatim (data, features, probe, plain/residual SCFF training)
from task_accuracy import (CFG, load_data, probe, train_scff, mean_alignment, epochs_iter)


def train_scff_aux(model, Xtr, cfg, gen, j):
    """SCFF training with auxiliary-depth look-ahead j for every block (auxdepth.aux_scff_step)."""
    for _ in range(cfg["epochs"]):
        for xb in epochs_iter(Xtr, cfg["batch"], gen):
            xp = xb + cfg["aug_noise"] * torch.randn(xb.shape)
            AD.aux_scff_step(model, xb, xp, cfg["tau"], cfg["lr_scff"], j)


def mean_alignment_aux(model, Xte, cfg, j):
    """Mean cos(aux_local_grad, global_grad) over non-final blocks (j-block look-ahead local grad)."""
    x = Xte[:cfg["batch"]]; xp = x + cfg["aug_noise"] * torch.randn(x.shape)
    vals = [alignment_cosine(AD.aux_local_grad(model, x, xp, l, cfg["tau"], j),
                             global_grad(model, x, xp, l, cfg["tau"]))
            for l in range(model.n_layers - 1)]
    return sum(vals) / len(vals)


def main():
    torch.set_default_dtype(torch.float64)
    name, Xtr, ytr, Xte, yte = load_data(CFG)
    d_in = Xtr.shape[1]
    print(f"data={name}  d_in={d_in}  train={len(Xtr)} test={len(Xte)}  "
          f"width={CFG['width']} L={CFG['n_layers']} {CFG['act']}  epochs={CFG['epochs']}\n")

    def fresh_gen():
        return torch.Generator().manual_seed(CFG["seed"])

    # --- baselines for reference ---
    print("baseline:")
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "plain", CFG["act"], seed=CFG["seed"])
    train_scff(m, Xtr, CFG, fresh_gen())
    acc0 = probe(m, Xtr, ytr, Xte, yte); A0 = mean_alignment(m, Xte, CFG)
    print(f"  plain-SCFF (j=0)         acc={acc0:.4f}  A={A0:.3f}", flush=True)

    # --- (A) accuracy vs alpha for residual-SCFF ---
    print("\n(A) residual-SCFF accuracy vs alpha (alignment fix = small alpha):")
    print(f"  {'alpha':>6}  {'acc':>7}  {'A':>6}")
    sweep = []
    for alpha in (0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "residual", CFG["act"],
                    alpha=alpha, seed=CFG["seed"])
        train_scff(m, Xtr, CFG, fresh_gen())
        acc = probe(m, Xtr, ytr, Xte, yte); Aval = mean_alignment(m, Xte, CFG)
        sweep.append((alpha, acc, Aval))
        print(f"  {alpha:>6.2f}  {acc:>7.4f}  {Aval:>6.3f}", flush=True)
    best_acc = max(sweep, key=lambda r: r[1])
    best_aln = max(sweep, key=lambda r: r[2])
    print(f"  best-accuracy  alpha={best_acc[0]:.2f} (acc={best_acc[1]:.4f}, A={best_acc[2]:.3f})")
    print(f"  best-alignment alpha={best_aln[0]:.2f} (acc={best_aln[1]:.4f}, A={best_aln[2]:.3f})")
    print(f"  => {'LAZY-CAPPED: best-acc alpha > best-align alpha' if best_acc[0] > best_aln[0] else 'NOT lazy-capped: best-acc alpha == best-align alpha'}")

    # --- (B) aux-depth-SCFF (see M, stay rich) on the plain arch ---
    print("\n(B) aux-depth-SCFF (plain arch, look-ahead j = sees M, M stays rich):")
    print(f"  {'j':>3}  {'acc':>7}  {'A':>6}")
    aux = []
    for j in (1, 2, 3):
        if j > CFG["n_layers"] - 1:
            break
        m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "plain", CFG["act"], seed=CFG["seed"])
        train_scff_aux(m, Xtr, CFG, fresh_gen(), j)
        acc = probe(m, Xtr, ytr, Xte, yte); Aval = mean_alignment_aux(m, Xte, CFG, j)
        aux.append((j, acc, Aval))
        print(f"  {j:>3}  {acc:>7.4f}  {Aval:>6.3f}", flush=True)

    # --- verdict ---
    print("\n=== VERDICT ===")
    best_aux = max(aux, key=lambda r: r[1]) if aux else None
    print(f"plain-SCFF (j=0)          {acc0:.4f}")
    print(f"residual-SCFF best-acc    {best_acc[1]:.4f}  (alpha={best_acc[0]:.2f})")
    if best_aux:
        print(f"aux-depth-SCFF best-acc   {best_aux[1]:.4f}  (j={best_aux[0]})")
        winner = max([("residual", best_acc[1]), ("aux-depth", best_aux[1])], key=lambda r: r[1])
        print(f"best gap-closer for ACCURACY: {winner[0]} ({winner[1]:.4f})")
    print(f"lazy-cap signal: best-acc alpha = {best_acc[0]:.2f} vs best-align alpha = {best_aln[0]:.2f}")


if __name__ == "__main__":
    main()
