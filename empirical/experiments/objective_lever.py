"""Objective lever (FINDINGS revision, direction #4): is the residual-SCFF gap to supervised BP
(0.888 vs 0.944) the OBJECTIVE, not the architecture?

The tension diagnostic (diag_tension.py) settled the architecture (residual, alpha in [0.1,0.4])
and showed more cross-layer alignment does not close the remaining gap. The one untested lever is
the local OBJECTIVE: SCFF uses a self-supervised contrastive goodness; supervised-BP uses
cross-entropy on labels. Here we give SCFF a *local supervised* objective and keep it BP-free.

local-supervised SCFF (Mono-Forward style):
  each block l gets a linear head h_l. The block output z = normalize(y^(l+1)) is differentiable
  wrt W[l] ONLY (block_output detaches the block input -> stop-grad between blocks, still local,
  forward-only, no weight transport). Loss = sum_l CE(h_l(z_l), labels); because each block's
  input is detached, gradients never cross blocks -> every W[l] and h_l updates from its OWN local
  CE. One optimizer step, but the locality is structural (detached inputs), not bookkeeping.

Four points on MNIST (same harness, concat-feature linear probe = apples-to-apples):
  plain  local-supervised  (per-block CE, local)        -- objective on plain arch
  residual local-supervised (per-block CE, local)       -- objective + the arch fix
  residual-SCFF (contrastive)                            -- the current method (0.888)
  supervised-BP (global CE, backprop)                   -- upper bound (0.944)

If residual local-supervised -> ~0.94, the gap was the objective and a BP-free method matches BP.
If it stalls < 0.94, the gap is fundamentally global credit assignment.

Run: python experiments/objective_lever.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import normalize
from arch import ArchMLP
from task_accuracy import (CFG, load_data, probe, features, train_scff,
                           train_bp_supervised, head_accuracy, mean_alignment)


def train_local_supervised(model, Xtr, ytr, cfg, gen, n_classes):
    """Per-block local cross-entropy. block_output(x,l) is differentiable wrt W[l] only (input
    detached), so summing block losses and one backward keeps each block's update local. Heads
    co-train. Returns the head list (for block-head ensemble accuracy)."""
    heads = [torch.nn.Linear(model.width, n_classes).to(torch.float64)
             for _ in range(model.n_layers)]
    params = list(model.parameters())
    for h in heads:
        params += list(h.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(cfg["epochs"]):
        idx = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr) - cfg["batch"] + 1, cfg["batch"]):
            b = idx[i:i + cfg["batch"]]
            xb, yb = Xtr[b], ytr[b]
            loss = 0.0
            for l in range(model.n_layers):
                z = model.block_output(xb, l)          # normalize(y^(l+1)), grad wrt W[l] only
                loss = loss + lossf(heads[l](z), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return heads


def ensemble_head_accuracy(model, heads, Xte, yte):
    """Mean of per-block head logits (each head reads its own block's normalized output)."""
    with torch.no_grad():
        logits = 0.0
        for l in range(model.n_layers):
            z = normalize(model(Xte)[l + 1])
            logits = logits + heads[l](z)
        pred = logits.argmax(1)
    return float((pred == yte).double().mean())


def run_local_sup(name, arch, alpha, d_in, Xtr, ytr, Xte, yte, n_classes, gen_fn):
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], arch, CFG["act"],
                alpha=alpha, seed=CFG["seed"])
    heads = train_local_supervised(m, Xtr, ytr, CFG, gen_fn(), n_classes)
    p = probe(m, Xtr, ytr, Xte, yte)
    h = ensemble_head_accuracy(m, heads, Xte, yte)
    print(f"  {name:28s} probe={p:.4f}  head-ens={h:.4f}", flush=True)
    return p, h


def main():
    torch.set_default_dtype(torch.float64)
    name, Xtr, ytr, Xte, yte = load_data(CFG)
    d_in = Xtr.shape[1]
    n_classes = int(ytr.max().item()) + 1
    print(f"data={name}  d_in={d_in}  train={len(Xtr)} test={len(Xte)}  "
          f"width={CFG['width']} L={CFG['n_layers']} {CFG['act']}  epochs={CFG['epochs']}\n")

    def gen_fn():
        return torch.Generator().manual_seed(CFG["seed"])

    print("local-supervised (per-block CE, stop-grad between blocks, forward-only):")
    p_plain, h_plain = run_local_sup("plain local-supervised", "plain", None,
                                     d_in, Xtr, ytr, Xte, yte, n_classes, gen_fn)
    p_res, h_res = run_local_sup("residual local-supervised", "residual", 0.2,
                                 d_in, Xtr, ytr, Xte, yte, n_classes, gen_fn)

    print("\nreference (from task_accuracy / diag):")
    # residual-SCFF contrastive (recompute here for same-run comparison)
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "residual", CFG["act"], alpha=0.2,
                seed=CFG["seed"])
    train_scff(m, Xtr, CFG, gen_fn())
    p_scff = probe(m, Xtr, ytr, Xte, yte)
    print(f"  residual-SCFF (contrastive)  probe={p_scff:.4f}  A={mean_alignment(m, Xte, CFG):.3f}", flush=True)
    # supervised-BP upper bound
    m = ArchMLP(d_in, CFG["width"], CFG["n_layers"], "plain", CFG["act"], seed=CFG["seed"])
    head = train_bp_supervised(m, Xtr, ytr, CFG, gen_fn(), n_classes)
    p_bp = probe(m, Xtr, ytr, Xte, yte); h_bp = head_accuracy(m, head, Xte, yte)
    print(f"  supervised-BP (global CE)    probe={p_bp:.4f}  head={h_bp:.4f}", flush=True)

    print("\n=== VERDICT ===")
    print(f"  plain    local-supervised    probe={p_plain:.4f}  head-ens={h_plain:.4f}")
    print(f"  residual local-supervised    probe={p_res:.4f}  head-ens={h_res:.4f}")
    print(f"  residual-SCFF (contrastive)  probe={p_scff:.4f}")
    print(f"  supervised-BP (upper bound)  probe={p_bp:.4f}  head={h_bp:.4f}")
    gap_obj = p_res - p_scff
    gap_left = h_bp - max(h_res, p_res)
    print(f"\nobjective lever (residual local-sup vs residual contrastive): {gap_obj:+.4f}")
    print(f"remaining gap to supervised-BP head: {gap_left:+.4f}")
    print("=> " + ("OBJECTIVE was the gap: BP-free local-supervised ~matches BP"
                   if gap_left <= 0.02 else
                   "gap persists: not just the objective -- global credit assignment matters"))


if __name__ == "__main__":
    main()
