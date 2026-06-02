"""E_LayerNorm -- does parameter-free per-block LayerNorm cut the cross-layer kernel
drift (delta) and lift SCFF<->backprop alignment, the way residual skips do?

Background (docs/FINDINGS.md). At init, for a non-final block l,
    1 - A^(l)  <=  C/sqrt(n)  +  C'*delta,     delta = ||p^(l) - p^(L)||  (kernel drift).
delta is the BINDING term and a DEPTH effect that width does NOT fix. A residual arch
fixes it (downstream Jacobian M ~ I). Gap #3 of FINDINGS asks whether delta is partly an
artifact of plain MLPs having NO normalization, since real nets use LayerNorm/BatchNorm
which already control kernel drift.

This experiment: plain MLP, depth sweep L in {4,8,16}, width n=256, d_in=64, batch B=4,
linear (primary) and relu (optional), >=5 seeds. Compare no-norm vs per-block LayerNorm.
Per non-final block, mean 1-A, delta (delta_gram vs final block), Aniso.

Falsifiable hypothesis: LayerNorm reduces delta and 1-A vs no-norm, gap widening with depth.

Run:  .venv/bin/python experiments/e_layernorm.py
Outputs: runs/e_layernorm_<act>.csv, plots/e_layernorm_<act>.png + console table + verdict.
"""
from __future__ import annotations
import os, sys, csv
import yaml
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from arch import ArchMLP, measure_blocks

torch.set_default_dtype(torch.float64)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS, PLOTS = os.path.join(HERE, "runs"), os.path.join(HERE, "plots")

CFG = dict(d_in=64, width=256, batch=4, tau=0.5,
           depths=[4, 8, 16], seeds=[0, 1, 2, 3, 4],
           conditions=["nonorm", "layernorm"], acts=["linear", "relu"])


def _data(seed, cfg):
    g = torch.Generator().manual_seed(1000 + seed)
    x = torch.randn(cfg["batch"], cfg["d_in"], generator=g)
    xp = x + 0.1 * torch.randn(x.shape, generator=g)
    return x, xp


def mean_nonfinal(rows, idx):
    """Mean of metric `idx` over non-final blocks (the final block is trivially A=1)."""
    nf = rows[:-1] if len(rows) > 1 else rows
    return float(np.mean([r[idx] for r in nf]))


def run(cfg, act):
    rows = []
    for cond in cfg["conditions"]:
        norm = (cond == "layernorm")
        for L in cfg["depths"]:
            for seed in cfg["seeds"]:
                x, xp = _data(seed, cfg)
                m = ArchMLP(cfg["d_in"], cfg["width"], L, arch="plain", act=act,
                            seed=seed, norm=norm)
                mb = measure_blocks(m, x, xp, cfg["tau"])   # (l, A, dgram, aniso, dV)
                rows.append(dict(cond=cond, L=L, seed=seed,
                                 one_minus_A=1.0 - mean_nonfinal(mb, 1),
                                 dgram=mean_nonfinal(mb, 2),
                                 aniso=mean_nonfinal(mb, 3),
                                 dV=mean_nonfinal(mb, 4)))
            print(f"[{act}] {cond:9s} L={L:2d} done", flush=True)
    return rows


def agg(rows, cond, L, key):
    vals = [r[key] for r in rows if r["cond"] == cond and r["L"] == L]
    return float(np.mean(vals)), float(np.std(vals))


def plot(rows, cfg, act, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, key, ylab in [(axes[0], "one_minus_A", r"$1-A$ (init, mean non-final block)"),
                          (axes[1], "dgram", r"$\delta$ (delta_gram vs final block)")]:
        for cond in cfg["conditions"]:
            mu = [agg(rows, cond, L, key)[0] for L in cfg["depths"]]
            sd = [agg(rows, cond, L, key)[1] for L in cfg["depths"]]
            ax.errorbar(cfg["depths"], mu, yerr=sd, marker="o", capsize=3, label=cond)
        ax.set_xlabel("depth L"); ax.set_ylabel(ylab); ax.legend()
    fig.suptitle(f"Per-block LayerNorm vs no-norm, plain MLP ({act}): does LN cut "
                 r"$\delta$ and $1-A$?")
    fig.tight_layout(); fig.savefig(path, dpi=130); print("plot ->", path)


def report(rows, cfg, act):
    print(f"\n=== [{act}] 1-A and delta by (condition, depth)  (mean +/- std over "
          f"{len(cfg['seeds'])} seeds) ===")
    print(f"  {'cond':10s} " + "  ".join(f"L{L:>2}{'':9s}" for L in cfg["depths"]))
    for key, lab in [("one_minus_A", "1-A"), ("dgram", "delta"), ("aniso", "Aniso")]:
        print(f"  metric: {lab}")
        for cond in cfg["conditions"]:
            cells = []
            for L in cfg["depths"]:
                mu, sd = agg(rows, cond, L, key)
                cells.append(f"{mu:.3f}+-{sd:.3f}")
            print(f"    {cond:10s} " + "  ".join(cells))

    # Verdict: does LN reduce 1-A and delta, with the gap widening with depth?
    print(f"\n=== [{act}] VERDICT ===")
    gaps_A, gaps_d = {}, {}
    for L in cfg["depths"]:
        a_nn = agg(rows, "nonorm", L, "one_minus_A")[0]
        a_ln = agg(rows, "layernorm", L, "one_minus_A")[0]
        d_nn = agg(rows, "nonorm", L, "dgram")[0]
        d_ln = agg(rows, "layernorm", L, "dgram")[0]
        gaps_A[L] = a_nn - a_ln          # >0 => LN lowers 1-A (helps)
        gaps_d[L] = d_nn - d_ln          # >0 => LN lowers delta (helps)
        print(f"  L={L:2d}: 1-A  nonorm {a_nn:.3f} -> LN {a_ln:.3f}  (LN improves "
              f"by {gaps_A[L]:+.3f});  delta  nonorm {d_nn:.3f} -> LN {d_ln:.3f}  "
              f"(LN improves by {gaps_d[L]:+.3f})")
    Ls = cfg["depths"]
    helps_A = all(gaps_A[L] > 0 for L in Ls)
    helps_d = all(gaps_d[L] > 0 for L in Ls)
    widen_A = gaps_A[Ls[-1]] > gaps_A[Ls[0]]
    widen_d = gaps_d[Ls[-1]] > gaps_d[Ls[0]]
    print(f"  LN reduces 1-A at every depth: {helps_A};  reduces delta at every depth: "
          f"{helps_d}")
    print(f"  1-A gap widens with depth: {widen_A};  delta gap widens with depth: "
          f"{widen_d}")
    if helps_A and helps_d and widen_A and widen_d:
        v = "HYPOTHESIS HELD: LayerNorm cuts both delta and 1-A, gap widening with depth."
    elif helps_A and helps_d:
        v = "PARTIAL: LayerNorm cuts both delta and 1-A, but the gap does NOT widen monotonically with depth."
    elif helps_d and not helps_A:
        v = "PARTIAL/SURPRISING: LayerNorm cuts delta but NOT 1-A -- delta is not the only binding term here."
    elif helps_A and not helps_d:
        v = "SURPRISING: LayerNorm lifts alignment without cutting delta -- mechanism is not kernel-drift."
    else:
        v = "HYPOTHESIS REJECTED: LayerNorm does not cut delta or 1-A vs no-norm."
    print(f"  --> {v}")
    return v


def main():
    os.makedirs(RUNS, exist_ok=True); os.makedirs(PLOTS, exist_ok=True)
    with open(os.path.join(RUNS, "e_layernorm.yaml"), "w") as f:
        yaml.safe_dump(CFG, f)
    verdicts = {}
    for act in CFG["acts"]:
        rows = run(CFG, act)
        with open(os.path.join(RUNS, f"e_layernorm_{act}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["cond", "L", "seed", "one_minus_A",
                                              "dgram", "aniso", "dV"])
            w.writeheader(); w.writerows(rows)
        plot(rows, CFG, act, os.path.join(PLOTS, f"e_layernorm_{act}.png"))
        verdicts[act] = report(rows, CFG, act)
    print("\n=== SUMMARY ===")
    for act, v in verdicts.items():
        print(f"  [{act}] {v}")


if __name__ == "__main__":
    main()
