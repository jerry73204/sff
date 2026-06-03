# Idea: FF + BP hybrid for large-model fine-tuning (parked)

**Status:** analyzed, parked in favor of the FF-generative direction. Captured here so it's not lost.

**Idea (user's):** for fine-tuning large models, run **BP on the tail layers** and **FF (forward-only,
local) on the rest**. Hard split: FF layers trained only by local goodness, BP layers only by the
task loss, no gradient crosses the boundary.

## Why it's well-motivated (by our own findings)

1. **It realizes the memory win we couldn't get naively.** Our GPU result: naive forward-only SCFF
   used ~1× BP memory because a full forward holds `O(L)` activations — the flat-memory advantage
   needs a *streaming/local* schedule. The hybrid IS that: BP only the last `k` layers → backward
   graph spans `k`, not `L`. `full BP: O(L·B·n)` → `FF + BP-k: O(k·B·n)`. For `k ≪ L` this is the
   real saving, and it's the deployment form of forward-only (shrink BP's depth, don't replace it).
2. **The split is theoretically right.** Our price-of-locality result: the gap to BP is *global
   credit assignment*, and it concentrates where the task error must be assigned — near the head in
   fine-tuning (early features transfer; the tail re-maps to the task). So BP-on-tail spends the
   expensive global credit assignment exactly where it's needed; FF-early is "good enough" for the
   light, task-agnostic adaptation generic features need.
3. **Niche: label-free domain adaptation + task head.** FF-adapt early layers self-supervised on the
   new domain (no labels, no BP memory, layer-parallel); BP-train the tail with the few labels.
   Decouples *domain shift* (FF) from *task learning* (BP). Strong for on-device / continual /
   large-model fine-tuning where full BP memory is infeasible and labels are scarce.

## The conv result that motivates it

Thread A (per-location SCFF) showed pure local cannot close the conv/CIFAR gap (~0.22 to BP) — the
price of locality grows with task difficulty (MLP ~3pt, conv/CIFAR ~22pt). So on hard tasks, global
credit assignment is irreducibly needed → put BP exactly where it matters (the tail), FF the rest.

## Design choices / risks (for when this is revived)

- **Hard vs soft split** — hard (no gradient crosses) is memory-optimal; start there.
- **Boundary `k`** — sweep it (more BP layers = more task adaptation + more memory).
- **Objective mismatch (main risk)** — FF's contrastive goodness ≠ the task; early layers drift to
  "contrastive-good", which may not be "task-useful". Minor for *fine-tuning* (small moves), fatal
  for *from-scratch* — so this idea is a fine-tuning method, not a training method.
- **Baseline to beat:** not full BP, but **"freeze early + BP last `k`"** (the cheap standard).
  FF-early earns its keep only if label-free early adaptation beats freezing — true under domain shift.

## Decisive test (when revived)

Domain-shift transfer (e.g. CIFAR-10 → CIFAR-10-C). Three arms at matched BP-depth `k`: full BP
(ceiling), freeze-early + BP-tail (cheap baseline), FF-early + BP-tail (hybrid). Plot accuracy and
peak memory vs `k`. Win condition: hybrid ≈ full-BP accuracy at the freeze baseline's memory.
