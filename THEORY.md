# THEORY.md — Single Source of Truth for Symbols

This file is the shared symbol table for both tracks. Every empirical quantity in
`empirical/` and every Lean definition in `lean/SffProof/` must map to a symbol named
here. Lean↔symbol mapping is maintained in `lean/LEAN_NOTES.md`; code↔symbol mapping is
maintained by matching variable names to the **Code symbol** column.

---

## 1. Network

| Symbol | Meaning | Code symbol |
|---|---|---|
| `L` | number of layers | `L`, `n_layers` |
| `n` | width (hidden dimension) | `n`, `width` |
| `B` | batch size | `B`, `batch_size` |
| `σ` | nonlinearity; `σ = id` in **linear mode** (primary), ReLU in stretch mode | `act` |
| `W^(ℓ)` | weight matrix of layer `ℓ`, shape `n × n` (`n × n_in` at `ℓ=1`) | `W[l]` |
| `y^(ℓ)` | layer activation, `y^(ℓ) = σ(W^(ℓ) y^(ℓ-1))`; `y^(0) = x` | `y[l]` |

μP-style init: `W^(ℓ)_{ab} ~ N(0, 1/fan_in)` for hidden layers, `1/fan_in²`-variance
(i.e. `1/fan_in` std → scaled by `1/fan_in`) for the last layer. Unit-test the variances
(Pitfall §6 in design).

---

## 2. Normalized representations and Gram matrices

| Symbol | Definition | Code symbol |
|---|---|---|
| `z^(ℓ)(x)` | `y^(ℓ)(x) / ‖y^(ℓ)(x)‖` (unit-normalized rep) | `z[l]` |
| `K^(ℓ)_{ij}` | `⟨z^(ℓ)(x_i), z^(ℓ)(x_j)⟩` (Gram over batch) | `K[l]` |
| `τ` | InfoNCE temperature | `tau` |
| `p^(ℓ)_{ij}` | `softmax_j(⟨z_i, z_j⟩ / τ)` (in-batch InfoNCE weights at layer ℓ) | `p[l]` |
| `x_{i+}` | augmentation-based positive partner of `x_i`; `z_{i+}` its normalized rep | `z_pos` |

`P⊥_z = (I − z z^T) / ‖y‖` — the Jacobian of the normalization map `y ↦ y/‖y‖`,
evaluated at `y` with unit direction `z`. (Projects off `z`, scales by `1/‖y‖`.)

---

## 3. Gradients

**Local goodness gradient** (per layer, derived symbolically; reproduce derivation in code
tests against autograd):

```
∇_{W^(ℓ)} g^(ℓ) = (1/τ) Σ_i  P⊥_{z_i}  [ z_{i+} − Σ_j p^(ℓ)_{ij} z_j ]  (y^(ℓ-1)_i)^T
```

- right factor: `(y^(ℓ-1)_i)^T` — the layer input.
- bracket: contrastive-signal vector `s_i := z_{i+} − Σ_j p^(ℓ)_{ij} z_j`.
- left factor: `P⊥_{z_i} s_i` — signal projected off the current direction.

**Global gradient projected to layer ℓ** (`∇_{W^(ℓ)} L_con`): **same right factor**
`(y^(ℓ-1)_i)^T`; left factor carries the **downstream Jacobian** `M^(ℓ+1→L)` and the
**final-layer** softmax weights `p^(L)` instead of `p^(ℓ)`. Used as a measurement probe
only — one real backward pass through `L_con`, never fed to the optimizer.

| Symbol | Meaning | Code symbol |
|---|---|---|
| `s_i^(ℓ)` | contrastive-signal vector `z_{i+} − Σ_j p^(ℓ)_{ij} z_j` | `s[l]` |
| `M^(ℓ+1→L)` | downstream Jacobian, layers `ℓ+1..L`. ReLU mode: `Π_k W^(k) D^(k)` (per-sample) | `M[l]` |
| `D^(k)` | diagonal ReLU activation mask at layer `k` | `D[k]` |

---

## 4. Measured / proven quantities

| Symbol | Definition | Code symbol |
|---|---|---|
| `A^(ℓ)` | **alignment cosine** `cos∠(∇_{W^(ℓ)} g^(ℓ), ∇_{W^(ℓ)} L_con)` | `align[l]` |
| `Δ_Gram^(ℓ)` | **Gram misalignment** `‖K^(ℓ) − K^(L)‖_F` (normalized) | `dgram[l]` |
| `V` | **contrastive subspace**: span of `{s_i^(ℓ)}`; top-`d_V` left singular vectors | `V[l]` |
| `Aniso^(ℓ)` | **subspace anisotropy** `‖V^T R V − (tr/dim)·I‖_F / ‖V^T R V‖_F`, `R=(M)^T M` | `aniso[l]` |
| `d_V` | **contrastive subspace dim**: numerical rank of `[s_i]` (σ > 1e-3·σ_max) | `d_V` |

---

## 5. The theorem

At initialization, wide network under μP, linear mode:

```
1 − A^(ℓ)  =  O(1/√n)  +  O(Δ_Gram^(ℓ))      valid while  d_V = o(√n)
```

Stated as a bound for the Lean assembly (Layer 3):

```
1 − A^(ℓ)  ≤  C/√n  +  C'·δ
```

where the two analytic inputs (Layer-2 hypotheses, axiomatized) are:

- **isotropy_at_init**: `‖V^T (Mᵀ M) V − c·I‖ ≤ K / √n` (random-matrix concentration).
- **gram_match**: `‖p^(ℓ) − p^(L)‖ ≤ δ`.

The "persists during training" clause (Track E2) is the open dynamical hypothesis:
`A^(ℓ)` stays near 1 iff `Δ_Gram` falls faster than `Aniso` rises. Not proven; instrumented.

---

## 6. Predictions (empirical acceptance)

- **E1**: `1 − A^(ℓ)` decays as `n^(−1/2)` at init; fitted exponent ∈ [−0.65, −0.35].
- **E2**: `A^(ℓ)` near 1 during SCFF training; `Δ_Gram↓` vs `Aniso↑` overlay + verdict.
- **E3**: alignment holds for `B ≪ √n`, breaks once `d_V ≳ √n`.
