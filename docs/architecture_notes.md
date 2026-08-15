# Architecture Notes — Equation ↔ Code Map

This document is the single source of truth for where each paper concept lives in our reimplementation.
Use it when reviewing PRs or when a bug's suspected in a specific paper equation.

## Paper equations → source files

| Paper reference | Equation | Our code | Official code (for cross-check) |
|-----------------|----------|----------|--------------------------------|
| Sec 2, App D    | Eq 1, Eq 21 — CCS definition | `src/riemann_gfm/manifolds/ccs.py :: ConstantCurvatureSpace` | `manifolds/lorentz.py`, `manifolds/spherical.py` |
| Sec 2, App D    | Eq 22 — Exp map | `ccs.py :: expmap` | ditto |
| Sec 2, App D    | Eq 23 — Log map | `ccs.py :: logmap` | ditto |
| Sec 2, App D    | Eq 20 — Parallel transport | `ccs.py :: parallel_transport` | ditto |
| Sec 3.1.1       | Eq 2 — Product tangent bundle | `manifolds/product_bundle.py :: ProductBundle` | (implicit in `model.py` of official) |
| Sec 3.1.2       | Eq 3 (Thm 1) — Manifold-preserving linear | `modules/riemann_linear.py :: RiemannLinear` | `modules/layers.py :: LorentzLinear` |
| Sec 3.1.2       | Eq 4 (Thm 2) — Geometric midpoint | `modules/midpoint.py :: geometric_midpoint` | `modules/layers.py :: LorentzCentroid` |
| Sec 3.1.3       | Eq 5–6, Algo 2 — Cross-geometry attention | `modules/cross_geom_attn.py :: CrossGeometryAttention` | `modules/attention.py` |
| Sec 3.1.4       | Eq 7 — Bundle convolution | `modules/bundle_conv.py :: BundleConvolution` | `modules/conv.py` |
| Sec 3.2         | Eq 8 — Geometric contrastive loss | `modules/contrastive.py :: geometric_contrastive_loss` | `modules/loss.py` |
| Algorithm 1     | Pretrain loop | `training/pretrain.py :: pretrain` | `main.py` (pretrain branch) |
| Sec E.3.2       | Laplacian eigenvector init | `data/init_encoding.py :: laplacian_positional_encoding` | `utils/init.py` |
| Sec E.3.1       | Few-shot protocol | `training/fewshot.py :: FewShotProtocol` | `main.py` (finetune branch) |
| Fig 1(a), Sec 3.1 | Tree + cycle sampler | `data/vocab_sampler.py :: sample_substructures` | `utils/sampler.py` |

## Design choices worth flagging

**Unified curvature formalism.** `ConstantCurvatureSpace(kappa)` handles both $\kappa < 0$ (hyperbolic) and $\kappa > 0$ (hyperspherical) — the sign switches the metric signature. This mirrors App D and eliminates code duplication vs. the official repo where hyperbolic and spherical live in separate files.

**Numerical safety.** All `arccosh`/`arcsinh` inputs are clamped to `[-1 + ε, 1 - ε]` or `[1 + ε, ∞)` respectively. `expmap` uses the tangent-vector norm safely (no zero-division). We use `torch.clamp_min(1e-15)` for norm denominators.

**Batching convention.** Coordinates and encodings are always shape `[..., d + 1]` where the leading `-1` axis is the time-like Lorentz coordinate. This convention is stable across all modules — do not permute.

**Cross-geometry attention direction.** Confirmed from Sec 3.1.3 that on trees the update is bottom-up (children → parent) and on cycles both directions are used (symmetric). Encoded in `CrossGeometryAttention` via a `direction: Literal["tree", "cycle"]` parameter.

**Weight sharing.** The scalar map $\phi$ (Eq 6) is a two-layer MLP with hidden 256, tanh activation. Not shared across layers or geometries (each `CrossGeometryAttention` instance carries its own MLP).

**Bundle convolution message.** The exact rewriting in Eq 7 has a subtle sign in the denominator: $1 + \kappa \langle p_i, p_l\rangle_\kappa$ — we replicate this exactly. When $\kappa < 0$ and points are close, the denominator can approach zero; we add a small offset in the divisor for stability.

## What we deliberately do NOT reimplement

- **Riemannian optimizer.** The paper uses vanilla Adam in Euclidean space (Sec E.3.3). No RSGD or Riemannian-Adam is needed.
- **Custom CUDA kernels.** None used by the paper.
- **Distributed training.** Single-GPU is sufficient at the paper's dataset sizes.

## Numerical constants used across modules

| Constant | Value | Where |
|----------|-------|-------|
| ε (norm floor) | 1e-15 | `manifolds/ccs.py`, `modules/*.py` |
| clamp bound (Lorentz) | ±(1 + 1e-6) for arccosh input | `ccs.py :: logmap` |
| clamp bound (Spherical) | ±(1 - 1e-6) for arccos input | `ccs.py :: logmap` |
| K (Laplacian eigvecs) | 32 (matches paper `dim`) | `data/init_encoding.py` |
