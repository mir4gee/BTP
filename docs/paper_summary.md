# Paper Summary — RiemannGFM (WWW '25)

**Title.** *RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry*
**Authors.** Li Sun, Zhenhao Huang, Suyang Zhou, Qiqi Wan (North China Electric Power University), Hao Peng (Beihang), Philip S. Yu (UIC).
**Venue.** ACM Web Conference (WWW) 2025, Sydney, April 28 – May 2, 2025.
**Official code.** <https://github.com/RiemannGraph/RiemannGFM>

---

## 1. Motivation

Foundation models in NLP and vision succeed because they learn a shared *vocabulary* (subword tokens, image patches) that transfers across datasets. Graphs have no such universal alphabet, so existing Graph Foundation Models (GFMs) fall into two camps:

1. **LLM-coupled GFMs** (OFA, LLaGA, OpenGraph, GraphAny). Describe nodes/edges in natural language, then feed the sequence into an LLM. They work only when the target graph carries fruitful textual attributes — **most real-world graphs do not** — and the sequential description discards structural information.
2. **Structure-only GFMs** (GCOPE and follow-ups). Train on multi-domain graphs but stay in Euclidean space, which is a poor fit for hierarchical and cyclic structures that dominate real graphs.

The paper asks: **can we build a GFM that learns purely structural knowledge, works on any graph (with or without text), and respects the non-Euclidean geometry of graph substructures?**

---

## 2. Central Insight — Structural Vocabulary + Riemannian Geometry

The authors introduce a **structural vocabulary** made of just two primitives:

- **Trees** — model hierarchical / acyclic components (Fig 1a).
- **Cycles** — model relational loops (triangles, quadruples, …).

*Definition 1.* A collection of substructures is a **structural vocabulary** if any graph can be constructed from them. Trees + cycles satisfy this — any connected component is a tree unless it contains a cycle, in which case one can decompose it into a tree plus a cycle basis.

The key second step is aligning each primitive with a matching Riemannian geometry:

| Primitive | Geometry | Curvature | Why |
|-----------|----------|-----------|-----|
| Tree      | Hyperbolic space $\mathcal{H}$ | $\kappa < 0$ | Volume grows exponentially with radius — same as the number of nodes at depth $d$ in a $b$-ary tree. Bounded distortion for tree embedding. |
| Cycle     | Hyperspherical space $\mathcal{S}$ | $\kappa > 0$ | Rotational invariance matches the cyclic symmetry of a closed loop. |

---

## 3. Formalism — Unified Lorentz/Spherical Model

The paper works in a single unified constant-curvature space (Eq 1, App D):

$$\mathcal{L}^d_\kappa = \{x = (x_t, x_s)^\top \in \mathbb{R}^{d+1} : \langle x, x \rangle_\kappa = 1/\kappa,\ x_t > 0,\ x_s \in \mathbb{R}^d\}$$

with **curvature-aware inner product**

$$\langle x, y \rangle_\kappa := \operatorname{sgn}(\kappa) x_t y_t + x_s^\top y_s.$$

- $\kappa < 0$ yields the Lorentz model of hyperbolic space (Minkowski signature).
- $\kappa > 0$ yields the Spherical model of hyperspherical space (Euclidean signature).

This single formalism unifies both geometries so the code path is shared. Exp / Log / Parallel Transport (App D Eq 22, 23, 20) follow.

---

## 4. Novel Representation Space — Product Tangent Bundle

For each node they carry **both a manifold coordinate and a tangent-space encoding**:

- $p_i \in \mathcal{H}$ or $\mathcal{S}$ (position within a substructure → *local* geometry).
- $z_i \in T_{p_i}\mathcal{H}$ or $T_{p_i}\mathcal{S}$ (global encoding).

The overall representation is a **product tangent bundle** (Eq 2):

$$\mathcal{P}^{d_P} = \big(\mathcal{H}^{d_H}_{\kappa_H} \otimes T\mathcal{H}^{d_H}_{\kappa_H}\big) \otimes \big(\mathcal{S}^{d_S}_{\kappa_S} \otimes T\mathcal{S}^{d_S}_{\kappa_S}\big),\quad d_P = 2d_H + 2d_S.$$

This is the first tangent-bundle formulation on graphs. The manifold factor keeps *substructure* information; the tangent factor keeps *global* information.

---

## 5. Core Riemannian Operations

**Theorem 1 (Manifold-preserving linear op, Eq 3).** For any $W \in \mathbb{R}^{d_1 \times d_2}$,

$$f_W(x) = \begin{bmatrix} 1 & 0^\top \\ 0 & \alpha W \end{bmatrix} x,\quad \alpha = \sqrt{\frac{\kappa^{-1} - \operatorname{sgn}(\kappa) x_t^2}{\lVert W x_s \rVert^2}},$$

maps $\mathcal{L}^{d_1}_\kappa \to \mathcal{L}^{d_2}_\kappa$ — i.e. it is a Riemannian linear layer with **no auxiliary tangent-space excursion**. This avoids the isometry loss of prior HGCN-style layers that Exp / Log around every op.

**Theorem 2 (Geometric midpoint, Eq 4).** The weighted arithmetic mean divided by its curvature-aware norm,

$$\operatorname{mid}_\kappa(\{x_i, v_i\}) = \frac{1}{\sqrt{|\kappa|}} \sum_i \frac{v_i x_i}{\lVert \sum_j v_j x_j \rVert_\kappa},$$

is exactly the geometric midpoint (Fréchet mean under squared Riemannian distance). This is the aggregation primitive used everywhere else.

**Parallel transport** (App D Eq 20) — bridges tangent spaces at different manifold points so message passing on graphs (which lives in tangent space) can aggregate encodings whose tangent spaces are not naïvely compatible.

---

## 6. Model — Universal Riemannian Layer

Each layer has two sub-modules (Fig 1b, c):

**6.1 Vocabulary Learning — Cross-Geometry Attention (Eq 5–6, Algo 2).**
- Trees are embedded in $\mathcal{H}$, cycles in $\mathcal{S}$.
- Attention key/query/value are all obtained via the Riemannian linear op $f_W$.
- **Cross-geometry.** The *query* for the hyperbolic substructure comes from the counterpart node's *hyperspherical* coordinate, and vice versa. This is what "cross-geometry" attention means — it forces the two geometries to inform each other. The ablation in Sec 4.2.3 shows this beats a single-geometry variant.
- On trees the attention is unidirectional (bottom-up), on cycles it is symmetric.
- Node coordinate update = geometric midpoint of the descendant / neighbour attention outputs.

**6.2 Global Learning — Bundle Convolution (Eq 7).**

$$\operatorname{BC}_{p_i}(\{p_l, z_l\}_{l \in \Lambda}) = \sum_{l \in \Lambda} \Big( \alpha_{il} z_l - \frac{\kappa \alpha_{il}\langle z_i, p_l\rangle_\kappa}{1 + \kappa \langle p_i, p_l\rangle_\kappa}(p_i + p_l) \Big)$$

This is a message-passing step **on the tangent bundle**: each neighbour's tangent encoding $z_l$ is parallel-transported to the tangent space at $p_i$, then aggregated with weights $\alpha_{il}$ derived from Eq 6. The parallel transport is the canonical way to resolve tangent-space incompatibility (Fig 2).

**Stacking.** Two universal Riemannian layers stacked, following the paper's config.

---

## 7. Self-supervised Objective — Geometric Contrastive Learning (Eq 8)

**Key idea.** No graph augmentation is needed because the hyperbolic and hyperspherical views of the same node are already two *different geometric perspectives* of the same object — they form natural positive pairs.

Encodings are first parallel-transported to a **shared tangent space at the north pole $o$** of each factor so they can be compared:

$$\mathcal{J}(H, S) = -\sum_{i=1}^N \log \frac{\exp(\langle \operatorname{PT}_{p_i^H \to o}(z_i^H),\ \operatorname{PT}_{p_i^S \to o}(z_i^S) \rangle)}{\sum_{j=1}^N \exp(\langle \operatorname{PT}_{p_i^H \to o}(z_i^H),\ \operatorname{PT}_{p_j^S \to o}(z_j^S) \rangle)}.$$

Symmetrised objective: $\mathcal{J}_0 = \mathcal{J}(H, S) + \mathcal{J}(S, H)$.

Complexity is $\mathcal{O}(|\mathcal{V}|^2 + |\mathcal{E}|)$ but mini-batch training (batch 32, SAGE-style [20, 10] sampler) keeps this tractable.

---

## 8. Pre-training Recipe (Algorithm 1)

1. Initialize node coordinates on the CCSs and node encodings via **Laplacian eigenvectors** ($K$ largest eigenvectors of $L = I - D^{-1/2} A D^{-1/2}$) — normalises differing graphs with a common $K$-dimensional feature.
2. Sample tree + cycle substructures.
3. For each geometry: cross-geometry attention updates node coordinates; bundle convolution updates node encodings.
4. Induce graph-level encoding via geometric midpoint + parallel transport.
5. Compute contrastive loss and update parameters (Adam, lr 0.01).

**Pretraining datasets.** ogbn-arxiv (169k nodes), Physics (34k), Amazon-Computers (13k).
**Hyperparameters.** dim = 32 per factor → total $32 \times 4 = 128$; hidden of scalar map φ = 256; layers = 2; dropout = 0.1; lr = 0.01; batch = 32; SAGE sampler [20, 10]; 3 epochs × 3 iterations.

---

## 9. Experimental Results

**Downstream datasets.** Citeseer (text-attributed), Pubmed (text-attributed), GitHub (mixed), Airports (structure-only).

### Table 1 — Cross-domain transfer (headline)
- On **non-attributed structural** datasets (GitHub, Airports) RiemannGFM is the clear winner on both node classification and link prediction (e.g., GitHub NC ACC 85.96 vs. best GFM competitor 45–77).
- On text-attributed datasets (Citeseer, Pubmed) RiemannGFM is *competitive* with text-based GFMs but not the top — expected because it uses no textual features, only structural.
- On link prediction it is state-of-the-art across all four datasets (AUC 89–99).

### Table 2 / Table 7 — Geometric ablation
Product bundle $(\mathcal{H}^{32}_{-1}, \mathcal{S}^{32}_1)$ beats $(\mathcal{H}, \mathcal{H})$, $(\mathcal{S}, \mathcal{S})$, and Euclidean variants. Confirms the "trees→hyperbolic, cycles→hyperspherical" alignment is the source of performance.

### Table 3 — Few-shot NC (1-shot, 5-shot)
RiemannGFM significantly outperforms all GFMs and self-supervised baselines. LLM-based GFMs (OpenGraph, LLaGA) exhibit *negative transfer* on GitHub / Airports because they lean on textual attributes absent from those graphs.

### Table 4 — Impact of pre-training dataset
Robust across Flickr / AComp / WikiCS pre-training (variance ≤ 1.5% on downstream LP AUC). Structural knowledge is far less domain-dependent than textual GFMs.

### Figure 5 — Visualization (Cora)
t-SNE of pretrained RiemannGFM node encodings shows cleaner class separation than a fully-trained specialised GCN.

---

## 10. Contributions

**A.** First GFM that works on graphs *without* textual attributes — grounded in structural geometry.
**B.** New representation: product tangent bundle unifying hyperbolic + hyperspherical geometries with a shared Riemannian linear op.
**C.** Extensive empirical validation across NC, LP, few-shot, and cross-domain transfer.

---

## 11. Limitations & Directions

- Fixed structural vocabulary — only trees + cycles. A richer vocabulary (k-cliques, motifs) could improve expressiveness.
- Two-factor product bundle — extends naturally to more factors (SPD manifolds for correlations, Grassmann for subspaces).
- Text-attributed setting under-explored — hybridising with an LLM path is an open direction.
- Complexity $\mathcal{O}(|\mathcal{V}|^2)$ in the contrastive loss is mitigated by mini-batching but could be attacked with sparse-negative sampling.

---

## 12. Notation Cheat-Sheet

| Symbol | Meaning |
|--------|---------|
| $\mathcal{H}, \mathcal{S}$ | Hyperbolic / Hyperspherical space |
| $\mathcal{L}^d_\kappa$ | Unified Lorentz/Spherical model, dim $d$, curvature $\kappa$ |
| $T_x \mathcal{M}$ | Tangent space at $x$ |
| $\mathcal{TM}$ | Tangent bundle |
| $p_i$ | Node coordinate on manifold factor |
| $z_i$ | Node encoding in tangent factor |
| $o$ | North pole $[1/\sqrt{|\kappa|}, 0, \ldots, 0]^\top$ |
| $\operatorname{Exp}_x, \operatorname{Log}_x$ | Exponential / logarithmic maps |
| $\operatorname{PT}_{x \to y}$ | Parallel transport along geodesic $x \to y$ |
| $f_W$ | Manifold-preserving linear op (Thm 1) |
| $\operatorname{mid}_\kappa$ | Geometric midpoint (Thm 2) |
