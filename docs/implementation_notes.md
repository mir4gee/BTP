# Implementation Notes & Observations

Living document — append as we hit issues, make deviations, or find improvements.

## Environment

- **Local dev.** Docker CPU image `riemann-gfm:cpu` (see `Dockerfile`). Used for unit tests, smoke pretraining on tiny datasets, and downstream evaluation of Colab-trained checkpoints.
- **Training.** Google Colab notebooks in `notebooks/`. Each notebook installs the CUDA torch build inline (see the first cell); Drive-mounted `/content/drive/MyDrive/RiemannGFM` holds checkpoints across sessions.

## Setup issues encountered

_(Populate as we hit them.)_

**torch_scatter wheel resolution.** The Python 3.10 + torch 2.0.0 CPU wheel exists at `https://data.pyg.org/whl/torch-2.0.0+cpu.html`. On Colab (torch 2.0.0+cu118) use `https://data.pyg.org/whl/torch-2.0.0+cu118.html`. Baked into `Dockerfile` and notebook install cells respectively.

**No preinstalled OGB on Colab.** `pip install ogb==1.3.6` triggers a rebuild of scikit-learn on first Colab session — safe but slow (~2 min). Notebooks pre-install everything before mounting Drive.

## Deviations from paper

_(Track only what actually differs from the paper description; empty entry is OK.)_

- **None so far.** Config in `configs/pretrain.yaml` mirrors `scripts/pretrain.sh` from the official repo exactly.

## Observations while reading the paper

- The claim that "trees + cycles form a structural vocabulary" (Def 1) is proven only by construction — every connected component reduces to a spanning tree + a cycle basis. Fine for a foundation model but worth calling out to advisors.
- The cross-geometry attention design is unusual: query on geometry A is drawn from geometry B. This is what forces the two Riemannian streams to talk. If we were to add a third geometry we'd need to decide the pairing (e.g., round-robin cross-attention).
- The contrastive objective has zero augmentation cost — this is a genuine engineering win over prior graph SSL methods that spend significant compute on augmentation views.

## Ideas for extensions (Phase 2 seed list)

1. **Richer vocabulary** — add k-cliques or motifs as a third primitive, embedded in a mixed-curvature or SPD space.
2. **Learnable curvatures** — treat $\kappa_H, \kappa_S$ as trainable parameters (currently fixed to $-1, +1$).
3. **Fine-grained ablations** — measure contribution of parallel transport vs. naïve tangent-space aggregation by disabling PT in `bundle_conv.py`.
4. **Text-attributed variant** — concatenate LLM-derived features into $z_i$ initial encoding to test whether structural pretraining and textual features compose.
5. **Scaling study** — pre-train on larger graphs (ogbn-products) and measure the marginal downstream lift.

## Version history

- 2026-08-16: Repo scaffolded. Docker + docs + baseline scripts in place. Manifold implementation in progress.
