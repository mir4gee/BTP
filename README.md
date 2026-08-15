# QuantAI — RiemannGFM BTP Reproduction

Reproduction of *RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry* (Li Sun et al., WWW 2025) for a BTP project.

- **Paper.** `/home/mir4ge/Desktop/btp.pdf` — a summary is in [`docs/paper_summary.md`](docs/paper_summary.md).
- **Architecture / equation ↔ code map.** [`docs/architecture_notes.md`](docs/architecture_notes.md).
- **Implementation log.** [`docs/implementation_notes.md`](docs/implementation_notes.md).
- **Official repo (baseline).** <https://github.com/RiemannGraph/RiemannGFM> — cloned into `third_party/` (gitignored) by `scripts/setup_baseline.sh`.

## Quick start

Two workflows are supported:

### A. Local development (CPU Docker)

Used for reading the code, running unit tests, and smoke-testing pipelines. No training happens here.

```bash
docker compose build
docker compose run --rm dev pytest tests/                # unit tests
docker compose run --rm dev bash scripts/setup_baseline.sh
```

### B. Training on Colab (GPU)

All Table 1–4 reproductions run on Colab. Notebooks:

| Notebook | Purpose |
|----------|---------|
| `00_install.ipynb` | pip-install pinned deps, mount Drive |
| `01_pretrain.ipynb` | Pre-train on ogbn-arxiv + Computers + Physics (Table 1 recipe) |
| `02_downstream_nc.ipynb` | Node classification — Table 1 NC columns |
| `03_downstream_lp.ipynb` | Link prediction — Table 1 LP columns |
| `04_fewshot.ipynb` | 1-shot / 5-shot — Table 3 |
| `05_ablations.ipynb` | Geometric ablations — Table 2 / Table 7 |
| `06_pretrain_impact.ipynb` | Alternative pre-training data — Table 4 |

Checkpoints and downloaded datasets are stored under `MyDrive/RiemannGFM/` so they survive session resets.

## Repository layout

```
├── Dockerfile, docker-compose.yml, .dockerignore
├── requirements.txt
├── main.py                          # CLI: pretrain / nc / lp / fewshot
├── configs/                         # pretrain + downstream YAML configs
├── docs/                            # paper summary, arch notes, implementation notes
├── notebooks/                       # Colab entrypoints
├── scripts/                         # setup_baseline.sh, run_baseline.sh, eval_*.sh
├── src/riemann_gfm/                 # our reimplementation
│   ├── manifolds/                   # unified Lorentz/Spherical CCS + product bundle
│   ├── modules/                     # Riemannian linear, midpoint, cross-geom attention,
│   │                                #   bundle conv, contrastive loss, RiemannGFM model
│   ├── data/                        # PyG loaders, vocab sampler, Laplacian init
│   ├── training/                    # pretrain, NC, LP, fewshot
│   └── utils/                       # seed, CSV logger
├── tests/                           # pytest — manifold + module unit tests
├── third_party/                     # gitignored — official RiemannGFM clone
├── data/, checkpoints/, results/    # gitignored
```

## Reproducing the paper's numbers

1. **Baseline (official code).**
   ```bash
   bash scripts/setup_baseline.sh
   bash scripts/run_baseline.sh
   ```
   Logs go to `results/baseline/`.

2. **Our reimplementation.** Open Colab, run notebooks 00 → 01 → 02 → 03 → 04. Results printed inline and can be piped to `results/ours/` via `tee`.

3. **Ablations & pre-training impact.** Notebooks 05 and 06 respectively.

Targets (from the paper's Table 1, RiemannGFM row):

| Metric | Citeseer | Pubmed | GitHub | Airport |
|--------|---------:|-------:|-------:|--------:|
| NC ACC | 66.38 | 76.20 | 85.96 | 55.29 |
| NC F1  | 66.41 | 75.83 | 85.57 | 53.27 |
| LP AUC | 99.40 | 94.12 | 89.18 | 93.68 |
| LP AP  | 98.42 | 91.64 | 93.52 | 96.07 |

We aim to match within ±1–2% (paper reports std ≈ 0.5–5%).

## What's not in Phase 1

- Public release of the repo.
- Extensions beyond the paper (see `docs/implementation_notes.md` for a Phase 2 idea list).
- Full comparison against every Table 1 baseline — we treat the paper's baseline numbers as ground truth and only re-run RiemannGFM itself.
