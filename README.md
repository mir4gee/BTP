# QuantAI — RiemannGFM BTP Reproduction

Reproduction of *RiemannGFM: Learning a Graph Foundation Model from Riemannian Geometry* (Li Sun et al., WWW 2025) for a BTP project.

- **Paper.** `/home/mir4ge/Desktop/btp.pdf` — a summary is in [`docs/paper_summary.md`](docs/paper_summary.md).
- **Architecture / equation ↔ code map.** [`docs/architecture_notes.md`](docs/architecture_notes.md).
- **Implementation log.** [`docs/implementation_notes.md`](docs/implementation_notes.md).
- **Official repo (baseline).** <https://github.com/RiemannGraph/RiemannGFM> — cloned into `third_party/` (gitignored) by `scripts/setup_baseline.sh`.
- **R-GFM reproduction (related work).** <https://github.com/USTC-DataDarknessLab/R-GFM> — a follow-up GFM cited alongside RiemannGFM; reproduced independently in [`rgfm_reproduction/`](rgfm_reproduction/README.md) on Colab.

## Quick start

Three tracks are supported. Track A verifies code correctness locally, tracks B and C both run on Colab and produce the paper's numbers using different codebases. A fourth, independent track (`rgfm_reproduction/`) reproduces a different but related paper, R-GFM — see its own README.

### A. Local development (CPU Docker)

Read the code, run unit tests, smoke-test pipelines. No real training.

```bash
docker compose build
docker compose run --rm dev pytest tests/
```

### B. Baseline reproduction on Colab (official code)

`baseline_reproduction/` runs the **paper authors' own code** on Colab. Use this to get ground-truth numbers we can compare against.

| Notebook | Purpose |
|----------|---------|
| `baseline_reproduction/notebooks/00_setup.ipynb` | Clone official repo, install pinned deps, mount Drive |
| `baseline_reproduction/notebooks/01_pretrain.ipynb` | Run `scripts/pretrain.sh` — ogbn-arxiv + Computers + Physics |
| `baseline_reproduction/notebooks/02_nc.ipynb` | Node classification — Table 1 NC columns |
| `baseline_reproduction/notebooks/03_lp.ipynb` | Link prediction — Table 1 LP columns |
| `baseline_reproduction/notebooks/04_fewshot.ipynb` | 1-shot / 5-shot — Table 3 |
| `baseline_reproduction/notebooks/05_results_summary.ipynb` | Parse logs, build a paper-vs-ours comparison CSV |

See `baseline_reproduction/README.md` for details.

### C. Our reimplementation on Colab (`src/riemann_gfm/`)

Same experiments but using our own from-scratch code.

| Notebook | Purpose |
|----------|---------|
| `notebooks/00_install.ipynb` | pip-install pinned deps, mount Drive |
| `notebooks/01_pretrain.ipynb` | Pre-train on ogbn-arxiv + Computers + Physics |
| `notebooks/02_downstream_nc.ipynb` | Node classification — Table 1 NC columns |
| `notebooks/03_downstream_lp.ipynb` | Link prediction — Table 1 LP columns |
| `notebooks/04_fewshot.ipynb` | 1-shot / 5-shot — Table 3 |
| `notebooks/05_ablations.ipynb` | Geometric ablations — Table 2 / Table 7 |
| `notebooks/06_pretrain_impact.ipynb` | Alternative pre-training data — Table 4 |

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
