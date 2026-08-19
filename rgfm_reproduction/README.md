# R-GFM Reproduction (official code, Colab)

This folder is a **separate, independent track** from both `src/riemann_gfm/` (our RiemannGFM reimplementation) and `baseline_reproduction/` (the official RiemannGFM baseline). It reproduces a *different* paper — **R-GFM**: *Learning Graph Foundation Models on Riemannian Graph-of-Graphs* (Liu, Ding & Xie, ICML 2026 preprint) — using the authors' own code at [`USTC-DataDarknessLab/R-GFM`](https://github.com/USTC-DataDarknessLab/R-GFM).

R-GFM is cited in our RiemannGFM work as a follow-up that generalizes fixed-hop, fixed-manifold GFMs (like RiemannGFM) to adaptive-hop Graph-of-Graphs with dynamic Riemannian MoE routing. This track exists to get ground-truth R-GFM numbers for comparison.

## Why Colab, not local

This machine's GPU is a 4GB RTX 3050; the paper's experiments ran on A100 80GB GPUs, and R-GFM requires compiling a custom CUDA extension (`graph_aug/`) which needs a CUDA toolkit (`nvcc`) that isn't installed locally. Colab provides both a working CUDA toolkit and (with Pro/Pro+) A100-class GPUs, matching the `baseline_reproduction/` precedent already in this repo.

## What this folder contains

```
rgfm_reproduction/
├── README.md                        # this file
├── notebooks/
│   ├── 00_setup.ipynb                # clone repo, install deps, build CUDA ext, mount Drive
│   ├── 01_node_classification.ipynb  # 1-shot NC, leave-one-dataset-out — Table 1
│   ├── 02_fewshot.ipynb              # 3-shot / 5-shot NC — Tables 2 & 3
│   ├── 03_link_prediction.ipynb      # link prediction — Table 9 (Table 5 is a subset)
│   └── 04_results_summary.ipynb      # parse logs, build a paper-vs-ours comparison CSV
└── results/                           # gitignored — logs and captured metrics (Drive-backed)
```

**No committed copy of the official code.** `00_setup.ipynb` clones `github.com/USTC-DataDarknessLab/R-GFM` at runtime into Colab's `/content/R-GFM/`, matching the `baseline_reproduction/` convention of not mingling upstream code with ours.

## Usage

1. Open `notebooks/00_setup.ipynb` in Colab, switch runtime to GPU, run all cells. Clones the repo, installs dependencies (inferred from actual imports — see caveat below), builds the CUDA extension, mounts Drive.
2. Open `01_node_classification.ipynb`, run. 8 datasets × leave-one-dataset-out pretraining + 1-shot fine-tuning — expect this to take a while per dataset (150 epochs of Stage-1 + Stage-2 training on 7-9 source graphs each time).
3. Open `02_fewshot.ipynb` for 3-shot/5-shot (16 more full runs — the most expensive notebook here).
4. Open `03_link_prediction.ipynb` for the 7-dataset link-prediction table.
5. Open `04_results_summary.ipynb` to build the comparison table/CSV.

## Repo caveats found while preparing this track

- The official README references a `requirements.txt` that **does not exist** in the repo (verified against the `main` branch tree). `00_setup.ipynb` installs the dependency set inferred from actual source imports instead: `torch==2.8.0`, `torch_geometric`, `torch_scatter`/`torch_sparse` (via the matching PyG wheel index), `geoopt`, `ogb`, `scikit-learn`.
- `graph_aug/cuda_backend.py` is dead code — it JIT-compiles from `cuda_kernels/*_host.cpp` files that don't exist in the repo, and nothing imports it. The actual runtime path (`utils/data/augmentation.py` → `graph_aug.graph_aug_cuda`) uses the precompiled extension from `setup.py build_ext --inplace`, exactly as the README's build instructions say, and that path is internally consistent.
- The paper's leave-one-dataset-out protocol (pretrain on N-1 datasets, evaluate on the held-out one) is **not** a CLI flag — it's hardcoded in `trainers/node2graph_trainer.py`: passing `--dataset X` automatically pretrains on every other dataset in `["wisconsin","texas","cornell","cora","citeseer","pubmed","computers","photo","chameleon","squirrel"]`.
- Chameleon/Squirrel are excluded as **targets** in the notebooks here (matching the paper's main Table 1, due to documented duplicate-node/train-test-leakage issues — Platonov et al. 2023) but still serve as automatic pretraining sources for the other 8.

## Targets (paper's R-GFM row)

1-shot NC (Table 1, accuracy %):

| Wisconsin | Cornell | Citeseer | Cora | Pubmed | Computers | Photos | Texas |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 35.41 | 36.71 | 57.54 | 49.50 | 49.80 | 52.30 | 61.08 | 32.36 |

Link prediction (Table 9, AUC-ROC %):

| Wisconsin | Cornell | Citeseer | Pubmed | Cora | Photos | Texas |
|---:|---:|---:|---:|---:|---:|---:|
| 84.15 | 85.90 | 90.88 | 88.62 | 89.27 | 81.53 | 87.94 |

3-shot/5-shot targets are in `02_fewshot.ipynb`'s closing markdown cell.

We aim to match within a small margin of the paper's reported std (typically 1–12% depending on dataset — WebKB datasets like Wisconsin/Cornell/Texas have small test sets and high variance even in the paper's own numbers).
