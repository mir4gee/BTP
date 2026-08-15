# Baseline Reproduction — Official RiemannGFM Code

This folder is a **separate track** from our own reimplementation. It exists to reproduce the paper's numbers using the authors' own code, so we have ground-truth measurements to compare our reimplementation (`src/riemann_gfm/`) against.

## What this folder contains

```
baseline_reproduction/
├── README.md                  # this file
├── notebooks/
│   ├── 00_setup.ipynb         # clone official repo, install deps, mount Drive
│   ├── 01_pretrain.ipynb      # official pretrain — ogbn-arxiv + computers + Physics
│   ├── 02_nc.ipynb            # node classification on Citeseer / Pubmed / GitHub / USA
│   ├── 03_lp.ipynb            # link prediction on same four
│   ├── 04_fewshot.ipynb       # 1-shot / 5-shot NC
│   └── 05_results_summary.ipynb  # parse logs, build a Table-1 comparison
└── results/                   # gitignored — logs and captured metrics
```

## What this folder does NOT contain

**No committed copy of the official code.** The notebooks clone `github.com/RiemannGraph/RiemannGFM` at runtime into Colab's `/content/RiemannGFM_official/`. This keeps our git history clean and respects upstream licensing (their code is MIT-licensed; we still don't want it mingled with ours).

## Usage

1. Open `notebooks/00_setup.ipynb` in Colab, switch runtime to T4 GPU, run all cells. This clones the repo, installs the pinned deps, mounts your Drive.
2. Open `notebooks/01_pretrain.ipynb`, run. Takes ~2–3 hours on T4. Checkpoint saved to `MyDrive/RiemannGFM/baseline/checkpoints/`.
3. Open `02_nc.ipynb`, `03_lp.ipynb`, `04_fewshot.ipynb`, run. Each takes 15–30 min.
4. Open `05_results_summary.ipynb` to build a comparison table.

## What we're targeting (Table 1, RiemannGFM row from the paper)

| Metric | Citeseer | Pubmed | GitHub | Airport (USA) |
|--------|---------:|-------:|-------:|--------------:|
| NC ACC | 66.38 | 76.20 | 85.96 | 55.29 |
| NC F1  | 66.41 | 75.83 | 85.57 | 53.27 |
| LP AUC | 99.40 | 94.12 | 89.18 | 93.68 |
| LP AP  | 98.42 | 91.64 | 93.52 | 96.07 |

Once we have baseline numbers close to these, we run the same evals on our own model and put both in one table.

## Why keep the two folders separate?

- `src/riemann_gfm/` — our understanding, defensible in the BTP viva
- `baseline_reproduction/` — the yardstick to check our understanding is right

If our numbers match, great. If they don't, the gap tells us where our reimplementation diverges from the paper.
