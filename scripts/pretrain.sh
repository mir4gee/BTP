#!/usr/bin/env bash
# Run our reimplementation's pretraining locally (CPU Docker or Colab).
# For Colab, prefer the notebook UI in notebooks/01_pretrain.ipynb.

set -euo pipefail

CONFIG="${1:-configs/pretrain.yaml}"
CKPT="${2:-checkpoints/riemann_gfm.pt}"

python main.py pretrain --config "${CONFIG}" --checkpoint "${CKPT}"
