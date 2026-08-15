#!/usr/bin/env bash
# Reproduce Table 1 (NC + LP on Citeseer, Pubmed, GitHub, Airports).
# Assumes a pretrained checkpoint exists at $CKPT.

set -euo pipefail

CKPT="${1:-checkpoints/riemann_gfm.pt}"
OUT="results/ours/table1"
mkdir -p "${OUT}"

for ds in citeseer pubmed github airports; do
    echo "==== ${ds} ===="
    python main.py nc --dataset "${ds}" --checkpoint "${CKPT}" | tee "${OUT}/nc_${ds}.log"
    python main.py lp --dataset "${ds}" --checkpoint "${CKPT}" | tee "${OUT}/lp_${ds}.log"
done

echo "Logs written to ${OUT}"
