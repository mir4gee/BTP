#!/usr/bin/env bash
# Table 3 — 1-shot and 5-shot node classification.

set -euo pipefail

CKPT="${1:-checkpoints/riemann_gfm.pt}"
OUT="results/ours/table3"
mkdir -p "${OUT}"

for ds in citeseer pubmed github airports; do
    for k in 1 5; do
        python main.py fewshot --dataset "${ds}" --k "${k}" --checkpoint "${CKPT}" \
            | tee "${OUT}/fs_${ds}_k${k}.log"
    done
done

echo "Logs written to ${OUT}"
