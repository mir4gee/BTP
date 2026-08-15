#!/usr/bin/env bash
# Run the official RiemannGFM baseline end-to-end:
#   1) pretrain on ogbn-arxiv + Physics + Amazon-Computers,
#   2) evaluate node classification on Citeseer + Pubmed + GitHub + Airports,
#   3) evaluate link prediction on the same four downstream datasets.
#
# Results are written to results/baseline/.
#
# Prereqs: bash scripts/setup_baseline.sh
# Usage:   bash scripts/run_baseline.sh

set -euo pipefail

BASELINE_DIR="third_party/RiemannGFM"
RESULTS_DIR="results/baseline"

if [[ ! -d "${BASELINE_DIR}" ]]; then
    echo "[run_baseline] ${BASELINE_DIR} not found. Run scripts/setup_baseline.sh first."
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

pushd "${BASELINE_DIR}" > /dev/null

echo "[run_baseline] Pretraining (ogbn-arxiv + Physics + Amazon-Computers)"
source ./scripts/pretrain.sh 2>&1 | tee "../../${RESULTS_DIR}/pretrain.log"

echo "[run_baseline] Node classification downstream"
for ds in citeseer pubmed github airports; do
    if [[ -f "./scripts/NC/${ds}.sh" ]]; then
        echo "[run_baseline]   NC/${ds}"
        source "./scripts/NC/${ds}.sh" 2>&1 | tee "../../${RESULTS_DIR}/nc_${ds}.log"
    else
        echo "[run_baseline]   (skip NC/${ds} — script not in baseline repo)"
    fi
done

echo "[run_baseline] Link prediction downstream"
for ds in citeseer pubmed github airports; do
    if [[ -f "./scripts/LP/${ds}.sh" ]]; then
        echo "[run_baseline]   LP/${ds}"
        source "./scripts/LP/${ds}.sh" 2>&1 | tee "../../${RESULTS_DIR}/lp_${ds}.log"
    else
        echo "[run_baseline]   (skip LP/${ds} — script not in baseline repo)"
    fi
done

popd > /dev/null

echo "[run_baseline] Done. Logs under ${RESULTS_DIR}/"
