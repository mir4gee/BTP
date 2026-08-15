#!/usr/bin/env bash
# Clone the official RiemannGFM repository into third_party/ so we can run it
# as a baseline. third_party/ is gitignored — nothing here gets committed to
# our repo.
#
# Usage: bash scripts/setup_baseline.sh

set -euo pipefail

REPO_URL="https://github.com/RiemannGraph/RiemannGFM.git"
TARGET_DIR="third_party/RiemannGFM"

if [[ -d "${TARGET_DIR}/.git" ]]; then
    echo "[setup_baseline] Baseline already present at ${TARGET_DIR}. Pulling latest."
    git -C "${TARGET_DIR}" pull --ff-only
else
    echo "[setup_baseline] Cloning ${REPO_URL} into ${TARGET_DIR}"
    mkdir -p third_party
    git clone --depth 1 "${REPO_URL}" "${TARGET_DIR}"
fi

# Create the datasets/ dir the baseline expects, wiring it to our shared data/
# directory so we don't download the same graphs twice.
if [[ ! -e "${TARGET_DIR}/datasets" ]]; then
    ln -s "$(pwd)/data" "${TARGET_DIR}/datasets"
    echo "[setup_baseline] Symlinked ${TARGET_DIR}/datasets -> $(pwd)/data"
fi

# Same for checkpoints.
if [[ ! -e "${TARGET_DIR}/checkpoints" ]]; then
    ln -s "$(pwd)/checkpoints" "${TARGET_DIR}/checkpoints"
    echo "[setup_baseline] Symlinked ${TARGET_DIR}/checkpoints -> $(pwd)/checkpoints"
fi

echo "[setup_baseline] Done. Next: bash scripts/run_baseline.sh"
