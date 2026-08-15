# CPU dev image for RiemannGFM reproduction.
#
# Actual training (Table 1–4 reproductions) happens on Google Colab where a
# GPU is available. This image exists so contributors can:
#   * open the repo end-to-end without a GPU,
#   * run the manifold / linear-op / midpoint unit tests,
#   * run smoke pretraining on Citeseer/Airports to validate the pipeline,
#   * run downstream evaluation heads on saved checkpoints.
#
# Base image is slim to keep the pull small; scientific wheels are all
# available as manylinux binaries so no compiler toolchain is needed.

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git is needed for `pip install` from VCS and for `setup_baseline.sh`
# to clone the official RiemannGFM repo into third_party/.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install torch first (its wheel is huge) so this layer is cached across
# subsequent edits to requirements.txt.
RUN pip install torch==2.0.0

# torch_scatter needs to see torch already installed. Point at the CPU wheel
# index for torch 2.0.0.
RUN pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt \
    || pip install -r requirements.txt

# Copy the source tree last so code edits don't invalidate the deps layers.
COPY . .

# Make src/ importable without an install step so contributors can iterate.
ENV PYTHONPATH=/workspace/src

CMD ["/bin/bash"]
