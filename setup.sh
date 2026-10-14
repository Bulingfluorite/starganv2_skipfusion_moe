#!/usr/bin/env bash
# One-shot environment setup for the MRIxFields2026 Task 3 codebase.
#
#   bash setup.sh
#
# Creates the conda environment from environment.yml, installs the Baseline/
# package in editable mode, and prepares .env from .env.example.
# Override the environment name with:  ENV_NAME=myenv bash setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ENV_NAME="${ENV_NAME:-mf}"

if ! command -v conda >/dev/null 2>&1; then
    echo "error: conda not found on PATH — install Miniconda/Anaconda first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "==> creating conda environment '$ENV_NAME' from environment.yml"
if ! conda env create -n "$ENV_NAME" -f environment.yml; then
    echo "==> environment may already exist — updating it instead"
    conda env update -n "$ENV_NAME" -f environment.yml
fi

conda activate "$ENV_NAME"

echo "==> installing Baseline/ in editable mode"
pip install -e Baseline

if [ -f .env ]; then
    echo "==> .env already exists — leaving it untouched"
else
    cp .env.example .env
    echo "==> wrote .env from .env.example"
fi

cat <<EOF

Setup finished. Next steps:

    conda activate $ENV_NAME
    vim .env                 # point DATA_DIR / PREPROCESSED_DIR / OUTPUT_DIR / ... at your data
    source .env
    cd Baseline
    python scripts/train.py --config configs/task3/stargan/bal_pretrain.yaml --mode retro_scratch

For multi-GPU training, prefix the last command with e.g.:
    torchrun --nproc_per_node=8 --dist
EOF
