#!/bin/bash
# Phase 1 — unpaired 15-domain pretraining (36k iterations, 8-GPU DDP).
# This is the launcher used for the pretraining run reported in the README.
# Edit ENV_NAME / the config path / the GPU count for your own setup.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-mf}"
CONFIG="${CONFIG:-configs/task3/stargan/bal_pretrain.yaml}"
NPROC="${NPROC:-8}"

# conda (set the path explicitly if `conda info --base` is not available in your shell)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# load DATA_DIR / PREPROCESSED_DIR / OUTPUT_DIR ... from the repo-root .env
set -a; source "$REPO_ROOT/.env"; set +a

cd "$REPO_ROOT/Baseline"

LOGFILE="../logs/pretrain_$(date +%m%d_%H%M).log"
mkdir -p ../logs

echo "Starting Task 3 pretraining (36K steps, ${NPROC}-GPU DDP)"
echo "Config: $CONFIG"
echo "Log: $LOGFILE"

PYTHONUNBUFFERED=1 torchrun --nproc_per_node="$NPROC" scripts/train.py \
  --config "$CONFIG" \
  --mode retro_scratch --dist \
  2>&1 | tee "$LOGFILE"

echo "Training complete. Log: $LOGFILE"
