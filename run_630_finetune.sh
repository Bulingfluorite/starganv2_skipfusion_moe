#!/bin/bash
# Phase 2 — paired fine-tuning (50 epochs) from a Phase-1 checkpoint.
# This is the launcher used for the fine-tuning run reported in the README.
# Point CKPT at your own Phase-1 best checkpoint (or export PRETRAIN_CKPT).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-mf}"
CONFIG="${CONFIG:-configs/task3/stargan/bal_finetune.yaml}"
NPROC="${NPROC:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

set -a; source "$REPO_ROOT/.env"; set +a

: "${OUTPUT_DIR:=${REPO_ROOT}/Baseline/runs}"
CKPT="${PRETRAIN_CKPT:-${OUTPUT_DIR}/stargan_v2_expert_fix/stargan_v2/retro_scratch/weights/checkpoint_best_26000.pth}"
LOGFILE="../logs/finetune_$(date +%m%d_%H%M).log"

cd "$REPO_ROOT/Baseline"
mkdir -p ../logs

if [ ! -f "$CKPT" ]; then
  echo "ERROR: Phase-1 checkpoint not found at $CKPT" >&2
  echo "Run run_630_pretrain.sh first, then export PRETRAIN_CKPT=<path>." >&2
  exit 1
fi

echo "Starting Task 3 fine-tuning (50 epochs, from checkpoint)"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Log: $LOGFILE"

PYTHONUNBUFFERED=1 torchrun --nproc_per_node="$NPROC" scripts/train.py \
  --config "$CONFIG" \
  --mode pro_pretrained \
  --pretrain_ckpt "$CKPT" \
  --dist \
  2>&1 | tee "$LOGFILE"

echo "Training complete. Log: $LOGFILE"
