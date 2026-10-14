#!/bin/bash
# Convenience wrapper for Task 3 StarGAN v2 training.
#
#   ./train.sh [mode] [run_name] [gpus] [config]
#
# modes:  retro_scratch  — unpaired 15-domain pretraining (Phase 1)
#         pro_pretrained — paired fine-tuning from a Phase-1 checkpoint (Phase 2)
#         pro_scratch    — paired training from random init
# example: ./train.sh retro_scratch run1 8 configs/task3/stargan/bal_pretrain.yaml

set -e

MODE=${1:-retro_scratch}
RUN_NAME=${2:-run1}
GPUS=${3:-2}
CONFIG=${4:-configs/task3/stargan/bal_pretrain.yaml}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Activate the conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME:-mf}"

# Paths come from the repo-root .env (see .env.example); fall back to a local runs/ dir
if [ -f "$REPO_ROOT/.env" ]; then
    set -a; source "$REPO_ROOT/.env"; set +a
fi
: "${OUTPUT_DIR:=${REPO_ROOT}/Baseline/runs}"
OUTPUT_DIR="${OUTPUT_DIR}/${RUN_NAME}"
export OUTPUT_DIR

cd "$REPO_ROOT/Baseline"

OUTPUT_SUBDIR="${OUTPUT_DIR}/task3_any_to_any_multimodal/stargan_v2/${MODE}"
mkdir -p "$OUTPUT_SUBDIR"

echo "Starting StarGAN v2 ${MODE} training at $(date)"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_SUBDIR}/"

torchrun --nproc_per_node="${GPUS}" scripts/train.py \
    --config "${CONFIG}" \
    --mode "${MODE}" \
    --dist 2>&1 | tee "${OUTPUT_SUBDIR}/train.log"

echo "Training completed at $(date)"
