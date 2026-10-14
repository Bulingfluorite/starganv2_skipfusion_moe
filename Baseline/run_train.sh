#!/bin/bash
# Train a baseline model.
#
# Usage:
#   bash run_train.sh <config> <mode> [gpu_id]
#
# Modes:
#   retro_scratch   — Unpaired pretraining on Training_retrospective
#   pro_scratch     — Supervised training from scratch on Training_prospective
#   pro_pretrained  — Finetune a pretrained model on Training_prospective
#
# Examples:
#   bash run_train.sh configs/task1/cyclegan/0.1T_to_7T_T1W.yaml retro_scratch 0
#   bash run_train.sh configs/task1/cyclegan/0.1T_to_7T_T1W.yaml pro_pretrained 0
#   bash run_train.sh configs/task1/cyclegan/0.1T_to_7T_T1W.yaml pro_scratch 0

set -e

CONFIG=${1:?Usage: bash run_train.sh <config> <mode> [gpu_id]}
MODE=${2:?Usage: bash run_train.sh <config> <mode> [gpu_id]}
GPU=${3:-0}

echo "Config: $CONFIG"
echo "Mode:   $MODE"
echo "GPU:    $GPU"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python scripts/train.py --config "$CONFIG" --mode "$MODE" --device cuda
