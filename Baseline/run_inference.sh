#!/bin/bash
# Run inference with a trained model.
#
# Usage:
#   bash run_inference.sh <config> <checkpoint> <input_dir> <output_dir> [gpu_id]
#
# Examples:
#   bash run_inference.sh \
#       configs/task1/cyclegan/0.1T_to_7T_T1W.yaml \
#       $OUTPUT_DIR/task1_0.1T_to_7T_T1W/cyclegan/retro_scratch/weights/checkpoint_epoch100.pth \
#       $DATA_DIR/Testing_prospective/T1W/0.1T/ \
#       $INFERENCE_DIR/task1_0.1T_to_7T_T1W/cyclegan/

set -e

CONFIG=${1:?Usage: bash run_inference.sh <config> <checkpoint> <input_dir> <output_dir> [gpu_id]}
CHECKPOINT=${2:?Missing checkpoint path}
INPUT_DIR=${3:?Missing input directory}
OUTPUT_DIR=${4:?Missing output directory}
GPU=${5:-0}

echo "Config:     $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "Input:      $INPUT_DIR"
echo "Output:     $OUTPUT_DIR"
echo "GPU:        $GPU"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python scripts/inference.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda
