#!/bin/bash
# Generate distillation data on 8 GPUs in parallel
# Usage: ./run_distillation_data.sh [num_chunks] [output_dir] [target_layers] [stride]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_CHUNKS="${1:-400}"
OUTPUT_DIR="${2:-/data/dflash_train}"
TARGET_LAYERS="${3:-24,30,31,32,34}"
STRIDE="${4:-1}"
NUM_GPUS=8
CHUNKS_PER_GPU=$((NUM_CHUNKS / NUM_GPUS))

echo "========================================"
echo "DFlash Distillation Data Generation"
echo "========================================"
echo "Total chunks: ${NUM_CHUNKS}"
echo "GPUs: ${NUM_GPUS}"
echo "Chunks per GPU: ${CHUNKS_PER_GPU}"
echo "Output: ${OUTPUT_DIR}"
echo "Target layers: [${TARGET_LAYERS}]"
echo "Stride: ${STRIDE}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}"

for rank in {0..7}; do
    START=$((rank * CHUNKS_PER_GPU))
    END=$(((rank + 1) * CHUNKS_PER_GPU))

    echo "Starting rank ${rank}: chunks ${START}-$((END - 1))..."

    CUDA_VISIBLE_DEVICES=${rank} python "${SCRIPT_DIR}/generate_distillation_data.py" \
        --rank ${rank} \
        --start-chunk ${START} \
        --end-chunk ${END} \
        --output-dir "${OUTPUT_DIR}" \
        --target-layers "${TARGET_LAYERS}" \
        --stride ${STRIDE} \
        --top-k-logits 128 \
        --full-vocab \
        2>&1 | tee "${OUTPUT_DIR}/rank${rank}.log" &
done

wait

echo "All processes complete!"
echo "Output: ${OUTPUT_DIR}"
