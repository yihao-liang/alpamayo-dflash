#!/bin/bash
# Train DFlash from scratch for Alpamayo VLM
#
# This script creates a NEW DFlash model with Alpamayo's vocabulary (155,698 tokens)
# and trains it using pre-computed distillation data.
#
# Key features:
# - Creates DFlash from scratch (not loading pre-trained weights)
# - Uses Alpamayo's vocab_size (155,697 + 1 MASK token)
# - Loads embed_tokens and lm_head from Alpamayo
# - Supports trainable or frozen embeddings

set -e

# ============== Configuration ==============
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS_PER_NODE=8

# Model paths
TARGET_MODEL="/models/Alpamayo-R1-10B"
DATA_DIR="/data/dflash_train"
OUTPUT_DIR="/exp"

# Training hyperparameters
NUM_EPOCHS=5
BATCH_SIZE=64           # Per GPU
LEARNING_RATE=3e-4      # Will be scaled by world size
BLOCK_SIZE=8            # Match your distillation data
NUM_DRAFT_LAYERS=5      # Number of draft decoder layers

# Embedding training (set to true to train embed_tokens and lm_head)
TRAIN_EMBEDDINGS=false
EMBED_LR_SCALE=0.1      # LR multiplier for embeddings (if trainable)

# Loss configuration
PREFIX_WEIGHT_GAMMA=3  # Geometric decay for prefix-weighted CE

# ============== Run Training ==============
echo "=============================================="
echo "Training DFlash from Scratch for Alpamayo"
echo "=============================================="
echo "Target model: $TARGET_MODEL"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Train embeddings: $TRAIN_EMBEDDINGS"
echo ""

TRAIN_ARGS=(
    --target-model "$TARGET_MODEL"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --num-epochs $NUM_EPOCHS
    --batch-size $BATCH_SIZE
    --learning-rate $LEARNING_RATE
    --block-size $BLOCK_SIZE
    --num-draft-layers $NUM_DRAFT_LAYERS
    --prefix-weight-gamma $PREFIX_WEIGHT_GAMMA
    --save-every 1
    --num-workers 4
)

if [ "$TRAIN_EMBEDDINGS" = true ]; then
    TRAIN_ARGS+=(--train-embeddings --embed-lr-scale $EMBED_LR_SCALE)
fi

torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    train_dflash.py \
    "${TRAIN_ARGS[@]}"

echo ""
echo "Training complete!"
