#!/bin/bash

# RexBERT Distributional Reranker Training Script
# This script trains a RexBERT model for reranking with distributional outputs

set -e  # Exit on error

# Model and dataset configuration
MODEL_NAME="thebajajra/RexBERT-mini"
DATASET_NAME="thebajajra/Amazebay-reranker-training-data"
OUTPUT_DIR="./outputs/rexbert-mini-reranker-$(date +%Y%m%d_%H%M%S)"

# Model architecture
MAX_LENGTH=2048
NUM_BINS=11
DROPOUT=0.0
POOLING_STRATEGY="mean"  # Options: "mean" or "cls"

# Dynamic sigma parameters (for uncertainty near transition points)
TRANSITIONS="0.2,0.5,0.8"
SIGMA_MIN=0.04
SIGMA_MAX=0.12
SIGMA_DELTA=0.08

# Loss configuration
LAMBDA_MEAN=0.0  # Weight for mean MSE auxiliary loss

# Learning rates
HEAD_LR=5e-4       # Higher LR for classification head
BACKBONE_LR=5e-4   # Lower LR for backbone fine-tuning

# Backbone unfreezing
UNFREEZE_BACKBONE_AFTER=0.1  # Unfreeze backbone after this many epochs (0.1 = 10% of first epoch)

# Training configuration
NUM_EPOCHS=5
BATCH_SIZE=32
EVAL_BATCH_SIZE=512
GRAD_ACCUM_STEPS=1
WEIGHT_DECAY=0.01
WARMUP_RATIO=0.05
LR_SCHEDULER="cosine"

# Logging and checkpointing
LOGGING_STEPS=100
EVAL_STEPS=19500
SAVE_STEPS=19500
SAVE_TOTAL_LIMIT=-1
SEED=42

# Mixed precision training (set one of these to true)
USE_BF16=true  # Use bfloat16 (recommended for A100/H100)
USE_FP16=false  # Use float16 (for V100 and older GPUs)

# Memory optimization
USE_GRADIENT_CHECKPOINTING=true  # Enable if running out of memory

# Weights & Biases logging
USE_WANDB=true               # Enable W&B logging
WANDB_PROJECT="RexReranker-rexbert"  # W&B project name
WANDB_RUN_NAME="rexbert-mini-reranker"                 # W&B run name (empty = auto-generated)

# Construct the command
CMD="python train_rexbert.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name $DATASET_NAME \
  --output_dir $OUTPUT_DIR \
  --max_length $MAX_LENGTH \
  --num_bins $NUM_BINS \
  --dropout $DROPOUT \
  --pooling_strategy $POOLING_STRATEGY \
  --transitions $TRANSITIONS \
  --sigma_min $SIGMA_MIN \
  --sigma_max $SIGMA_MAX \
  --sigma_delta $SIGMA_DELTA \
  --lambda_mean $LAMBDA_MEAN \
  --head_lr $HEAD_LR \
  --backbone_lr $BACKBONE_LR \
  --unfreeze_backbone_after $UNFREEZE_BACKBONE_AFTER \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
  --weight_decay $WEIGHT_DECAY \
  --warmup_ratio $WARMUP_RATIO \
  --lr_scheduler_type $LR_SCHEDULER \
  --logging_steps $LOGGING_STEPS \
  --eval_steps $EVAL_STEPS \
  --save_steps $SAVE_STEPS \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --seed $SEED"

# Add optional flags
if [ "$USE_BF16" = true ]; then
  CMD="$CMD --bf16"
fi

if [ "$USE_FP16" = true ]; then
  CMD="$CMD --fp16"
fi

if [ "$USE_GRADIENT_CHECKPOINTING" = true ]; then
  CMD="$CMD --gradient_checkpointing"
fi

# Add W&B flags
if [ "$USE_WANDB" = true ]; then
  CMD="$CMD --use_wandb"
  CMD="$CMD --wandb_project $WANDB_PROJECT"
  
  if [ -n "$WANDB_RUN_NAME" ]; then
    CMD="$CMD --wandb_run_name $WANDB_RUN_NAME"
  fi
  
  if [ -n "$WANDB_ENTITY" ]; then
    CMD="$CMD --wandb_entity $WANDB_ENTITY"
  fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Setup W&B API key if enabled and key file exists
if [ "$USE_WANDB" = true ]; then
  WANDB_KEY_FILE="./wandb_key"
  if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
    echo "Loaded W&B API key from $WANDB_KEY_FILE"
  else
    echo "Warning: W&B enabled but $WANDB_KEY_FILE not found. Will prompt for login."
  fi
fi

# Print configuration
echo "================================"
echo "RexBERT Reranker Training"
echo "================================"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS)))"
echo "Head LR: $HEAD_LR"
echo "Backbone LR: $BACKBONE_LR"
if [ "$USE_WANDB" = true ]; then
  echo "W&B: enabled (project: $WANDB_PROJECT)"
else
  echo "W&B: disabled"
fi
echo "================================"
echo ""

# Run training
echo "Starting training..."
eval $CMD

echo ""
echo "================================"
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "================================"
