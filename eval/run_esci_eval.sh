#!/bin/bash

# Evaluation script for Amazon ESCI dataset with Qwen3 Reranker

# Default values
MODEL="Qwen/Qwen3-Reranker-0.6B"
DATASET="thebajajra/amazon-esci-english-small"
SPLIT="test"
BATCH_SIZE=32

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --flash_attention)
            FLASH_ATTENTION="--flash_attention"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running evaluation with:"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET"
echo "  Split: $SPLIT"
echo "  Batch Size: $BATCH_SIZE"
echo ""

python evaluate_qwen_esci.py \
    --model "$MODEL" \
    --dataset_name "$DATASET" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    $FLASH_ATTENTION

