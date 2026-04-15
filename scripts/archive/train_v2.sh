#!/bin/bash
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen2.5-VL-7B-Instruct"
DATA_DIR="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/outputs/edge_protection_v2_clean"

python "$PROJECT_ROOT/scripts/prepare_train_v2_data.py" \
    --input-dir "$DATA_DIR" \
    --train-name train_balanced.jsonl \
    --val-name val_balanced.jsonl \
    --train-output train_balanced_clean.jsonl \
    --val-output val_balanced_clean.jsonl

mkdir -p "$OUTPUT_DIR"

swift sft \
    --model "$MODEL_DIR" \
    --train_type lora \
    --dataset "$DATA_DIR/train_balanced_clean.jsonl" \
    --val_dataset "$DATA_DIR/val_balanced_clean.jsonl" \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 2048 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --gradient_checkpointing true \
    --max_pixels 602112 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Training completed. Output saved to: $OUTPUT_DIR"
