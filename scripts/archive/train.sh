#!/bin/bash
# ============================================================
# Qwen2.5-VL-7B QLoRA 微调训练脚本
# 使用 ms-swift 4.x 框架，适配 RTX 5060 Ti (16GB VRAM)
# ============================================================

# 激活 conda 环境
eval "$(conda shell.bash hook)"
conda activate qwen-ft

# 配置路径
MODEL_DIR="/home/fs-ai/llama-qwen/models/Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_DATA="/home/fs-ai/llama-qwen/data/train_balanced.jsonl"
VAL_DATA="/home/fs-ai/llama-qwen/data/val_balanced.jsonl"
OUTPUT_DIR="/home/fs-ai/llama-qwen/outputs/edge_protection_v2"

mkdir -p "$OUTPUT_DIR"

# ms-swift 4.x QLoRA 微调
swift sft \
    --model "$MODEL_DIR" \
    --train_type lora \
    --dataset "$TRAIN_DATA" \
    --val_dataset "$VAL_DATA" \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 2048 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --gradient_checkpointing true \
    --max_pixels 602112 \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "Training completed. Output saved to: $OUTPUT_DIR"
