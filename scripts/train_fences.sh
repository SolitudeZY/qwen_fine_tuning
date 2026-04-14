#!/bin/bash
# ============================================================
# Qwen3-VL QLoRA 围栏精细化微调脚本
#
# 用法:
#   bash scripts/train_fences.sh
# ============================================================
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"
DATASET_JSONL="$DATA_DIR/finetune_fences_minimal_cot.jsonl"

MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen3vl_2b_fences_minimal_cot_lora"

LORA_RANK=16
LORA_ALPHA=32
LR="5e-5"
EPOCHS=15
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_PIXELS=4000000 # 4000000

echo "============================================="
echo "  Qwen3-VL 2B 围栏精细化 QLoRA 微调"
echo "  Model: $MODEL_DIR"
echo "  Dataset: $DATASET_JSONL"
echo "  Output: $OUTPUT_DIR"
echo "============================================="

if [ ! -f "$DATASET_JSONL" ]; then
    echo "[Error] 找不到训练数据: $DATASET_JSONL"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ---------- QLoRA 训练 ----------
echo "Step 1: 开始 QLoRA 训练..."
swift sft \
    --model "$MODEL_DIR" \
    --train_type lora \
    --dataset "$DATASET_JSONL" \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 4096 \
    --truncation_strategy left \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --gradient_checkpointing true \
    --max_pixels "$MAX_PIXELS" \
    --enable_thinking false 
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo "============================================="
echo "  微调完成！"
echo "  LoRA 权重已保存在: $OUTPUT_DIR"
echo "============================================="
