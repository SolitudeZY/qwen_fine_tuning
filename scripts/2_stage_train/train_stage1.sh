#!/bin/bash
# Stage 1: Qwen3-VL Grounding 预训练
# 用法: bash scripts/2_stage_train/train_stage1.sh
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"
GROUNDING_JSONL="$DATA_DIR/stage1_grounding.jsonl"   # api_annotate_stage12.py 生成

MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="$PROJECT_ROOT/outputs/stage1_grounding"

LORA_RANK=16
LORA_ALPHA=32
LR="2e-5"
EPOCHS=5
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_PIXELS=1003520

echo "============================================="
echo "  Stage 1: Grounding 预训练"
echo "  Model:   $MODEL_DIR"
echo "  Dataset: $GROUNDING_JSONL"
echo "  Output:  $OUTPUT_DIR"
echo "============================================="

if [ ! -f "$GROUNDING_JSONL" ]; then
    echo "[Error] 找不到训练数据: $GROUNDING_JSONL"
    echo "请先运行: python scripts/2_stage_train/api_annotate_stage12.py"
    exit 1
fi

echo "训练样本数: $(wc -l < "$GROUNDING_JSONL")"
mkdir -p "$OUTPUT_DIR"

swift sft \
    --model "$MODEL_DIR" \
    --train_type lora \
    --dataset "$GROUNDING_JSONL" \
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
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 3 \
    --logging_steps 5 \
    --max_length 2048 \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --gradient_checkpointing true \
    --max_pixels "$MAX_PIXELS" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "Stage 1 训练完成！下一步: bash scripts/2_stage_train/merge_stage1.sh"
