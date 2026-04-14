#!/bin/bash
# Stage 2: JSON 精调（从 Stage 1 合并模型开始）
# 用法: bash scripts/2_stage_train/train_stage2.sh
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"
DATASET_JSONL="$DATA_DIR/stage2_json.jsonl"          # api_annotate_stage12.py 生成

MERGED_MODEL_DIR="$PROJECT_ROOT/outputs/stage1_merged"
OUTPUT_DIR="$PROJECT_ROOT/outputs/stage2_json"

LORA_RANK=32
LORA_ALPHA=64
LR="5e-5"
EPOCHS=15
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_PIXELS=1003520

echo "============================================="
echo "  Stage 2: JSON 精调"
echo "  Base:    $MERGED_MODEL_DIR"
echo "  Dataset: $DATASET_JSONL"
echo "  Output:  $OUTPUT_DIR"
echo "============================================="

if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "[Error] 找不到 Stage 1 合并模型: $MERGED_MODEL_DIR"
    echo "请先运行: bash scripts/2_stage_train/merge_stage1.sh"
    exit 1
fi

if [ ! -f "$DATASET_JSONL" ]; then
    echo "[Error] 找不到训练数据: $DATASET_JSONL"
    echo "请先运行: python scripts/2_stage_train/api_annotate_stage12.py"
    exit 1
fi

echo "训练样本数: $(wc -l < "$DATASET_JSONL")"
mkdir -p "$OUTPUT_DIR"

swift sft \
    --model "$MERGED_MODEL_DIR" \
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
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --gradient_checkpointing true \
    --max_pixels "$MAX_PIXELS" \
    2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "============================================="
echo "  Stage 2 完成！"
echo "  验证: python scripts/test_fence_violations.py"
echo "  推理: python scripts/chat.py --lora_path $OUTPUT_DIR/vN/checkpoint-X --visualize"
echo "============================================="
