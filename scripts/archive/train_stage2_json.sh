#!/bin/bash
# ============================================================
# Stage 2: JSON 精调（从 Stage 1 合并模型开始）
#
# 目的：在 Stage 1 空间对齐能力基础上，教模型用 JSON 格式
#       表达定位结果，violation_boxes 使用 LabelMe 真实坐标
#
# 用法:
#   bash scripts/train_stage2_json.sh
# ============================================================
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"
DATASET_JSONL="$DATA_DIR/finetune_stage2_gt.jsonl"

# 从 Stage 1 合并后的模型开始（而非原始基础模型）
MERGED_MODEL_DIR="$PROJECT_ROOT/outputs/qwen3vl_2b_stage1_grounding/v4-20260413-145230/checkpoint-90-merged"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen3vl_2b_stage2_json"

# Stage 2 超参数（与 train_fences.sh 一致，rank 提升到 32）
LORA_RANK=16
LORA_ALPHA=32
LR="5e-5"
EPOCHS=15
BATCH_SIZE=1
GRAD_ACCUM=8
MAX_PIXELS=1003520

echo "============================================="
echo "  Stage 2: JSON 精调（真实坐标）"
echo "  Base:    $MERGED_MODEL_DIR"
echo "  Dataset: $DATASET_JSONL"
echo "  Output:  $OUTPUT_DIR"
echo "============================================="

# 检查 Stage 1 合并模型
if [ ! -d "$MERGED_MODEL_DIR" ]; then
    echo "[Error] 找不到 Stage 1 合并模型: $MERGED_MODEL_DIR"
    echo "请先运行: bash scripts/train_stage1_grounding.sh"
    exit 1
fi

# 检查数据文件
if [ ! -f "$DATASET_JSONL" ]; then
    echo "[Error] 找不到 Stage 2 训练数据: $DATASET_JSONL"
    echo "请先运行: python scripts/prepare_stage2_json.py"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "$DATASET_JSONL")
echo "训练样本数: $SAMPLE_COUNT"

mkdir -p "$OUTPUT_DIR"

# ---------- Stage 2 QLoRA 训练 ----------
echo ""
echo "Step 1: 开始 JSON 精调..."
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
echo "  LoRA 权重: $OUTPUT_DIR"
echo ""
echo "  验证方式："
echo "  1. python scripts/test_fence_violations.py"
echo "     （对比 recall rate 与基线）"
echo "  2. python scripts/chat.py --lora_path $OUTPUT_DIR/checkpoint-xxx --visualize"
echo "     （目视检查坐标是否准确落在违规区域）"
echo "============================================="
