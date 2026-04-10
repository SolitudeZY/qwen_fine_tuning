#!/bin/bash
# ============================================================
# Stage 1: Qwen3-VL Grounding 预训练
#
# 目的：用 LabelMe 真实标注教模型视觉空间对齐
#       （看到哪里 → 输出那里的坐标）
#
# 完成后自动合并 LoRA 权重，供 Stage 2 使用
#
# 用法:
#   bash scripts/train_stage1_grounding.sh
# ============================================================
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"
GROUNDING_JSONL="$DATA_DIR/grounding_stage1.jsonl"

MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3-VL-2B-Instruct"
STAGE1_OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen3vl_2b_stage1_grounding"
MERGED_MODEL_DIR="$PROJECT_ROOT/outputs/qwen3vl_2b_stage1_merged"

# Stage 1 超参数（grounding 任务用更低 LR 和更小 rank）
LORA_RANK=16
LORA_ALPHA=32
LR="2e-5"
EPOCHS=5
BATCH_SIZE=1
GRAD_ACCUM=4
MAX_PIXELS=1003520

echo "============================================="
echo "  Stage 1: Qwen3-VL Grounding 预训练"
echo "  Model:   $MODEL_DIR"
echo "  Dataset: $GROUNDING_JSONL"
echo "  Output:  $STAGE1_OUTPUT_DIR"
echo "============================================="

# 检查数据文件
if [ ! -f "$GROUNDING_JSONL" ]; then
    echo "[Error] 找不到 grounding 训练数据: $GROUNDING_JSONL"
    echo "请先运行: python scripts/convert_labelme_to_grounding.py"
    exit 1
fi

SAMPLE_COUNT=$(wc -l < "$GROUNDING_JSONL")
echo "训练样本数: $SAMPLE_COUNT"

mkdir -p "$STAGE1_OUTPUT_DIR"

# ---------- Stage 1 QLoRA 训练 ----------
echo ""
echo "Step 1: 开始 Grounding QLoRA 训练..."
swift sft \
    --model "$MODEL_DIR" \
    --train_type lora \
    --dataset "$GROUNDING_JSONL" \
    --quant_method bnb \
    --quant_bits 4 \
    --bnb_4bit_quant_type nf4 \
    --output_dir "$STAGE1_OUTPUT_DIR" \
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
    2>&1 | tee "$STAGE1_OUTPUT_DIR/train.log"

echo ""
echo "Step 1 完成！"

# ---------- 合并 LoRA 权重 ----------
# 找到最新的 checkpoint
BEST_CKPT=$(ls -d "$STAGE1_OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "[Error] 找不到训练 checkpoint，合并失败"
    exit 1
fi

echo ""
echo "Step 2: 合并 LoRA 权重到基础模型..."
echo "  Checkpoint: $BEST_CKPT"
echo "  输出目录:   $MERGED_MODEL_DIR"

mkdir -p "$MERGED_MODEL_DIR"

swift export \
    --adapters "$BEST_CKPT" \
    --merge_lora true \
    --output_dir "$MERGED_MODEL_DIR" \
    2>&1 | tee "$MERGED_MODEL_DIR/merge.log"

echo ""
echo "============================================="
echo "  Stage 1 完成！"
echo "  合并后模型: $MERGED_MODEL_DIR"
echo "  下一步: bash scripts/train_stage2_json.sh"
echo "============================================="
