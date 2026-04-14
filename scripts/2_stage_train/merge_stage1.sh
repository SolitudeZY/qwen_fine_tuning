#!/bin/bash
# Stage 1 LoRA 合并
# 用法: bash scripts/2_stage_train/merge_stage1.sh
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3-VL-2B-Instruct"
STAGE1_OUTPUT_DIR="$PROJECT_ROOT/outputs/stage1_grounding"

echo "============================================="
echo "  合并 Stage 1 LoRA 权重"
echo "  Base: $MODEL_DIR"
echo "============================================="

# 找最新 checkpoint（排除已合并的 -merged 目录）
BEST_CKPT=$(ls -d "$STAGE1_OUTPUT_DIR"/v*/checkpoint-* 2>/dev/null | grep -v '\-merged' | sort -V | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "[Error] 找不到 checkpoint，请先运行 train_stage1.sh"
    exit 1
fi
echo "Checkpoint: $BEST_CKPT"

# swift export 会在 checkpoint 同级目录创建 checkpoint-xxx-merged
MERGED_MODEL_DIR="${BEST_CKPT}-merged"
if [ -d "$MERGED_MODEL_DIR" ]; then
    echo "[Info] 合并目录已存在，跳过合并: $MERGED_MODEL_DIR"
else
    swift export \
        --model_type qwen3_vl \
        --model "$MODEL_DIR" \
        --adapters "$BEST_CKPT" \
        --merge_lora true \
        2>&1
    echo ""
fi

echo "合并后模型: $MERGED_MODEL_DIR"
echo "下一步: bash scripts/2_stage_train/train_stage2.sh"
echo "请将 train_stage2.sh 中 MERGED_MODEL_DIR 设为:"
echo "  $MERGED_MODEL_DIR"
