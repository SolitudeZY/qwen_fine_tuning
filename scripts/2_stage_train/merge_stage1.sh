#!/bin/bash
# Stage 1 LoRA 合并
# 用法: bash scripts/2_stage_train/merge_stage1.sh
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3-VL-2B-Instruct"
STAGE1_OUTPUT_DIR="$PROJECT_ROOT/outputs/stage1_grounding"
MERGED_MODEL_DIR="$PROJECT_ROOT/outputs/stage1_merged"

echo "============================================="
echo "  合并 Stage 1 LoRA 权重"
echo "  Base:   $MODEL_DIR"
echo "  Output: $MERGED_MODEL_DIR"
echo "============================================="

# 找最新 checkpoint
BEST_CKPT=$(ls -d "$STAGE1_OUTPUT_DIR"/v*/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "[Error] 找不到 checkpoint，请先运行 train_stage1.sh"
    exit 1
fi
echo "Checkpoint: $BEST_CKPT"

# swift export 要求输出目录不存在
if [ -d "$MERGED_MODEL_DIR" ]; then
    echo "[Error] 输出目录已存在: $MERGED_MODEL_DIR"
    echo "请手动删除后重试: rm -rf $MERGED_MODEL_DIR"
    exit 1
fi

swift export \
    --model_type qwen3_vl \
    --model "$MODEL_DIR" \
    --adapters "$BEST_CKPT" \
    --merge_lora true \
    --output_dir "$MERGED_MODEL_DIR" \
    2>&1 | tee "$MERGED_MODEL_DIR/merge.log"

echo ""
echo "合并完成！合并后模型: $MERGED_MODEL_DIR"
echo "下一步: bash scripts/2_stage_train/train_stage2.sh"
