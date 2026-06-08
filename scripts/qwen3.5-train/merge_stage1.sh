#!/bin/bash
# Stage 1 LoRA 合并脚本 — Qwen3.5-2B
# 将 Stage 1 最佳 checkpoint 的 LoRA 权重合并进基础模型
# 用法: bash scripts/qwen3.5-train/merge_stage1.sh [checkpoint_path]
# 例如：bash scripts/qwen3.5-train/merge_stage1.sh /home/fs-ai/llama-qwen/outputs/stage1_grounding/qwen3.5/v13-20260526-111426/checkpoint-150
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen35-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3.5-2B"
STAGE1_OUTPUT="$PROJECT_ROOT/outputs/stage1_grounding/qwen3.5"
MERGED_DIR="$PROJECT_ROOT/outputs/stage1_merged/qwen3.5"

# ===== 自动寻找最新 run 的 best checkpoint =====
# 优先使用命令行参数指定的 checkpoint
if [ $# -ge 1 ]; then
    CHECKPOINT_PATH="$1"
    echo "[Info] 使用指定 checkpoint: $CHECKPOINT_PATH"
else
    # 找最新 v* 目录下的 trainer_state.json，从中读取 best_model_checkpoint
    LATEST_RUN=$(ls -td "$STAGE1_OUTPUT"/v*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_RUN" ]; then
        echo "[Error] 找不到 Stage 1 训练输出目录: $STAGE1_OUTPUT/v*/"
        echo "请先运行: bash scripts/qwen3.5-train/train_stage1.sh"
        exit 1
    fi

    TRAINER_STATE="$LATEST_RUN/trainer_state.json"
    if [ -f "$TRAINER_STATE" ]; then
        CHECKPOINT_PATH=$(python3 -c "
import json, sys
with open('$TRAINER_STATE') as f:
    state = json.load(f)
best = state.get('best_model_checkpoint', '')
if best:
    print(best)
else:
    # fallback: 取最后一个 checkpoint
    print(state.get('log_history', [{}])[-1].get('step', ''))
" 2>/dev/null)
    fi

    # 如果解析失败，fallback 到最新 checkpoint 目录
    if [ -z "$CHECKPOINT_PATH" ] || [ ! -d "$CHECKPOINT_PATH" ]; then
        CHECKPOINT_PATH=$(ls -td "$LATEST_RUN"checkpoint-*/ 2>/dev/null | head -1)
        CHECKPOINT_PATH="${CHECKPOINT_PATH%/}"  # 去掉末尾斜杠
    fi

    if [ -z "$CHECKPOINT_PATH" ] || [ ! -d "$CHECKPOINT_PATH" ]; then
        echo "[Error] 找不到有效 checkpoint，请手动指定:"
        echo "  bash scripts/qwen3.5-train/merge_stage1.sh <checkpoint_path>"
        exit 1
    fi

    echo "[Info] 自动选取 checkpoint: $CHECKPOINT_PATH"
fi

# ===== 输出目录处理 =====
# swift export 要求输出目录不能已存在（否则报错），自动加时间戳
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
MERGED_OUT="$MERGED_DIR/merged-$TIMESTAMP"

echo "============================================="
echo "  Stage 1 LoRA 合并"
echo "  Base:       $MODEL_DIR"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  Output:     $MERGED_OUT"
echo "============================================="

mkdir -p "$MERGED_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF='expandable_segments:True'

swift export \
    --model "$MODEL_DIR" \
    --model_type qwen3_5 \
    --adapters "$CHECKPOINT_PATH" \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --output_dir "$MERGED_OUT"

echo ""
echo "合并完成！合并后模型在: $MERGED_OUT"
echo ""
echo "下一步运行 Stage 2 训练:"
echo "  bash scripts/qwen3.5-train/train_stage2.sh $MERGED_OUT"
