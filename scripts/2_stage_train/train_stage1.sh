#!/bin/bash
# Stage 1: Qwen3-VL Grounding 预训练
# 用法: bash scripts/2_stage_train/train_stage1.sh
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"
GROUNDING_JSONL="$DATA_DIR/stage1_grounding.jsonl"
# TILED_JSONL="$DATA_DIR/stage1_tiled.jsonl"
# COMBINED_JSONL="$DATA_DIR/stage1_combined.jsonl"

# MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3-VL-2B-Instruct"
MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3.5-2B"
OUTPUT_DIR="$PROJECT_ROOT/outputs/stage1_grounding/qwen3.5"

LORA_RANK=32
LORA_ALPHA=64
LR="3e-5"
EPOCHS=6
BATCH_SIZE=2                  # 【修改】1 → 2，充分利用 1024×1024 比原来 max_pixels 小的显存余量
GRAD_ACCUM=2                  # 【修改】4 → 2，配合 batch_size 翻倍，有效 batch size 保持 4
MAX_PIXELS=1048576            # 【修改】1505280 → 1048576（即 1024×1024），精确匹配训练图片尺寸

echo "============================================="
echo "  Stage 1: Grounding 预训练"
echo "  Model:   $MODEL_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "============================================="

if [ ! -f "$GROUNDING_JSONL" ]; then
    echo "[Error] 找不到训练数据: $GROUNDING_JSONL"
    echo "请先运行: python scripts/2_stage_train/api_annotate_stage12.py"
    exit 1
fi

# # 合并原始样本 + 分块样本——————2026.5.25 这段删去，图片在预处理时已经被分块成1080*1080大小，不需要再使用分块
# if [ -f "$TILED_JSONL" ]; then
#     cat "$GROUNDING_JSONL" "$TILED_JSONL" > "$COMBINED_JSONL"
#     TRAIN_DATA="$COMBINED_JSONL"
#     echo "训练样本数: $(wc -l < "$COMBINED_JSONL")（原始 $(wc -l < "$GROUNDING_JSONL") + 分块 $(wc -l < "$TILED_JSONL")）"
# else
#     TRAIN_DATA="$GROUNDING_JSONL"
#     echo "[提示] 未找到 stage1_tiled.jsonl，仅使用原始样本"
#     echo "建议先运行: python scripts/2_stage_train/gen_tiled_stage1.py"
#     echo "训练样本数: $(wc -l < "$GROUNDING_JSONL")"
# fi

TRAIN_DATA="$GROUNDING_JSONL"
mkdir -p "$OUTPUT_DIR"

swift sft \
    --model "$MODEL_DIR" \
    --train_type lora \
    --dataset "$TRAIN_DATA" \
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
