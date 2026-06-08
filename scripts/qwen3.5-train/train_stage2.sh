#!/bin/bash
# Stage 2: Qwen3.5-2B JSON 结构化输出微调（基于 Stage 1 合并模型）
# 用法:
#   bash scripts/qwen3.5-train/train_stage2.sh                    # 自动选最新合并模型
#   bash scripts/qwen3.5-train/train_stage2.sh <merged_model_dir> # 手动指定基模型
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen35-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"
STAGE2_JSONL="$DATA_DIR/stage2_json.jsonl"

STAGE1_MERGED_DIR="$PROJECT_ROOT/outputs/stage1_merged/qwen3.5"
OUTPUT_DIR="$PROJECT_ROOT/outputs/stage2_json/qwen3.5"

# ===== 选择基模型 =====
if [ $# -ge 1 ]; then
    BASE_MODEL="$1"
else
    # 自动选取最新合并目录
    BASE_MODEL=$(ls -td "$STAGE1_MERGED_DIR"/merged-*/ 2>/dev/null | head -1)
    BASE_MODEL="${BASE_MODEL%/}"
fi

if [ -z "$BASE_MODEL" ] || [ ! -d "$BASE_MODEL" ]; then
    echo "[Error] 找不到 Stage 1 合并模型: $STAGE1_MERGED_DIR/merged-*/"
    echo "请先运行: bash scripts/qwen3.5-train/merge_stage1.sh"
    exit 1
fi

if [ ! -f "$STAGE2_JSONL" ]; then
    echo "[Error] 找不到 Stage 2 数据: $STAGE2_JSONL"
    exit 1
fi

# ===== LoRA 超参 =====
# Stage 2 在已对齐的基模型上做格式微调，用更小学习率
LORA_RANK=8
LORA_ALPHA=32
LR="5e-5"
EPOCHS=4
BATCH_SIZE=2
GRAD_ACCUM=2

# ===== 多模态 token 上限 =====
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0

echo "============================================="
echo "  Stage 2: JSON 结构化输出微调"
echo "  Base:    $BASE_MODEL"
echo "  Data:    $STAGE2_JSONL ($(wc -l < "$STAGE2_JSONL") samples)"
echo "  Output:  $OUTPUT_DIR"
echo "============================================="

mkdir -p "$OUTPUT_DIR"

# 关键说明:
# 1. Stage 2 输入更长（system prompt + JSON），max_length 提到 4096
# 2. 学习率减半（5e-5），避免破坏 Stage 1 学到的定位能力
# 3. 仍关闭 thinking：本任务直接输出 JSON，不需要思考链
# 4. 数据量略大（814 vs 342），epoch 数适当减少

{
swift sft \
    --model "$BASE_MODEL" \
    --model_type qwen3_5 \
    --tuner_type lora \
    --dataset "$STAGE2_JSONL" \
    --split_dataset_ratio 0.05 \
    --torch_dtype bfloat16 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --freeze_vit false \
    --group_by_length true \
    --gradient_checkpointing true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --max_length 4096 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --eval_steps 100 \
    --logging_steps 5 \
    --load_from_cache_file true \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --attn_impl flash_attention_2 \
    --use_liger_kernel true
} 2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "Stage 2 训练完成！checkpoint 在: $OUTPUT_DIR"
