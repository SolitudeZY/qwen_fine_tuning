#!/bin/bash
# 环水保：Qwen3.5-2B 单阶段微调（普通 LoRA + bf16）
# 用法: bash scripts/qwen3.5-water/03_train_stage1.sh
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen35-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data/环水保_labeled"
TRAIN_JSONL="$DATA_DIR/water_finetune_reviewed.jsonl"

MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3.5-2B"
OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen3.5-water/train"

# ===== LoRA 超参 =====
LORA_RANK=8
LORA_ALPHA=32
LR="1e-4"
EPOCHS=5
BATCH_SIZE=2
GRAD_ACCUM=2

export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0

echo "============================================="
echo "  环水保: Qwen3.5-2B 微调（LoRA）"
echo "  Model:   $MODEL_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "============================================="

if [ ! -f "$TRAIN_JSONL" ]; then
    echo "[Error] 找不到训练数据: $TRAIN_JSONL"
    echo "请先运行 02_review_ui.py 并在网页上导出为 JSONL 文件。"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "训练样本数: $(wc -l < "$TRAIN_JSONL")"

{
swift sft \
    --model "$MODEL_DIR" \
    --model_type qwen3_5 \
    --tuner_type lora \
    --dataset "$TRAIN_JSONL" \
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
    --max_length 2048 \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 3 \
    --eval_strategy steps \
    --eval_steps 50 \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end true \
    --logging_steps 5 \
    --load_from_cache_file true \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --attn_impl flash_attention_2 \
    --use_liger_kernel true
} 2>&1 | tee "$OUTPUT_DIR/train.log"

echo ""
echo "环水保单阶段训练完成！模型及最佳 checkpoint 保存在: $OUTPUT_DIR"
