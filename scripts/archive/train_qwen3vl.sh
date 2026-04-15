#!/bin/bash
# ============================================================
# Qwen3-VL QLoRA 微调脚本
#
# 用法:
#   bash scripts/train_qwen3vl.sh              # 默认使用 2B 模型
#   bash scripts/train_qwen3vl.sh 4b           # 使用 4B 模型
#   bash scripts/train_qwen3vl.sh 2b           # 使用 2B 模型
# ============================================================
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"

# ---------- 选择模型大小 ----------
MODEL_SIZE="${1:-2b}"

case "$MODEL_SIZE" in
    2b|2B)
        MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3-VL-2B-Instruct"
        OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen3vl_2b_lora"
        LORA_RANK=16
        LORA_ALPHA=32
        LR="1e-4"
        EPOCHS=5
        BATCH_SIZE=2
        GRAD_ACCUM=2
        MAX_PIXELS=602112
        ;;
    4b|4B)
        MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3-VL-4B-Instruct"
        OUTPUT_DIR="$PROJECT_ROOT/outputs/qwen3vl_4b_lora"
        LORA_RANK=32
        LORA_ALPHA=64
        LR="5e-5"
        EPOCHS=3
        BATCH_SIZE=1
        GRAD_ACCUM=4
        MAX_PIXELS=602112
        ;;
    *)
        echo "Usage: $0 [2b|4b]"
        exit 1
        ;;
esac

echo "============================================="
echo "  Qwen3-VL ${MODEL_SIZE^^} QLoRA Fine-tuning"
echo "  Model: $MODEL_DIR"
echo "  Output: $OUTPUT_DIR"
echo "============================================="

# ---------- 数据清洗 ----------
echo "Step 1: 清洗训练数据..."
python "$PROJECT_ROOT/scripts/prepare_train_v2_data.py" \
    --input-dir "$DATA_DIR" \
    --train-name train_balanced.jsonl \
    --val-name val_balanced.jsonl \
    --train-output train_balanced_clean.jsonl \
    --val-output val_balanced_clean.jsonl

mkdir -p "$OUTPUT_DIR"

# ---------- QLoRA 训练 ----------
echo "Step 2: 开始 QLoRA 训练..."
swift sft \
    --model "$MODEL_DIR" \
    --train_type lora \
    --dataset "$DATA_DIR/train_balanced_clean.jsonl" \
    --val_dataset "$DATA_DIR/val_balanced_clean.jsonl" \
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
    --eval_strategy steps \
    --eval_steps 20 \
    --save_strategy steps \
    --save_steps 20 \
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
echo "============================================="
echo "  训练完成！"
echo "  输出目录: $OUTPUT_DIR"
echo ""
echo "  测试推理:"
echo "    python scripts/chat.py \\"
echo "      --model_path $MODEL_DIR \\"
echo "      --lora_path $OUTPUT_DIR/checkpoint-best \\"
echo "      --image your_image.jpg"
echo "============================================="
