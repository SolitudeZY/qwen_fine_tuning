#!/bin/bash
# Stage 1: Qwen3.5-2B Grounding 预训练（普通 LoRA + bf16）
# 参考: Qwen3.5 最佳实践 (swift 4.3.0.dev0)
# 用法: bash scripts/qwen3.5-train/train_stage1.sh
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate qwen35-ft

PROJECT_ROOT="/home/fs-ai/llama-qwen"
DATA_DIR="$PROJECT_ROOT/data"
GROUNDING_JSONL="$DATA_DIR/stage1_grounding.jsonl"

MODEL_DIR="$PROJECT_ROOT/models/Qwen/Qwen3.5-2B"
OUTPUT_DIR="$PROJECT_ROOT/outputs/stage1_grounding/qwen3.5"

# ===== LoRA 超参（按官方文档推荐起点）=====
LORA_RANK=8
LORA_ALPHA=32
LR="1e-4"
EPOCHS=4
BATCH_SIZE=2
GRAD_ACCUM=2

# ===== 多模态 token 上限（Qwen3-VL 同名环境变量）=====
# 训练图片为 1024×1024，patch=16, spatial_merge=2 → 32×32=1024 tokens
export IMAGE_MAX_TOKEN_NUM=1024
export VIDEO_MAX_TOKEN_NUM=128
export FPS_MAX_FRAMES=12
export PYTORCH_ALLOC_CONF='expandable_segments:True'
export CUDA_VISIBLE_DEVICES=0

echo "============================================="
echo "  Stage 1: Qwen3.5-2B Grounding 预训练（LoRA）"
echo "  Model:   $MODEL_DIR"
echo "  Output:  $OUTPUT_DIR"
echo "============================================="

if [ ! -f "$GROUNDING_JSONL" ]; then
    echo "[Error] 找不到训练数据: $GROUNDING_JSONL"
    echo "请先运行: python scripts/2_stage_train/api_annotate_stage12.py"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "训练样本数: $(wc -l < "$GROUNDING_JSONL")"

# ===== 关键说明 =====
# 1. Qwen3.5 的 linear_attention/Mamba 层不适合 4bit 量化，故不使用 QLoRA，
#    改为普通 LoRA + bf16（2B 模型在单卡 24GB 显存上完全够用）。
# 2. transformers 后端的 GatedDeltaNet 不支持 packing/padding_free，
#    使用 --group_by_length true 平衡 DP 负载、减少 padding。
# 3. --target_modules all-linear：让 LoRA 自动注入所有 nn.Linear（含 ViT、
#    full_attention、MLP），跳过 conv1d 等非 Linear 层。
# 4. --add_non_thinking_prefix + --loss_scale ignore_empty_think：
#    Qwen3.5 是混合思考模型，本任务为定位任务不需要思考链，关闭 thinking。

{
swift sft \
    --model "$MODEL_DIR" \
    --model_type qwen3_5 \
    --tuner_type lora \
    --dataset "$GROUNDING_JSONL" \
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
echo "Stage 1 训练完成！checkpoint 在: $OUTPUT_DIR"