#!/bin/bash
# 一键启动大模型 MQTT 推理服务 (环水保 + 临边防护)

set -euo pipefail

PROJECT_ROOT_DEFAULT="/home/fs-ai/jiahe/Qwen-3.5"
PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"
CONDA_ENV="${CONDA_ENV:-qwen35}"
LOG_DIR="$PROJECT_ROOT/logs"
PID_DIR="$LOG_DIR/pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

WATER_SCRIPT="$PROJECT_ROOT/scripts/qwen3.5-water/04_mqtt_inference.py"
EDGE_SCRIPT="$PROJECT_ROOT/scripts/qwen3.5-edge/video_infer_qwen35.py"

WATER_MODEL="${WATER_MODEL:-$PROJECT_ROOT/models/Qwen/Qwen3.5-2B}"
WATER_LORA="${WATER_LORA:-$PROJECT_ROOT/outputs/qwen3.5-water/train/v1-20260616-085800/checkpoint-745}"

EDGE_MODEL="${EDGE_MODEL:-$PROJECT_ROOT/outputs/qwen3.5-edge/merged-20260609-164452}"
EDGE_LORA="${EDGE_LORA:-$PROJECT_ROOT/outputs/qwen3.5-edge/json-20260609-164542/checkpoint-500}"

CUDA_WHEELS_DIR="${CUDA_WHEELS_DIR:-$PROJECT_ROOT/wheels}"
CONDA_LIB_DIR="$HOME/miniconda3/envs/$CONDA_ENV/lib"
NVIDIA_CUDART_DIR="$HOME/miniconda3/envs/$CONDA_ENV/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"

require_path() {
    local target_path="$1"
    local desc="$2"
    if [ ! -e "$target_path" ]; then
        echo "[ERROR] $desc 不存在: $target_path"
        exit 1
    fi
}

echo "======================================================"
echo "准备启动大模型 MQTT 推理服务..."
echo "项目根目录: $PROJECT_ROOT"
echo "Conda 环境: $CONDA_ENV"
echo "注意: 同时启动两套 2B 模型大约需要 10GB+ 显存"
echo "======================================================"

require_path "$PROJECT_ROOT" "项目根目录"
require_path "$WATER_SCRIPT" "环水保脚本"
require_path "$EDGE_SCRIPT" "临边防护脚本"
require_path "$WATER_MODEL" "环水保基础模型"
require_path "$WATER_LORA" "环水保 LoRA"
require_path "$EDGE_MODEL" "临边防护 merge 基座"
require_path "$EDGE_LORA" "临边防护 JSON LoRA"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "[ERROR] 未找到 conda.sh，请先确认 Conda 安装路径"
    exit 1
fi

conda activate "$CONDA_ENV"

if [ ! -d "$CONDA_LIB_DIR" ]; then
    echo "[ERROR] Conda 环境 lib 目录不存在: $CONDA_LIB_DIR"
    exit 1
fi

if [ -f "$CUDA_WHEELS_DIR/libcudart.so.12" ]; then
    echo "[INFO] 检测到部署机 libcudart 补丁: $CUDA_WHEELS_DIR/libcudart.so.12"
else
    echo "[WARN] 未检测到 $CUDA_WHEELS_DIR/libcudart.so.12"
    echo "[WARN] 如果启动时报 libcudart.so.12 缺失，请先把该文件放到上述目录"
fi

export LD_LIBRARY_PATH="$CONDA_LIB_DIR:$CUDA_WHEELS_DIR:$NVIDIA_CUDART_DIR:${LD_LIBRARY_PATH:-}"
echo "[INFO] LD_LIBRARY_PATH 已设置"

if pgrep -f "$WATER_SCRIPT" >/dev/null 2>&1; then
    echo "[WARN] 环水保服务已在运行，跳过重复启动"
else
    echo "-> 正在启动 [环水保] MQTT 服务..."
    nohup python "$WATER_SCRIPT" \
        --model-path "$WATER_MODEL" \
        --lora-path "$WATER_LORA" \
        --mqtt-serve > "$LOG_DIR/run_water_console.log" 2>&1 &
    WATER_PID=$!
    echo "$WATER_PID" > "$PID_DIR/water.pid"
    echo "   [PID: $WATER_PID] 控制台日志: $LOG_DIR/run_water_console.log"
    echo "   [PID: $WATER_PID] 业务日志: $LOG_DIR/mqtt_water.log"
fi

if pgrep -f "$EDGE_SCRIPT" >/dev/null 2>&1; then
    echo "[WARN] 临边防护服务已在运行，跳过重复启动"
else
    echo "-> 正在启动 [临边防护] MQTT 服务..."
    nohup python "$EDGE_SCRIPT" \
        --model-path "$EDGE_MODEL" \
        --lora-path "$EDGE_LORA" \
        --mqtt-serve \
        --mqtt-publish-mode immediate > "$LOG_DIR/run_edge_console.log" 2>&1 &
    EDGE_PID=$!
    echo "$EDGE_PID" > "$PID_DIR/edge.pid"
    echo "   [PID: $EDGE_PID] 控制台日志: $LOG_DIR/run_edge_console.log"
    echo "   [PID: $EDGE_PID] 业务日志: $LOG_DIR/mqtt_edge.log"
fi

echo "======================================================"
echo "启动命令已提交到后台"
echo "查看实时业务日志:"
echo "  tail -f $LOG_DIR/mqtt_water.log"
echo "  tail -f $LOG_DIR/mqtt_edge.log"
echo "查看启动控制台日志:"
echo "  tail -f $LOG_DIR/run_water_console.log"
echo "  tail -f $LOG_DIR/run_edge_console.log"
echo "停止服务:"
echo "  bash $PROJECT_ROOT/scripts/stop_mqtt_services.sh"
echo "======================================================"
