#!/bin/bash
# 一键停止大模型 MQTT 推理服务

echo "======================================================"
echo "准备停止大模型 MQTT 推理服务..."
echo "======================================================"

# 查找并终止环水保脚本
WATER_PIDS=$(pgrep -f "04_mqtt_inference.py" || true)
if [ -n "$WATER_PIDS" ]; then
    echo "正在终止 [环水保] 进程: $WATER_PIDS"
    kill $WATER_PIDS
else
    echo "[环水保] 进程未运行。"
fi

# 查找并终止临边防护脚本
EDGE_PIDS=$(pgrep -f "video_infer_qwen35.py" || true)
if [ -n "$EDGE_PIDS" ]; then
    echo "正在终止 [临边防护] 进程: $EDGE_PIDS"
    kill $EDGE_PIDS
else
    echo "[临边防护] 进程未运行。"
fi

echo "======================================================"
echo "所有大模型 MQTT 推理服务已停止！"
echo "======================================================"
