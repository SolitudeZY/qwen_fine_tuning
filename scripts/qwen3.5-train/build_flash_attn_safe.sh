#!/bin/bash
# 专门为 32G 内存机器编写的 flash-attn 安全编译脚本
# 该脚本会严格限制编译并发数，防止 OOM 导致 SSH 卡死。

set -e

echo "==========================================================="
echo "开始编译 Flash-Attention (适用于 Torch 2.11 + CUDA 12.8)"
echo "当前环境: qwen35-ft-rollback"
echo "==========================================================="

# 1. 环境变量设置
export CONDA_ENV="qwen35-ft-rollback"
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# 最核心的防卡死参数 (32G内存只能单线程编译)
export MAX_JOBS=1
export OMP_NUM_THREADS=1

# 显卡架构 (RTX 5060 Ti -> sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"

# 工作目录
WORK_DIR="/home/fs-ai/llama-qwen/env_packages/qwen3.5/src"
WHEEL_DIR="/home/fs-ai/llama-qwen/env_packages/qwen3.5/wheels"
mkdir -p $WORK_DIR
mkdir -p $WHEEL_DIR

cd $WORK_DIR

# 2. 拉取或更新源码
if [ ! -d "flash-attention" ]; then
    echo "[1/4] 拉取 flash-attention 源码..."
    git clone https://github.com/Dao-AILab/flash-attention.git
fi

cd flash-attention
# 切换到最新的稳定版本 2.7.2.post1 或 2.8.3
echo "[2/4] 切换版本 v2.8.3..."
git fetch --tags
git checkout v2.8.3
git submodule update --init --recursive

# 3. 执行编译
echo "[3/4] 开始编译 (预计需要 15 - 30 分钟，请耐心等待，切勿中断)..."
# 使用 conda run 在指定环境中执行编译命令，生成 wheel 而不是直接安装
conda run -n $CONDA_ENV env MAX_JOBS=1 OMP_NUM_THREADS=1 TORCH_CUDA_ARCH_LIST="12.0" python setup.py bdist_wheel

# 4. 拷贝产物
echo "[4/4] 编译完成，正在拷贝 wheel 文件..."
cp dist/*.whl $WHEEL_DIR/
ls -lh $WHEEL_DIR/*.whl

echo "==========================================================="
echo "编译成功！您可以在 $WHEEL_DIR 下找到生成的 wheel 文件。"
echo "安装命令: conda run -n $CONDA_ENV pip install $WHEEL_DIR/<生成的文件名>.whl"
echo "==========================================================="
