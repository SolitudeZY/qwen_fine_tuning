# PRD：工业场景边防护违规识别系统

## 1. 项目概述

### 1.1 背景
工业生产场景中，边防护设施（围栏、护栏、防护网等）是保障人员安全的关键设施。围栏缺损、防护不全等问题如果未及时发现，可能导致严重安全事故。本项目利用视觉大模型微调技术，实现对边防护设施违规情况的自动识别。

### 1.2 目标
基于 Qwen2.5-VL-7B-Instruct 模型，通过 QLoRA 微调，构建一个能够识别工业场景中边防护设施违规情况的视觉问答系统。

### 1.3 范围（V1.0 - 边防护场景）
- 围栏缺损/缺失检测
- 防护栏倾倒/变形检测
- 防护网破损检测
- 临边作业无防护检测

---

## 2. 硬件约束与模型选型

### 2.1 硬件配置
| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 5060 Ti (16GB VRAM) |
| CPU | Intel i5-13400F |
| 内存 | 32GB DDR |
| 磁盘 | 792GB 可用 |
| CUDA | 13.1 |
| PyTorch | 2.10.0+cu130 |

### 2.2 模型选型决策

| 候选模型 | 参数量 | 显存需求(推理) | 显存需求(QLoRA微调) | 工业场景能力 | 是否可行 |
|----------|--------|----------------|---------------------|-------------|---------|
| Qwen2.5-VL-2B | 2B | ~5GB | ~8GB | 一般 | 可行但效果有限 |
| **Qwen2.5-VL-7B-Instruct** | **7B** | **~15GB** | **~12GB (4bit)** | **良好** | **推荐** |
| Qwen2.5-VL-72B | 72B | ~140GB | ~48GB | 优秀 | 不可行 |

**最终选择：Qwen2.5-VL-7B-Instruct + QLoRA (4-bit 量化)**

理由：
1. 16GB VRAM 使用 4-bit QLoRA 微调 7B 模型显存占用 ~12GB，留有余量
2. 7B 模型对复杂场景理解能力远优于 2B
3. Qwen2.5-VL 系列支持动态分辨率，适合工业场景不同尺寸图片
4. Instruct 版本自带指令跟随能力，微调数据量需求更少

---

## 3. 功能需求

### 3.1 核心功能
1. **图片输入** → 模型分析 → 输出违规判定结果（JSON 格式）
2. 支持的违规类型：
   - `fence_missing`：围栏缺失
   - `fence_damaged`：围栏破损
   - `guardrail_deformed`：护栏变形/倾倒
   - `safety_net_broken`：防护网破损
   - `no_edge_protection`：临边无防护
   - `normal`：正常/合规

### 3.2 输出格式
```json
{
  "violation_detected": true,
  "violation_type": "fence_damaged",
  "severity": "high",
  "description": "围栏右侧存在约2米长的缺口，铁丝网已脱落",
  "suggestion": "需要立即修补围栏缺口，设置临时警戒带"
}
```

### 3.3 性能指标
- 准确率目标：>85%（V1.0）
- 单张图片推理时间：<5s（RTX 5060 Ti）
- 支持图片分辨率：最大 1280x1280

---

## 4. 数据需求

### 4.1 数据来源策略
由于工业边防护场景的公开数据集稀缺，采用以下混合策略：

1. **合成数据生成**（主要）：使用大模型生成训练对话数据
2. **公开数据集改造**：从建筑安全/工地安全数据集中筛选相关图片
3. **人工标注**：后续收集真实场景数据并标注

### 4.2 V1.0 数据规模
- 训练集：~500 条（合成 + 公开数据集改造）
- 验证集：~50 条
- 测试集：~50 条

### 4.3 数据格式
使用 Qwen2-VL 微调标准对话格式：
```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一个专业的工业安全检查AI助手，专注于边防护设施的违规识别。"
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/image.jpg"},
        {"type": "text", "text": "请检查这张图片中的边防护设施是否存在违规情况。"}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"violation_detected\": true, ...}"
    }
  ]
}
```

---

## 5. 技术方案

### 5.1 微调方案
- **方法**：QLoRA (4-bit NF4 量化 + LoRA)
- **LoRA 参数**：
  - rank: 64
  - alpha: 128
  - target_modules: 视觉编码器 + LLM 层的 q_proj, k_proj, v_proj, o_proj
  - dropout: 0.05
- **训练参数**：
  - batch_size: 1（梯度累积 4 步）
  - learning_rate: 1e-4
  - epochs: 3-5
  - warmup_ratio: 0.1
  - fp16/bf16: 根据硬件支持选择

### 5.2 项目结构
```
llama-qwen/
├── docs/
│   └── PRD_边防护违规识别.md
├── data/
│   ├── raw/                    # 原始图片
│   ├── train.json              # 训练数据
│   ├── val.json                # 验证数据
│   └── test.json               # 测试数据
├── scripts/
│   ├── prepare_data.py         # 数据准备脚本
│   ├── generate_synthetic.py   # 合成数据生成
│   ├── train.py                # 微调训练脚本
│   ├── eval.py                 # 评估脚本
│   └── inference.py            # 推理脚本
├── configs/
│   └── train_config.yaml       # 训练配置
├── outputs/                    # 模型输出/checkpoint
└── requirements.txt
```

### 5.3 推理部署
- 本地推理：使用 transformers + 4-bit 量化加载
- 后续可接入 API 服务（FastAPI）

---

## 6. 里程碑

| 阶段 | 内容 | 交付物 |
|------|------|--------|
| M1 | 环境搭建 + 依赖安装 | 可运行的开发环境 |
| M2 | 数据准备（合成数据生成） | train/val/test 数据集 |
| M3 | 微调训练 | 微调后的模型 checkpoint |
| M4 | 评估与推理 | 评估报告 + 推理脚本 |

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 16GB 显存不足 | 训练 OOM | 降低图片分辨率/减小 batch size/使用 2B 模型 |
| 合成数据质量差 | 模型效果差 | 设计高质量 prompt，多样化场景描述 |
| 模型过拟合 | 泛化能力差 | 数据增强 + early stopping + dropout |
