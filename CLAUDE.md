# CLAUDE.md — llama-qwen 项目指南

## 项目定位

施工现场安全监测的多模态微调项目。核心任务：对边坡、基坑、平台临边等场景进行识别，输出**结构化 JSON 安全结论**，不是自由文本问答。

项目有两条并行训练线：
- **主线**（通用临边监测）：`train_v2.sh` → Qwen2.5-VL-7B-Instruct
- **专项线**（围栏精细化）：`train_fences.sh` → Qwen3-VL-2B-Instruct

## 关键约定

### assistant 输出格式
训练目标是**纯 JSON，不含推理文本**。这是有意设计，不是 bug。不要在 assistant 字段里加推理过程或 markdown 代码块。

### 标签一致性约束（必须遵守）
```
compliance_status=compliant    → violation_detected=false, violation_type=normal
compliance_status=non_compliant → violation_detected=true
compliance_status=uncertain    → violation_detected=null, violation_type=uncertain, severity=unknown
```

### 坐标格式
violation_boxes 使用 Qwen-VL 的 **0-1000 千分比相对坐标**，不是像素绝对坐标。

## 目录结构

```
data/                    # 训练/验证/测试 JSONL（直接喂给 ms-swift）
dataset/                 # 原始图片（.gitignore 忽略）
dataset_compliant_fences/ # 合规围栏标注（LabelMe JSON）
dataset_fences_only/     # 仅围栏标注
Fences_noncomlaint/      # 违规围栏标注
models/                  # 本地模型权重（.gitignore 忽略）
outputs/                 # 训练输出、评估结果、可视化（.gitignore 忽略）
scripts/                 # 所有 Python 脚本和 shell 训练脚本
```

## 核心脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/api_annotate_v2.py` | 调用 Qwen-VL-Max API 批量自动打标，需要 `DASHSCOPE_API_KEY` |
| `scripts/label_ui.py` | Gradio 人工复核界面，导出 train/val JSONL |
| `scripts/prepare_train_v2_data.py` | 数据清洗：剥离推理文本、统一字段、修正一致性冲突 |
| `scripts/train_v2.sh` | 先清洗再训练（推荐主线，Qwen2.5-VL-7B） |
| `scripts/train_fences.sh` | 围栏专项 QLoRA 微调（Qwen3-VL-2B） |
| `scripts/eval.py` | 测试集离线评估（violation_detected + violation_type 准确率） |
| `scripts/inference.py` | 单图/批量推理，支持 `--lora_path` |
| `scripts/chat.py` | 交互式对话 + 违规区域可视化画框 |
| `scripts/test_fence_violations.py` | 围栏违规召回率测试（50张图，输出 recall rate） |

## 训练参数（当前版本）

**train_fences.sh（围栏专项）**：
- Model: Qwen3-VL-2B-Instruct, 4-bit NF4 量化
- LR: 5e-5, Epochs: 15, Batch: 1, GradAccum: 4
- LoRA rank: 32, alpha: 64, dropout: 0.05
- MAX_PIXELS: 1003520（约 1000×1000，针对无人机俯拍小目标）
- lr_scheduler: cosine, warmup_ratio: 0.05

**train_v2.sh（主线）**：
- Model: Qwen2.5-VL-7B-Instruct, 4-bit 量化
- LR: 5e-5, Epochs: 3, Batch: 1, GradAccum: 4
- LoRA rank: 32, alpha: 64
- MAX_PIXELS: 602112

## 数据文件说明

- `data/train_balanced_clean.jsonl`：135条，清洗后训练集（主线用）
- `data/val_balanced_clean.jsonl`：15条，清洗后验证集
- `data/test.jsonl`：15条，测试集
- `data/finetune_fences.jsonl`：围栏专项训练数据（train_fences.sh 用）

## 输出 JSON Schema

```json
{
  "scene_type": "foundation_pit|slope|platform_or_edge|mixed|unknown",
  "monitoring_content": "...",
  "monitoring_result": "...",
  "key_observations": [],
  "risk_points": [],
  "compliance_status": "compliant|non_compliant|uncertain",
  "violation_detected": true/false/null,
  "violation_type": "normal|no_edge_protection|fence_damaged|guardrail_deformed|warning_missing|unsafe_access|multiple_issues|uncertain",
  "severity": "low|medium|high|critical|unknown",
  "confidence": "low|medium|high",
  "suggestion": "..."
}
```

## 常见陷阱

- **边坡 ≠ 违规**：边坡坡面本体不需要围挡，只有危险临边才需要。这是最核心的误判来源。
- **证据不足时输出 uncertain**，不要强行判断。
- 修改数据时，先跑 `prepare_train_v2_data.py` 清洗，再训练。
- 显存不足时降低 `MAX_PIXELS`，不要降低 `LORA_RANK`（多模态任务需要较高 rank）。

## 环境

```bash
conda activate qwen-ft
# 自动打标需要：
export DASHSCOPE_API_KEY=your_key
```
