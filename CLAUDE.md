# CLAUDE.md — llama-qwen 项目指南

## 项目定位

施工现场安全监测的多模态微调项目。核心任务：对边坡、基坑、平台临边等场景进行识别，输出**结构化 JSON 安全结论**，不是自由文本问答。

项目有两条并行训练线：
- **主线**（通用临边监测）：`train_v2.sh` → Qwen2.5-VL-7B-Instruct
- **专项线**（围栏精细化）：`scripts/2_stage_train/` → Qwen3-VL-2B-Instruct，两阶段训练

## 关键约定

### assistant 输出格式
训练目标是**纯 JSON，不含推理文本**。这是有意设计，不是 bug。不要在 assistant 字段里加推理过程或 markdown 代码块。

### 标签一致性约束（必须遵守）
```
violation_detected=false  → violation_boxes: []
violation_detected=true   → violation_boxes 非空
violation_type 只能是：围栏断口 / 围栏倒伏 / 临边防护缺失 / 临边开口未防护 / 多类型用顿号连接
```

### 坐标格式
violation_boxes 使用 Qwen-VL 的 **0-1000 千分比相对坐标**，不是像素绝对坐标。

### 常见模型输出 bug
- 模型有时输出 `violation_detection`（少了 `ed`），`_extract_json_object` 和 `tiled_infer.py` 已做兼容
- 模型有时把大面积缺失标为"围栏断口"，`tiled_infer.py` 里有面积阈值修正（框面积 > 原图 8% 时改为"临边防护缺失"）

## 目录结构

```
data/                         # 训练/验证/测试 JSONL（直接喂给 ms-swift）
  ├── unique_images.txt        # 去重后图片列表（2_stage_train 流程产物）
  ├── annotate_cache.json      # Qwen-VL-Max 打标断点续传缓存
  ├── stage1_grounding.jsonl   # Stage 1 grounding 训练数据
  ├── stage2_json.jsonl        # Stage 2 JSON 训练数据
  ├── train_balanced_clean.jsonl  # 旧主线训练集（135条）
  └── finetune_fences.jsonl    # 旧围栏专项训练数据
dataset/                      # 原始图片（.gitignore 忽略）
  ├── 1-100/
  ├── 221-330/
  ├── 标注111-220/
  ├── 北二绕无人机图片标注/
  └── 无人机图片标识441-550(1)/
Fences_noncomlaint/           # 违规围栏图片 + LabelMe 标注（旧流程用）
dataset_compliant_fences/     # 合规围栏图片 + LabelMe 标注（旧流程用）
models/                       # 本地模型权重（.gitignore 忽略）
outputs/                      # 训练输出、评估结果、可视化（.gitignore 忽略）
scripts/
  ├── 2_stage_train/           # 新两阶段训练流程（推荐使用）
  │   ├── dedup_dataset.py     # Step 1: 去重过滤
  │   ├── api_annotate_stage12.py  # Step 2: Qwen-VL-Max 打标
  │   ├── train_stage1.sh      # Step 3: Stage 1 训练
  │   ├── merge_stage1.sh      # Step 4: 合并 LoRA
  │   └── train_stage2.sh      # Step 5: Stage 2 训练
  ├── chat.py                  # 交互式对话 + 违规区域可视化（支持大图分块推理）
  ├── tiled_infer.py           # 大图分块推理（全图+2×2滑窗双路合并）
  ├── model_utils.py           # 模型加载/推理统一接口
  ├── test_fence_violations.py # 违规召回率测试（支持大图分块）
  ├── eval.py                  # 测试集离线评估
  └── inference.py             # 单图/批量推理
docs/
  └── PRD_two_stage_training.md  # 两阶段训练方案 PRD
```

## 两阶段训练流程（新，推荐）

```bash
# Step 1: 去重
python scripts/2_stage_train/dedup_dataset.py

# Step 2: Qwen-VL-Max 打标（需要 API Key）
export DASHSCOPE_API_KEY=your_key
python scripts/2_stage_train/api_annotate_stage12.py

# Step 3: Stage 1 训练
bash scripts/2_stage_train/train_stage1.sh

# Step 4: 合并 LoRA
bash scripts/2_stage_train/merge_stage1.sh

# Step 5: Stage 2 训练
bash scripts/2_stage_train/train_stage2.sh
```

## 训练参数（当前版本）

**Stage 1（grounding 空间对齐）**：
- Model: Qwen3-VL-2B-Instruct 原始基础模型，4-bit NF4 量化
- LR: 2e-5, Epochs: 5, Batch: 1, GradAccum: 4
- LoRA rank: 16, alpha: 32, dropout: 0.05
- MAX_PIXELS: 1003520

**Stage 2（JSON 精调）**：
- Model: Stage 1 合并后的模型，4-bit NF4 量化
- LR: 5e-5, Epochs: 15, Batch: 1, GradAccum: 8
- LoRA rank: 32, alpha: 64, dropout: 0.05
- MAX_PIXELS: 1003520

## 推理与测试

```bash
# 交互式测试（支持大图自动分块）
python scripts/chat.py --lora_path outputs/stage2_json/vN/checkpoint-X --visualize

# 批量召回率测试
python scripts/test_fence_violations.py

# 单图推理
python scripts/inference.py --image /path/to/image.jpg --lora_path ...
```

## 大图推理说明

`chat.py` 和 `test_fence_violations.py` 对超过 4,000,000 像素的图片（DJI 4032×3024）自动启用双路推理：
1. **全图低分辨率推理**：识别大面积临边防护缺失
2. **2×2 分块高分辨率推理**：识别小目标围栏断口
3. IoU > 0.3 的重复框合并取外接矩形

## 输出 JSON Schema

```json
{
  "violation_detected": true,
  "violation_type": "围栏断口",
  "severity": "低|中|高",
  "suggestion": "...",
  "violation_boxes": [
    {"label": "围栏断口", "bbox": [x_min, y_min, x_max, y_max]}
  ]
}
```

## 常见陷阱

- **边坡 ≠ 违规**：边坡坡面本体不需要围挡，只有危险临边才需要。
- **证据不足时输出 uncertain**，不要强行判断。
- Stage 1 和 Stage 2 的 MAX_PIXELS 必须一致，否则推理效果下降。
- 显存不足时降低 `MAX_PIXELS`，不要降低 `LORA_RANK`。
- Stage 2 训练数据**不要混入 grounding 格式样本**，会导致重复生成 bug。
- `swift export` 的 `--output_dir` 不能是已存在的目录，否则报错。
- swift 把 checkpoint 存在 `v{N}-xxx/checkpoint-*` 子目录，合并时路径要用 `v*/checkpoint-*` 通配。

## 环境

```bash
conda activate qwen-ft
export DASHSCOPE_API_KEY=your_key
```
