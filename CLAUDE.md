# CLAUDE.md — llama-qwen 项目指南

## 项目定位

施工现场安全监测的多模态微调项目。核心任务：对边坡、基坑、平台临边等场景进行识别，输出**结构化 JSON 安全结论**，不是自由文本问答。

当前主线：`scripts/2_stage_train/` → Qwen3-VL-2B-Instruct，两阶段训练（grounding + JSON精调）

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
- 模型有时输出 `violation_detection`（少了 `ed`），`chat.py` 的 `_extract_json` 已做兼容
- 模型有时把 bbox 输出为字符串 `"[x,y,x,y]"` 而非数组，`_fix_malformed_json` 已做兼容（括号平衡修复）

## 目录结构

```
data/                         # 训练/验证/测试 JSONL（直接喂给 ms-swift）
  ├── unique_images.txt        # 去重后图片列表（2_stage_train 流程产物）
  ├── annotate_cache.json      # Qwen-VL-Max 打标断点续传缓存
  ├── stage1_grounding.jsonl   # Stage 1 grounding 训练数据
  ├── stage1_tiled.jsonl       # Stage 1 大图分块视角训练数据（gen_tiled_stage1.py 产物）
  ├── stage2_json.jsonl        # Stage 2 JSON 训练数据
  └── tiled_crops/             # 大图分块子图（gen_tiled_stage1.py 产物）
dataset/                      # 原始图片（.gitignore 忽略）
Fences_noncomlaint/           # 违规围栏测试图片
dataset_compliant_fences/     # 合规围栏测试图片
models/                       # 本地模型权重（.gitignore 忽略）
outputs/                      # 训练输出、评估结果、可视化（.gitignore 忽略）
  ├── stage1_grounding/        # Stage 1 训练 checkpoint
  ├── stage2_json/             # Stage 2 训练 checkpoint
  └── visualized/              # chat.py --visualize 输出图片
scripts/
  ├── 2_stage_train/           # 两阶段训练流程
  │   ├── dedup_dataset.py     # Step 1: 去重过滤
  │   ├── api_annotate_stage12.py  # Step 2: Qwen-VL-Max 打标
  │   ├── gen_tiled_stage1.py  # Step 2b: 大图分块 grounding 样本生成
  │   ├── train_stage1.sh      # Step 3: Stage 1 训练
  │   ├── merge_stage1.sh      # Step 4: 合并 LoRA
  │   ├── train_stage2.sh      # Step 5: Stage 2 训练
  │   ├── review_annotations.py  # 人工审查标注缓存
  │   ├── fix_cache_bbox.py    # 修复缓存中的 bbox 格式
  │   └── test_v2.py           # 两阶段模型评估
  ├── chat.py                  # 交互式推理 + 可视化（--tiled 启用分块）
  ├── tiled_infer.py           # 大图分块推理（全图+2×2滑窗双路合并）
  ├── model_utils.py           # 模型加载/推理统一接口
  ├── prompts.py               # 系统提示词、查询模板、测试目录配置
  ├── eval.py                  # 测试集离线评估
  ├── inference.py             # 单图/批量推理
  └── archive/                 # 旧流程脚本（LabelMe、旧训练、旧数据准备）
docs/
  └── PRD_two_stage_training.md
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
- LR: 3e-5, Epochs: 8, Batch: 1, GradAccum: 4
- LoRA rank: 32, alpha: 64, dropout: 0.05
- MAX_PIXELS: 1505280
- 训练数据：`stage1_grounding.jsonl` + `stage1_tiled.jsonl`（大图分块视角）

**Stage 2（JSON 精调）**：
- Model: Stage 1 合并后的模型，4-bit NF4 量化
- LR: 5e-5, Epochs: 15, Batch: 1, GradAccum: 8
- LoRA rank: 32, alpha: 64, dropout: 0.05
- MAX_PIXELS: 1003520

## 推理与测试

```bash
# 交互式测试（支持大图自动分块）
python scripts/chat.py --lora_path outputs/stage2_json/vN/checkpoint-X --visualize

# 单图推理
python scripts/inference.py --image /path/to/image.jpg --lora_path ...
```

## 文件命名规范

### scripts/ 根目录
只放**当前活跃、直接面向用户**的脚本，命名用动词或功能名，不加版本号后缀：
- `chat.py` — 交互式推理入口
- `inference.py` — 批量/单图推理
- `eval.py` — 离线评估
- `tiled_infer.py` — 分块推理核心逻辑（被 chat.py 调用）
- `model_utils.py` — 模型加载/推理工具函数
- `prompts.py` — 提示词与配置常量

### scripts/2_stage_train/
两阶段训练流程脚本，命名体现流程步骤：
- `step_N_动词_对象.py/.sh`，或直接用功能名
- 训练脚本：`train_stage{1,2}.sh`、`merge_stage1.sh`
- 数据脚本：`dedup_dataset.py`、`api_annotate_stage12.py`、`gen_tiled_stage1.py`
- 工具脚本：`review_annotations.py`、`fix_cache_bbox.py`、`test_v2.py`

### 禁止在活跃目录出现的命名模式
- `*_v1.py`、`*_v2.py` — 版本迭代用 git，不用文件名区分
- `*_old.py`、`*_bak.py` — 旧版本移入 `archive/`
- `test_*.py`（临时调试）— 调试完删除，正式测试脚本放 `2_stage_train/`
- `prepare_*.py`（旧数据准备）— 已归档，新数据准备统一走 `2_stage_train/` 流程

### archive/
存放已被新流程取代的旧脚本，只读参考，不再维护。

### data/
- 训练数据：`stage{1,2}_*.jsonl`
- 临时/中间产物：`*_cache.json`、`tiled_crops/`

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
