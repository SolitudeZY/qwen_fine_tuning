# PRD：基于 Qwen-VL-Max 自动打标的两阶段微调方案

## 1. 背景与目标

### 问题
- 现有 LabelMe 手工标注质量不可靠（非专业人员标注）
- `dataset/` 目录存在重复图片，需清洗
- 训练数据中 `围栏断口` 占比过高（67%），导致模型对所有违规都输出"围栏断口"
- 现有 `scripts/` 目录脚本混乱，职责不清

### 目标
用 Qwen-VL-Max 商用 API 替代手工标注，全自动生成高质量两阶段训练数据，重新训练本地 Qwen3-VL-2B 模型，提升围栏违规检测的召回率和标签准确率。

---

## 2. 数据处理流程

```
dataset/ (269张，含重复)
    │
    ▼ Step 1: 去重 + 过滤
    │  脚本: 2_stage_train/dedup_dataset.py
    │  输出: data/unique_images.txt（唯一图片路径列表）
    │
    ▼ Step 2: Qwen-VL-Max 批量打标
    │  脚本: 2_stage_train/api_annotate_stage12.py
    │  输出: data/stage1_grounding.jsonl
    │        data/stage2_json.jsonl
    │
    ▼ Step 3: Stage 1 训练（视觉空间对齐）
    │  脚本: 2_stage_train/train_stage1.sh
    │  模型: Qwen3-VL-2B-Instruct（原始基础模型）
    │  输出: outputs/stage1_grounding/
    │
    ▼ Step 4: 合并 LoRA 权重
    │  脚本: 2_stage_train/merge_stage1.sh
    │  输出: outputs/stage1_merged/
    │
    ▼ Step 5: Stage 2 训练（JSON 结构化输出）
       脚本: 2_stage_train/train_stage2.sh
       模型: outputs/stage1_merged/（Stage 1 合并后）
       输出: outputs/stage2_json/
```

---

## 3. 各步骤详细说明

### Step 1: 去重 + 过滤（dedup_dataset.py）

**输入**：`dataset/` 下所有子目录的图片

**处理逻辑**：
1. 遍历所有图片，计算 MD5 哈希
2. 保留每组重复图片中分辨率最高的一张
3. 过滤掉分辨率低于 640×480 的图片
4. 输出唯一图片路径列表到 `data/unique_images.txt`

**输出格式**（每行一个绝对路径）：
```
/home/fs-ai/llama-qwen/dataset/1-100/DJI_0001.JPG
/home/fs-ai/llama-qwen/dataset/221-330/DJI_0365.JPG
...
```

---

### Step 2: Qwen-VL-Max 批量打标（api_annotate_stage12.py）

**输入**：`data/unique_images.txt`

**API 调用策略**：
- 每张图调用一次 Qwen-VL-Max API
- 要求模型输出标准 JSON，包含 `violation_boxes`（0-1000 千分比坐标）
- 支持断点续传（已标注的图片跳过）
- 并发数：3（避免 API 限流）

**Prompt 设计**：
```
你是施工现场安全专家。请分析图片，判断是否存在以下违规：
- 围栏断口：围栏出现缺口、断裂
- 围栏倒伏：围栏倒塌、歪斜
- 临边防护缺失：危险临边完全没有防护

输出严格的 JSON，字段：
- violation_detected: bool
- violation_type: 违规类型描述
- severity: 低/中/高
- suggestion: 整改建议
- violation_boxes: [{"label": "...", "bbox": [x_min,y_min,x_max,y_max]}]
  坐标为 0-1000 千分比，violation_detected=false 时为 []
```

**从 Stage 2 JSON 派生 Stage 1 Grounding 数据**（无需二次 API 调用）：

```
Stage 2 输出:
{"violation_boxes": [{"label": "围栏断口", "bbox": [488, 180, 743, 294]}]}

↓ 转换

Stage 1 grounding 格式:
{
  "messages": [
    {"role": "user", "content": "<image>请找出图中所有围栏断口的位置"},
    {"role": "assistant", "content": "<ref-object><bbox>"}
  ],
  "objects": {
    "ref": ["围栏断口"],
    "bbox": [[488, 180, 743, 294]],
    "bbox_type": "norm1000"
  }
}
```

**输出**：
- `data/stage2_json.jsonl`：Stage 2 训练数据（含合规+违规）
- `data/stage1_grounding.jsonl`：Stage 1 grounding 数据（仅违规图片）
- `data/annotate_cache.json`：断点续传缓存

---

### Step 3: Stage 1 训练（train_stage1.sh）

**目的**：教模型视觉空间对齐，学会"看到哪里 → 输出那里的坐标"

**关键参数**：
```bash
MODEL: Qwen3-VL-2B-Instruct（原始基础模型）
EPOCHS: 5
LR: 2e-5
LORA_RANK: 16, ALPHA: 32
MAX_PIXELS: 1003520
```

---

### Step 4: 合并 LoRA（merge_stage1.sh）

```bash
swift export \
    --adapters outputs/stage1_grounding/v{N}/checkpoint-{best} \
    --merge_lora true \
    --output_dir outputs/stage1_merged
```

自动找最新 checkpoint，合并后输出完整模型权重。

---

### Step 5: Stage 2 训练（train_stage2.sh）

**目的**：在 Stage 1 空间对齐能力基础上，学习 JSON 结构化输出

**关键参数**：
```bash
MODEL: outputs/stage1_merged/（Stage 1 合并后）
EPOCHS: 15
LR: 5e-5
LORA_RANK: 32, ALPHA: 64
MAX_PIXELS: 1003520
```

---

## 4. 输出 JSON Schema

```json
{
  "violation_detected": true,
  "violation_type": "围栏断口",
  "severity": "中",
  "suggestion": "请立即修复围栏断口，确保临边防护完整。",
  "violation_boxes": [
    {"label": "围栏断口", "bbox": [488, 180, 743, 294]}
  ]
}
```

**标签一致性约束**：
- `violation_detected=false` → `violation_boxes: []`
- `violation_detected=true` → `violation_boxes` 非空
- `violation_type` 只能是：`围栏断口`、`围栏倒伏`、`临边防护缺失`、多类型用顿号连接

---

## 5. 目录结构

```
scripts/2_stage_train/
├── dedup_dataset.py          # Step 1: 去重过滤
├── api_annotate_stage12.py   # Step 2: Qwen-VL-Max 打标
├── train_stage1.sh           # Step 3: Stage 1 训练
├── merge_stage1.sh           # Step 4: 合并 LoRA
└── train_stage2.sh           # Step 5: Stage 2 训练

data/
├── unique_images.txt         # 去重后图片列表
├── annotate_cache.json       # 打标断点续传缓存
├── stage1_grounding.jsonl    # Stage 1 训练数据
└── stage2_json.jsonl         # Stage 2 训练数据
```

---

## 6. 环境要求

```bash
conda activate qwen-ft
export DASHSCOPE_API_KEY=your_key
```

依赖：`dashscope`、`Pillow`、`tqdm`、`ms-swift`

---

## 7. 预期效果

| 指标 | 当前基线 | 目标 |
|------|---------|------|
| 违规召回率 | 54% | > 75% |
| 标签准确率（围栏断口 vs 临边防护缺失） | 低 | 明显提升 |
| 重复生成问题 | 偶发 | 消除 |
