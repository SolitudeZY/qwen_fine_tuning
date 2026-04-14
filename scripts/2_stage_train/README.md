# 两阶段训练流程使用指南

施工现场围栏违规检测，基于 Qwen3-VL-2B-Instruct 的两阶段微调方案。

## 目录结构

```
scripts/2_stage_train/
├── dedup_dataset.py        # Step 1: 图片去重
├── api_annotate_stage12.py # Step 2: Qwen-VL-Max API 打标
├── review_annotations.py   # Step 3: 人工审查 Web UI
├── fix_cache_bbox.py       # 工具: 离线修复越界 bbox
├── train_stage1.sh         # Step 4: Stage 1 训练
├── merge_stage1.sh         # Step 5: 合并 LoRA
└── train_stage2.sh         # Step 6: Stage 2 训练
```

## 完整流程

### 环境准备

```bash
conda activate qwen-ft
export DASHSCOPE_API_KEY=your_key
```

---

### Step 1：图片去重

```bash
python scripts/2_stage_train/dedup_dataset.py
```

- 输入：`dataset/` 下所有图片
- 输出：`data/unique_images.txt`（去重后图片列表）
- 报告：`data/dedup_report.txt` / `data/dedup_report.json`
- 去重策略：精确去重（MD5）+ 感知去重（pHash，Hamming 阈值 10）

---

### Step 2：API 打标

```bash
python scripts/2_stage_train/api_annotate_stage12.py
```

- 输入：`data/unique_images.txt`
- 输出：
  - `data/annotate_cache.json`：canonical annotation（中间真值，断点续传）
  - `data/stage1_grounding.jsonl`：Stage 1 训练数据
  - `data/stage2_json.jsonl`：Stage 2 训练数据
  - `data/annotate_skip.json`：失败/跳过记录
- 支持断点续传，中断后重跑自动跳过已标注图片

常用参数：

| 参数 | 说明 |
|------|------|
| `--derive-only` | 跳过 API 调用，直接从 cache 重新派生 JSONL |
| `--dry-run` | 只打印，不调用 API |
| `--threshold 0.5` | 跳过置信度低于 medium 的图片 |

---

### Step 3：人工审查

```bash
python scripts/2_stage_train/review_annotations.py
```

本机通过 SSH 端口转发访问：

```bash
# 本机执行
ssh -L 5001:localhost:5001 user@server
# 浏览器打开
http://localhost:5001
```

功能说明：

- 低置信度图片自动排在前面
- 过滤栏：全部 / 未审查 / 低置信度 / 已审查
- 在图片上**拖拽鼠标**新增违规框，左上角选择标签
- 越界 bbox 用红色虚线标出，保存时自动修正
- "排除此图"按钮：将无关场景图片移出训练集

快捷键：`←` `→` 翻页，`Y` 违规保存，`N` 合规保存，`S` 仅保存

审查完成后重新派生 JSONL：

```bash
python scripts/2_stage_train/api_annotate_stage12.py --derive-only
```

---

### Step 4：Stage 1 训练（Grounding 空间对齐）

```bash
bash scripts/2_stage_train/train_stage1.sh
```

| 参数 | 值 |
|------|----|
| 基础模型 | Qwen3-VL-2B-Instruct |
| 训练数据 | `data/stage1_grounding.jsonl` |
| LoRA rank | 16，alpha 32 |
| LR | 2e-5，Epochs 5 |
| MAX_PIXELS | 1003520 |

训练日志：`outputs/stage1_grounding/train.log`

---

### Step 5：合并 LoRA

```bash
bash scripts/2_stage_train/merge_stage1.sh
```

- 自动找最新 checkpoint
- 合并后模型保存在 `outputs/stage1_grounding/vN/checkpoint-X-merged`
- 脚本会打印合并后路径，将其填入 `train_stage2.sh` 的 `MERGED_MODEL_DIR`

---

### Step 6：Stage 2 训练（JSON 精调）

编辑 `train_stage2.sh`，确认 `MERGED_MODEL_DIR` 指向 Step 5 的合并路径，然后：

```bash
bash scripts/2_stage_train/train_stage2.sh
```

| 参数 | 值 |
|------|----|
| 基础模型 | Stage 1 合并后模型 |
| 训练数据 | `data/stage2_json.jsonl` |
| LoRA rank | 32，alpha 64 |
| LR | 5e-5，Epochs 15 |
| MAX_PIXELS | 1003520（必须与 Stage 1 一致） |

---

### 训练完成后验证

```bash
# 召回率测试
python scripts/test_fence_violations.py

# 交互式推理（支持大图分块）
python scripts/chat.py \
  --lora_path outputs/stage2_json/vN/checkpoint-X \
  --visualize
```

---

## 标签体系

| 标签 | 说明 |
|------|------|
| 围栏断口 | 围栏本体存在破损或缺口 |
| 围栏倒伏 | 围栏整体或局部倒塌 |
| 临边防护缺失 | 危险临边完全没有任何防护结构 |
| 临边开口未防护 | 出入口或开口处缺少防护门、警示带等隔离措施 |

坐标格式：`bbox_1000` 使用 0-1000 千分比相对坐标 `[x_min, y_min, x_max, y_max]`。

---

## 常见问题

**显存不足**：降低 `MAX_PIXELS`（1003520 → 501760），Stage 1 和 Stage 2 必须保持一致，不要降低 `LORA_RANK`。

**Stage 2 重复生成**：训练数据混入了 grounding 格式样本（含 `<ref-object><bbox>`），`stage2_json.jsonl` 只能包含纯 JSON 格式样本。

**swift export 路径**：合并后模型在 `checkpoint-X-merged`，不在 `--output_dir` 指定的路径。

**bbox 越界**：运行 `fix_cache_bbox.py` 对现有 cache 做离线清洗。

```bash
# 快速测试（默认）
  python scripts/test_fence_violations.py

  # 高精度测试
  python scripts/test_fence_violations.py --tiled

  # chat 快速模式
  python scripts/chat.py --image /path/to/img.jpg --visualize

  # chat 分块模式
  python scripts/chat.py --image /path/to/img.jpg --tiled --visualize
```