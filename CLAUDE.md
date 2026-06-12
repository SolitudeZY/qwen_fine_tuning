# CLAUDE.md — llama-qwen 项目指南

## 项目定位

施工现场安全监测的多模态微调项目。核心任务：对边坡、基坑、平台临边等场景进行识别，输出**结构化 JSON 安全结论**，不是自由文本问答。

当前主线：`scripts/qwen3.5-train/` → Qwen3.5-2B，两阶段训练（grounding + JSON 精调）

## 关键约定

### assistant 输出格式
训练目标是**纯 JSON，不含推理文本**。这是有意设计，不是 bug。不要在 assistant 字段里加推理过程或 markdown 代码块。

### 标签一致性约束（必须遵守）
```
violation_detected=false  → violation_boxes: []
violation_detected=true   → violation_boxes 非空
violation_type 只能是：临边防护缺失
```

### 坐标格式
violation_boxes 使用 **0-1000 千分比相对坐标**，不是像素绝对坐标。

### 常见模型输出 bug
- 模型有时输出 `violation_detection`（少了 `ed`），`chat.py` 的 `_extract_json` 已做兼容
- 模型有时把 bbox 输出为字符串 `"[x,y,x,y]"` 而非数组，`_fix_malformed_json` 已做兼容（括号平衡修复）

## 目录结构

```
data/                         # 训练/验证/测试 JSONL（直接喂给 ms-swift）
  ├── unique_images.txt        # 去重后图片列表（2_stage_train 流程产物）
  ├── annotate_cache.json      # DashScope 打标断点续传缓存（历史数据兼容）
  ├── annotate_cache_nvidia.json  # NVIDIA API 打标缓存
  ├── annotate_skip.json       # 失败/跳过记录（DashScope）
  ├── annotate_skip_nvidia.json   # 失败/跳过记录（NVIDIA）
  ├── stage1_grounding.jsonl   # Stage 1 grounding 训练数据
  ├── stage1_grounding_nvidia.jsonl  # NVIDIA pipeline 产物
  ├── stage1_tiled.jsonl       # Stage 1 大图分块视角训练数据（gen_tiled_stage1.py 产物）
  ├── stage2_json.jsonl        # Stage 2 JSON 训练数据
  ├── stage2_json_nvidia.jsonl # NVIDIA pipeline 产物
  ├── test.jsonl               # eval.py 离线评估测试集
  ├── tiled_crops/             # 大图分块子图（gen_tiled_stage1.py 产物，DJI 大图切的小图）
  └── archive/                 # 旧 jsonl/json（被新流程取代，只读参考）
dataset/                      # 原始图片（.gitignore 忽略）
  ├── 1-100/、221-330/、标注111-220/、北二绕无人机图片标注/、无人机图片标识441-550(1)/
  │                             # 早期人工分批的原始图与 LabelMe 标注
  ├── compliant_fences/         # 合规围栏测试图片
  ├── tiled_1024/               # 大图切成的 1024×1024 子图（待标注）
  ├── video/                    # 无人机原始视频（1920×1080 MP4）
  └── video_frames/             # 视频抽帧产物（extract_video_frames.py 产物）
models/                       # 本地模型权重（.gitignore 忽略）
outputs/                      # 训练输出、评估结果、可视化（.gitignore 忽略）
  ├── stage1_grounding/        # Stage 1 训练 checkpoint
  ├── stage2_json/             # Stage 2 训练 checkpoint
  └── visualized/              # chat.py --visualize 输出图片
scripts/
  ├── 2_stage_train/           # 两阶段训练流程
  │   ├── dedup_dataset.py     # Step 1: 去重过滤
  │   ├── api_annotate_stage12.py  # Step 2: DashScope 打标/缓存生成
  │   ├── api_annotate_nvidia.py   # Step 2 备选: NVIDIA API 打标
  │   ├── gen_tiled_stage1.py  # Step 2b: 大图分块 grounding 样本生成
  │   ├── extract_video_frames.py  # 视频抽帧（场景去重）
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
  └── archive/                 # 旧流程脚本（LabelMe、旧训练、旧数据准备、旧 dedup 工作区）
docs/
  ├── PRD_two_stage_training.md
  └── archive/                 # 旧 PRD 与汇报材料
archive/                      # 项目级归档（早期手测样图、临时脚本）
```

## 当前推荐流程（Qwen3.5）

```bash
# Step 1: 去重
python scripts/2_stage_train/dedup_dataset.py

# Step 2: DashScope 打标/缓存生成（需要 API Key）
export DASHSCOPE_API_KEY=your_key
python scripts/2_stage_train/api_annotate_stage12.py

# Step 3: Stage 1 训练
bash scripts/qwen3.5-train/train_stage1.sh

# Step 4: 合并 LoRA
bash scripts/qwen3.5-train/merge_stage1.sh

# Step 5: Stage 2 训练
bash scripts/qwen3.5-train/train_stage2.sh
```

## 训练参数（当前版本）

**Stage 1（grounding 空间对齐）**：
- Model: `Qwen3.5-2B` 原始基础模型，普通 LoRA + `bfloat16`
- LR: `1e-4`, Epochs: `4`, Batch: `2`, GradAccum: `2`
- LoRA rank: `8`, alpha: `32`, dropout: `0.05`
- 图像 token 上限：`IMAGE_MAX_TOKEN_NUM=1024`
- 训练数据：`stage1_grounding.jsonl`

**Stage 2（JSON 精调）**：
- Model: Stage 1 合并后的 `Qwen3.5-2B` 模型，普通 LoRA + `bfloat16`
- LR: `5e-5`, Epochs: `4`, Batch: `2`, GradAccum: `2`
- LoRA rank: `8`, alpha: `32`, dropout: `0.05`
- 最大序列长度：`4096`

## 推理与测试

```bash
# 交互式测试（支持大图自动分块）
python scripts/chat.py --lora_path outputs/stage2_json/vN/checkpoint-X --visualize

# 单图推理
python scripts/inference.py --image /path/to/image.jpg --lora_path ...
```

### 非实时 MQTT 协议（当前接入）

- 当前非实时图片接入和协议转换逻辑在 `scripts/qwen3.5-train/video_infer_qwen35.py`
- 接收字段使用平台侧 JSON，核心输入字段是 `img`，内容为原图 base64
- 返回字段保持和平台实时格式一致，返回 `img` 时使用**画好框后的图片 base64**
- `desc` 直接使用模型输出里的 `suggestion`
- `detections[].bbox` 对外返回**像素坐标**，不是模型内部使用的 0-1000 千分比坐标
- `detections[].confidence` 优先取模型输出中的 `confidence/score`；若模型未提供，当前默认补 `1.0`

**输入 JSON 示例**：

```json
{
  "img": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "stream_id": "1581F7K3C253B00D1JZY",
  "time": "2026-06-11T14:19:06.636",
  "location": {
    "valid": false,
    "gps": {
      "height": 76.4566909790039,
      "latitude": 28.51231,
      "longitude": 116.09965
    }
  }
}
```

**输出 JSON 示例**：

```json
{
  "desc": "建议尽快安排现场复查并补齐防护",
  "detections": [
    {
      "bbox": [8, 385, 35, 444],
      "class_id": 0,
      "class_name": "临边防护缺失",
      "confidence": 1.0
    }
  ],
  "img": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "location": {
    "valid": false
  },
  "stream_id": "1581F7K3C253B00D1JZY",
  "time": "2026-06-11T14:32:58"
}
```

**字段映射约定**：

- 输入 `img` -> 脚本内部解码为临时图片文件
- 模型输出 `suggestion` -> 返回 `desc`
- 模型输出 `violation_boxes` -> 返回 `detections`
- 返回 `img` -> 画框后的图片 base64；若没有框则回原图 base64
- 输入 `location / stream_id / time` -> 原样透传到返回结果
- 当前 `class_id` 固定为 `0`
- 当前 `class_name` 优先取 box 的 `label`，否则回退到 `violation_type`

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
- 旧数据集（被新流程取代）：`data/archive/`

## 大图推理说明

`chat.py` 对超过 4,000,000 像素的图片（DJI 4032×3024）自动启用双路推理：
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
- 训练与推理阶段的图像 token 上限要保持一致，优先对齐 `IMAGE_MAX_TOKEN_NUM`。
- 显存不足时优先降低 `IMAGE_MAX_TOKEN_NUM`，不要先动 `LORA_RANK`。
- Stage 2 训练数据**不要混入 grounding 格式样本**，会导致重复生成 bug。
- `swift export` 的 `--output_dir` 不能是已存在的目录，否则报错。
- swift 把 checkpoint 存在 `v{N}-xxx/checkpoint-*` 子目录，合并时路径要用 `v*/checkpoint-*` 通配。

## 环境

```bash
conda activate qwen35-ft
export DASHSCOPE_API_KEY=your_key
```

## Qwen3.5 当前状态

### 当前主线
- 当前唯一主线是 `scripts/qwen3.5-train/`，用于 Qwen3.5 的训练、合并与推理。
- `scripts/2_stage_train/` 目前主要保留数据准备脚本和历史流程参考，不再作为主训练入口。
- 当前重点是基于 `Qwen3.5-2B` 做两阶段训练与推理，不再继续使用 `Qwen3-VL-2B-Instruct` 或其他 Qwen-VL 路线作为主实验线。

### 当前环境结论
- 训练历史环境：`qwen35-ft`
- 回滚验证环境：`qwen35-ft-rollback`
- Python：`3.12`
- 本机 `nvcc`：`12.8`
- `CUDA_HOME` 应使用：`/usr/local/cuda-12.8`
- GPU：`NVIDIA GeForce RTX 5060 Ti`
- capability：`sm_120`

### 推理速度关键结论
- 已确认 `torch` 版本会显著影响 `Qwen3.5` 推理速度，不能只看模型路径是否一致。
- 在同一张图、同一 `model-path`、同一 `lora-path`、同一推理脚本下：
  - `torch 2.11.0+cu128` + `attn_impl=flash_attention_2`：在 `sm_120` (RTX 5060 Ti) 架构下触发 `CUDA error: an illegal memory access was encountered`，**不支持生成推理**。
  - `torch 2.11.0+cu128` + `attn_impl=sdpa`：约 `3.9-4.0s/张`（**推荐推理配置**）
  - `torch 2.9.1+cu128` + `attn_impl=sdpa`：约 `17-20s/张`
- 因此做速度对比时，必须同时记录：
  - `torch / torchvision / torchaudio` 版本
  - `transformers / peft` 版本
  - `attn_impl`
  - 是否加载 `flash-attn` / `causal-conv1d`
- 在 `sm_120` 上，不要默认认为预编译 `flash-attn` wheel 一定适合当前环境；推理异常时优先先回到 `torch 2.11.0+cu128` 验证 `sdpa` 基线速度。

### 重要结论
- 不要再切回 `cu130`。
- 训练与推理不要混用不同 `torch` 路线做速度结论。
- 当前推荐优先固定在：
  - 推理基线：`torch=2.11.0+cu128`
  - `CUDA_HOME=/usr/local/cuda-12.8`
- 若继续使用 `torch 2.9.1+cu128`，需要把它视为单独环境，不要和 `torch 2.11.0+cu128` 的速度结果直接对比。
- `causal-conv1d` 和 `flash-attn` 都和当前 `torch + CUDA` 路线强相关，不能混用 `cu128` 和 `cu130`。

### 已确认可用
- `causal-conv1d` 已在 `cu128 + CUDA 12.8 + sm_120` 路线下可用。
- `flash-attn` 已通过预编译 wheel 稳定安装，当前版本：`2.8.3`
- 当前 wheel 已保存到：
  - `/home/fs-ai/llama-qwen/env_packages/qwen3.5/wheels/causal_conv1d-1.6.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
  - `/home/fs-ai/llama-qwen/env_packages/qwen3.5/wheels/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`
- 本地另有一个 `causal-conv1d` wheel：
  - `/home/fs-ai/llama-qwen/env_packages/qwen3.5/wheels/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl`
- 若在 `qwen35-ft-rollback` 中恢复 `torch 2.11.0+cu128`，优先用本地 wheel 验证，再决定是否重编。
- 相关日志和环境快照在：
  - `/home/fs-ai/llama-qwen/env_packages/qwen3.5/`
- 当前环境快照重点文件：
  - `env_report/`
  - `logs/causal_conv1d_build.log`
  - `logs/causal_conv1d_wheel_build.log`
  - `snapshots/current_causal_conv1d.tar.gz`

### 编译注意事项
- 编译 `flash-attn` 或其他 CUDA 扩展时，不要设置 `MAX_JOBS=8`。
- 这台机器内存只有 `32G`，高并发编译会导致 CPU 拉满、SSH 卡死。
- 建议：
  - `MAX_JOBS=1`
  - `OMP_NUM_THREADS=1`
  - 使用GitHub上预编译的wheel：
  ```bash
  wget -O /home/fs-ai/llama-qwen/env_packages/qwen3.5/wheels/causal_conv1d-1.6.2.post1+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
  "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

  wget -O /home/fs-ai/llama-qwen/env_packages/qwen3.5/wheels/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
  
  ```

### 当前 Qwen3.5 关键脚本
- Stage 1 训练：
  - `/home/fs-ai/llama-qwen/scripts/qwen3.5-train/train_stage1.sh`
- Stage 2 训练：
  - `/home/fs-ai/llama-qwen/scripts/qwen3.5-train/train_stage2.sh`
- Qwen3.5 专用推理脚本：
  - `/home/fs-ai/llama-qwen/scripts/qwen3.5-train/chat_qwen35.py`
- 从 cache 随机抽样测试脚本：
  - `/home/fs-ai/llama-qwen/scripts/qwen3.5-train/test_from_cache_qwen35.py`

### 当前脚本能力
- `chat_qwen35.py`
  - 专门适配 `Qwen3.5`
  - 默认不强制 4bit
  - 支持 `--visualize`
  - 支持 `--show-raw-json`
- `test_from_cache_qwen35.py`
  - 从 `annotate_cache.json` 抽样
  - 默认 50 张合规 + 50 张违规
  - 输出 `precision / recall / accuracy / 总耗时 / 平均耗时 / 混淆矩阵`

### 数据与评测
- cache 文件：
  - `/home/fs-ai/llama-qwen/data/annotate_cache.json`
- ground truth 优先取：
  - `global_conclusion.violation_detected`
- 没有时回退：
  - `is_compliant`

### 推理/训练现状
- `causal-conv1d` 最小运行样例已经通过。
- `torchvision nms` 导入问题已解决。
- `flash-attn` 已安装完成，当前训练环境已切换到 `torch 2.9.1+cu128` 路线。

### 下一个会话优先事项
- 优先在 `qwen35-ft-rollback` 中源码编译 `flash-attn`，并与 `sdpa` 做单图 A/B 推理测速。
- 继续保留 `torch 2.11.0+cu128` 作为当前速度基线，不要再用 `torch 2.9.1+cu128` 的结果覆盖它。
- 若训练或推理再次异常，先核对 `torch` 版本，再判断是否是模型或脚本问题。
- 需要时检查：
  - `/home/fs-ai/llama-qwen/env_packages/qwen3.5/logs/flash_attn_build.log`
  - `/home/fs-ai/llama-qwen/scripts/qwen3.5-train/train_stage2.sh`
