# llama-qwen

这是一个面向施工现场安全监测的多模态微调项目，当前重点任务是基于 Qwen2.5-VL 对边坡、基坑、平台临边等场景进行识别，并输出结构化的安全监测结论。项目覆盖了从自动打标、人工复核、数据清洗、QLoRA 微调到评估、推理和可视化的完整闭环。

## 1. 项目目标

本项目不是做通用图像问答，而是做一个偏工程安全场景的视觉监测助手。模型需要完成的核心任务包括：

- 识别施工现场更接近 `foundation_pit`、`slope`、`platform_or_edge`、`mixed` 还是 `unknown`
- 判断是否存在临边高坠风险与防护缺陷
- 输出结构化 JSON 监测结果，而不是自由文本描述
- 尽量减少“把边坡误判为必须围挡对象”的问题
- 在证据不足时输出 `uncertain`，避免过度推断

当前数据和脚本设计围绕“边坡/基坑临边防护监测”展开，尤其强调：

- 区分“边坡坡面本体”和“真正需要防护的危险临边”
- 只基于图像可见证据判断
- 保持 `compliance_status`、`violation_detected`、`violation_type` 之间的一致性

## 2. 当前工作流概览

项目目前形成了如下工作流：

1. 收集并整理现场图片，放入 `dataset/`
2. 使用线上视觉模型进行初始自动打标
3. 通过人工界面复核和修正标签
4. 导出为 ms-swift 可训练的 JSONL 数据
5. 对导出数据做二次清洗，统一成稳定的纯 JSON 输出格式
6. 使用 ms-swift 执行 QLoRA 微调
7. 用测试脚本、推理脚本和交互式脚本评估模型效果

如果只想快速上手，建议直接使用：

- 数据清洗 + 训练：`scripts/train_v2.sh`
- 单图推理：`scripts/inference.py`
- 测试集评估：`scripts/eval.py`
- 交互式对话与违规区域可视化：`scripts/chat.py`

## 3. 目录说明

项目主要目录含义如下：

```text
llama-qwen/
├── data/                   # 训练、验证、测试数据和清洗后数据
├── dataset/                # 原始图片与标注数据
├── docs/                   # 需求说明等补充文档
├── models/                 # 本地基础模型目录
├── outputs/                # 自动标注结果、训练输出、可视化结果
├── scripts/                # 数据准备、训练、推理、评估、交互脚本
├── 1.md                    # 临时记录文件
└── README.MD               # 当前说明文档
```

几个关键子目录：

- `dataset/`：原始图片数据，当前仓库中按批次目录存放
- `data/`：模型训练直接读取的数据文件
- `outputs/edge_protection_v2`：已有训练输出目录
- `outputs/visualized`：交互式脚本生成的违规标注图
- `models/Qwen/Qwen2.5-VL-7B-Instruct`：基础视觉模型目录

注意：

- `.gitignore` 已忽略 `dataset/`、`models/`、`outputs/` 等大目录
- 这意味着代码会跟踪，但原始数据、模型权重和训练结果通常不会被提交

## 4. 环境与依赖

脚本默认使用 conda 环境：

```bash
conda activate qwen-ft
```

从现有脚本依赖看，至少需要以下核心组件：

- Python 3.11 左右
- PyTorch
- transformers
- peft
- bitsandbytes
- ms-swift
- qwen-vl-utils
- gradio
- dashscope
- pillow
- opencv-python
- numpy
- tqdm

如果你要运行全部流程，建议优先保证以下两类依赖可用：

- 训练与推理相关：`torch`、`transformers`、`peft`、`bitsandbytes`、`ms-swift`
- 标注与界面相关：`dashscope`、`gradio`、`Pillow`、`opencv-python`

## 5. 数据格式

项目训练数据采用 ms-swift 兼容的多模态对话格式，每条样本通常包含：

- `messages`
- `images`

其中 `messages` 内部固定为三轮：

1. `system`：定义角色和判定规则
2. `user`：带 `<image>` 占位的任务指令
3. `assistant`：模型应学习输出的 JSON 结果

一个简化示例如下：

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是一名严谨的边坡与基坑施工安全监测专家..."
    },
    {
      "role": "user",
      "content": "<image>请对这张施工现场图片执行边坡/基坑安全监测..."
    },
    {
      "role": "assistant",
      "content": "{\"scene_type\":\"slope\",\"compliance_status\":\"non_compliant\",\"violation_detected\":true}"
    }
  ],
  "images": [
    "/absolute/path/to/image.jpg"
  ]
}
```

### 5.1 输出 JSON 规范

清洗后的训练目标要求模型只输出一个 JSON 对象，至少包含以下字段：

- `scene_type`
- `monitoring_content`
- `monitoring_result`
- `key_observations`
- `risk_points`
- `compliance_status`
- `violation_detected`
- `violation_type`
- `severity`
- `confidence`
- `suggestion`

### 5.2 一致性约束

项目对标签一致性要求较严格：

- `compliance_status=compliant` 时：
  - `violation_detected=false`
  - `violation_type=normal`
- `compliance_status=non_compliant` 时：
  - `violation_detected=true`
- `compliance_status=uncertain` 时：
  - `violation_detected=null`
  - `violation_type=uncertain`
  - `severity=unknown`

这部分约束已经在数据清洗脚本中做了统一修正。

## 6. 当前数据文件

当前 `data/` 目录中较重要的文件有：

- `train_balanced.jsonl`：人工复核后导出的平衡训练集
- `val_balanced.jsonl`：人工复核后导出的验证集
- `train_balanced_clean.jsonl`：清洗后的训练集
- `val_balanced_clean.jsonl`：清洗后的验证集
- `test.jsonl`：测试集

按当前仓库内容统计：

- `train_balanced.jsonl`：135 条
- `val_balanced.jsonl`：15 条
- `train_balanced_clean.jsonl`：135 条
- `val_balanced_clean.jsonl`：15 条
- `test.jsonl`：15 条

这说明项目当前更像是“小样本、强约束、特定场景”的垂直微调实验，而不是大规模通用视觉指令微调。

## 7. 核心脚本说明

### 7.1 自动打标：`scripts/api_annotate_v2.py`

用途：

- 调用阿里云百炼的 Qwen-VL-Max 接口进行批量自动打标
- 将现场图片转成结构化监测结果
- 输出中间态 JSON，供后续人工修正

特点：

- 目标场景已经从通用“边防护检查”收敛到“边坡/基坑临边监测”
- 强调 `uncertain` 输出，减少误判
- 会尽量把 API 返回结果标准化为固定字段

运行前需要准备：

- 正确的 Python 环境
- `DASHSCOPE_API_KEY` 环境变量
- `dataset/` 目录中存在图片

## 7.2 人工复核界面：`scripts/label_ui.py`

用途：

- 读取自动打标结果
- 以网页界面的方式逐张复核图片
- 允许直接修改推理文本和 JSON
- 手工确认“违规”或“合规”
- 最终导出训练和验证数据

导出结果：

- `data/train_balanced.jsonl`
- `data/val_balanced.jsonl`

这个脚本适合做“自动标注 + 人工纠偏”的中间环节，能够显著提升小样本场景下标签质量。

## 7.3 原始标注转训练数据：`scripts/prepare_data.py`

用途：

- 从 LabelMe 类标注数据中构造 ms-swift 所需的多模态对话格式
- 根据标签推导是否存在防护、是否存在高风险区域
- 自动生成训练/验证/测试拆分

适用情况：

- 你已经有多边形框、矩形框或目标标签
- 想从规则生成的方式构建第一版训练集

## 7.4 数据清洗：`scripts/prepare_train_v2_data.py`

用途：

- 清洗 `train_balanced.jsonl` 和 `val_balanced.jsonl`
- 把 assistant 输出统一为纯 JSON
- 修正枚举值和字段不一致问题
- 自动补全缺失字段

这是当前训练流程里非常关键的一步，因为早期样本里存在：

- 有的 assistant 带推理文本 + fenced JSON
- 有的字段缺失
- 有的 `violation_detected` 与 `compliance_status` 冲突

脚本会将其整理成更稳定、更适合 SFT 学习的格式。

## 7.5 训练脚本：`scripts/train.sh`

用途：

- 使用 ms-swift 对 Qwen2.5-VL-7B-Instruct 执行 QLoRA 微调

主要设置：

- 4bit bnb 量化
- LoRA rank 64
- LoRA alpha 128
- 训练 5 个 epoch
- 学习率 `1e-4`

适用情况：

- 想快速基于原始平衡数据跑一版训练

但从当前项目演进来看，更推荐使用 `train_v2.sh`。

## 7.6 改进版训练脚本：`scripts/train_v2.sh`

用途：

- 先清洗数据，再训练
- 使用更保守的训练参数，降低小数据集过拟合和格式漂移风险

主要设置：

- 先执行 `prepare_train_v2_data.py`
- LoRA rank 32
- LoRA alpha 64
- 学习率 `5e-5`
- 训练 3 个 epoch
- 每 20 step 验证和保存一次

推荐原因：

- 数据格式更干净
- 输出目标更统一
- 训练参数更稳

如果你现在继续迭代模型，优先从这个脚本开始。

## 7.7 推理脚本：`scripts/inference.py`

用途：

- 加载基础模型
- 可选加载 LoRA 权重
- 对单张图片或整个目录进行推理
- 将结果保存为 JSON 文件

支持方式：

- `--image`：单图推理
- `--image_dir`：批量推理
- `--lora_path`：指定微调后的 LoRA 权重
- `--query`：覆盖默认提问词

## 7.8 评估脚本：`scripts/eval.py`

用途：

- 使用测试集对模型做离线评估
- 对比标签中的 `violation_detected` 和 `violation_type`
- 输出准确率和详细结果

当前实现重点评估两个维度：

- 违规判定准确率
- 违规类型准确率

适合在每次训练后做快速回归检查。

## 7.9 交互与可视化：`scripts/chat.py`

用途：

- 交互式输入图片路径进行对话测试
- 支持直接指定单图
- 支持测试集批量测试
- 支持违规区域可视化

特点：

- 支持输出推理过程 + JSON
- 如果模型没有直接给出 `violation_boxes`，会触发第二阶段定位
- 可以自动生成带框的标注图

适合做模型质检和演示。

## 8. 推荐使用流程

### 8.1 自动打标

```bash
conda activate qwen-ft
python /home/fs-ai/llama-qwen/scripts/api_annotate_v2.py
```

输出通常写入：

```text
/home/fs-ai/llama-qwen/outputs/api_annotations_cot.json
```

### 8.2 人工复核

```bash
conda activate qwen-ft
python /home/fs-ai/llama-qwen/scripts/label_ui.py
```

在界面中：

- 查看图片和模型推理
- 修正 JSON
- 确认违规或合规
- 导出平衡训练集

### 8.3 清洗并训练

```bash
bash /home/fs-ai/llama-qwen/scripts/train_v2.sh
```

训练输出目录默认是：

```text
/home/fs-ai/llama-qwen/outputs/edge_protection_v2_clean
```

### 8.4 测试集评估

```bash
python /home/fs-ai/llama-qwen/scripts/eval.py \
  --model_path /home/fs-ai/llama-qwen/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --lora_path /home/fs-ai/llama-qwen/outputs/edge_protection_v2_clean/checkpoint-xxx \
  --test_data /home/fs-ai/llama-qwen/data/test.jsonl \
  --output /home/fs-ai/llama-qwen/outputs/eval_results.json
```

### 8.5 单图推理

```bash
python /home/fs-ai/llama-qwen/scripts/inference.py \
  --model_path /home/fs-ai/llama-qwen/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --lora_path /home/fs-ai/llama-qwen/outputs/edge_protection_v2_clean/checkpoint-xxx \
  --image /home/fs-ai/llama-qwen/dataset/221-330/DJI_0378.JPG \
  --output /home/fs-ai/llama-qwen/outputs/inference_results.json
```

### 8.6 交互式对话与画框

```bash
python /home/fs-ai/llama-qwen/scripts/chat.py \
  --lora_path /home/fs-ai/llama-qwen/outputs/edge_protection_v2_clean/checkpoint-xxx \
  --visualize \
  --output-dir /home/fs-ai/llama-qwen/outputs/visualized
```

## 9. 训练策略说明

当前项目采用的是 QLoRA 微调，而不是全参微调。

这么做的原因主要有：

- 视觉大模型 7B 规模较大
- 当前数据量较小
- 本地显存有限
- 训练目标更偏“格式约束 + 场景纠偏”

QLoRA 在这里的意义不是让模型学会所有视觉知识，而是：

- 让模型更理解“边坡 ≠ 一定违规”
- 让模型学会稳定输出项目要求的结构化 JSON
- 让模型在特定场景下减少误判与幻觉

### 9.1 为什么推荐清洗后训练

如果训练目标同时混有：

- 推理文本
- Markdown 代码块
- 纯 JSON
- 字段命名不统一

模型就容易出现以下问题：

- 输出格式不稳定
- 同一场景答案风格飘移
- 标签逻辑前后矛盾

因此 `prepare_train_v2_data.py` 的作用不是“锦上添花”，而是“稳定训练目标”的核心步骤。

## 10. 评估建议

当前 `eval.py` 已经可以快速统计基础效果，但如果你想更可靠地评估模型，建议同时观察以下几个层面：

- `violation_detected` 是否判断正确
- `violation_type` 是否分类正确
- `scene_type` 是否稳定
- `uncertain` 是否被合理使用
- `risk_points` 是否和图像证据一致
- 模型是否经常把单纯边坡坡面误判为违规

对于这个项目来说，最关键的并不只是“分类准确率”，而是：

- 是否减少工程语义误判
- 是否输出稳定可解析的 JSON
- 是否更适合作为后续安全巡检系统的结构化模块

## 11. 常见问题

### 11.1 为什么模型会把边坡误判为违规

这是该任务最核心的难点之一。原因通常有：

- 数据中“边坡”和“危险临边”没有严格区分
- 训练样本里负样本不足
- 提示词强调了风险，但没有强调“证据不足时输出 uncertain”
- assistant 标签格式不统一，导致模型学偏

当前项目已经通过以下方式缓解：

- 新系统提示明确区分边坡坡面与危险临边
- 清洗脚本统一标签逻辑
- 使用平衡数据导出

### 11.2 为什么推荐使用 `train_v2.sh`

因为它在训练前先做清洗，能降低以下风险：

- 格式污染
- 标签冲突
- 小样本场景下的输出抖动

### 11.3 为什么 assistant 最好只输出 JSON

因为当前任务更像结构化监测，不是开放式问答。只输出 JSON 的好处有：

- 更容易解析
- 更适合下游系统接入
- 更容易做一致性评估
- 更适合小样本微调

### 11.4 为什么测试集很小

当前仓库中的数据规模本身不大，更适合快速实验和任务定义收敛。后续如果要提升泛化能力，建议重点补充：

- 合规样本
- 证据不足样本
- 容易误判的坡面样本
- 真正存在临边缺失的强风险样本

## 12. 建议的后续迭代方向

如果你接下来还会继续扩展项目，建议按优先级考虑：

1. 扩充高质量 `uncertain` 样本
2. 增加“纯边坡但不违规”的强负样本
3. 单独评估 `scene_type` 准确率
4. 为 `violation_type` 做更细的混淆分析
5. 在 `chat.py` 基础上补充更稳定的框选定位评估
6. 将输出 JSON 进一步标准化到下游业务接口格式

## 13. 一句话总结

这个仓库已经不是单纯“跑一个 Qwen2.5-VL 微调脚本”，而是一个面向施工安全监测的小型完整实验链路：从自动打标、人工复核、数据清洗，到 QLoRA 微调、评估、交互式质检和违规区域可视化，核心目标是让模型在边坡/基坑场景下输出稳定、可信、可落地的结构化安全结论。
