# 环水保（水体污染检测）项目 PRD 及执行计划

## 1. 项目背景与目标
基于 CVAT 平台已标注好的“环水保”数据集（图像级标签：污染的水/正常的水），从头搭建一套处理流。使用 Qwen-VL 模型进行单阶段微调，实现输入一张图片，模型能够自动判断水体状态并给出一句话描述。

## 2. 技术选型与确认细节
通过沟通确认了以下关键技术方案：
- **JSON 输出格式**：模型最终输出全局分类（`normal` / `polluted`）以及一句话描述，无需输出 BBox 检测框。
- **API 标注方案**：调用 DashScope 的 `qwen-vl-plus`（Qwen3.6-plus），通过读取环境变量 `DASHSCOPE_API_KEY` 来获取权限。结合 CVAT 给定的标签和图片，让 API 生成对应的一句话描述，并保存为 JSON 训练格式。
- **MQTT 通信设计**：在推理脚本中，默认监听 `water/detect/request` 主题以获取图片，完成检测后将结果发布到 `water/detect/result`。并暴露 `--topic-req` 和 `--topic-res` 参数供后续修改。

## 3. 核心脚本设计
所有脚本均存放在 `/home/fs-ai/llama-qwen/scripts/qwen3.5-water` 目录下：

### 3.1 `01_api_auto_annotate.py` (API自动标注脚本)
- **输入**：`/home/fs-ai/llama-qwen/data/环水保_labeled/annotations.xml` 或 CVAT 导出的原始目录。
- **逻辑**：解析 XML 获取每张图的 `water_status`。调用 DashScope API，传入提示词“已知该水体状态为[正常/污染]，请结合图片给出一句话简短描述原因”。
- **输出**：生成 `annotate_cache.json` 供审查工具使用。

### 3.2 `02_review_ui.py` (人工审查工具)
- **逻辑**：复用并改造之前的 Flask UI 代码。移除 BBox 的拖拽绘制功能，仅保留图片展示、状态切换（合规/违规）以及一句话描述（Suggestion/Description）的文本框修改功能。
- **输出**：修改并保存审查后的结果。支持导出为最终的单阶段训练 JSONL 文件。

### 3.3 `03_train_stage1.sh` (单阶段训练脚本)
- **逻辑**：基于 swift 框架，直接读取审查导出的 JSONL 文件，使用 `qwen3-vl` 基础模型进行全量/LoRA 微调。
- **输出**：包含最佳 Checkpoint 的微调权重目录。

### 3.4 `04_mqtt_inference.py` (测试和推理脚本)
- **逻辑**：通过 `paho-mqtt` 连接 MQTT Broker，监听请求。收到图片 Base64 或 URL 后，利用微调后的 Qwen-VL 模型进行推理，生成 `water_status` 和 `description`，并按照约定格式组装 JSON 报告发送到结果 Topic。

## 4. 执行计划
1. 编写 PRD 文档（当前）。
2. 编写 `01_api_auto_annotate.py`。
3. 编写 `02_review_ui.py`。
4. 编写 `03_train_stage1.sh`。
5. 编写 `04_mqtt_inference.py`。
