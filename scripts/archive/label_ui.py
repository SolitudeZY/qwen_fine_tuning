import gradio as gr
import json
import os
import random

# ============ 配置 ============
INPUT_JSON = "/home/fs-ai/llama-qwen/outputs/api_annotations_cot.json"
OUTPUT_JSONL = "/home/fs-ai/llama-qwen/data/train_balanced.jsonl"
OUTPUT_VAL_JSONL = "/home/fs-ai/llama-qwen/data/val_balanced.jsonl"

# 全局状态
data = []
current_idx = 0
stats = {"checked": 0, "violation": 0, "normal": 0}

def load_data():
    global data
    if not os.path.exists(INPUT_JSON):
        return []
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 随机打乱，避免连着看全是同一个场景
    random.seed(42)
    random.shuffle(data)

    # 初始化人工检查结果
    for item in data:
        item["human_checked"] = False
        item["human_violation"] = None
    return data

def get_current_item():
    global data, current_idx
    if current_idx >= len(data):
        return None, "所有数据已检查完毕！", "", "", "进度: 已完成"

    item = data[current_idx]
    img_path = item["image_path"]

    info = f"**原始标签**: {', '.join(item.get('original_labels', []))}"

    # 兼容 GPT 结构化输出或原本的推理输出
    reasoning = item.get('vlm_reasoning', '')
    json_data = item.get('vlm_json', {})

    if not reasoning and isinstance(json_data, dict):
        # 尝试将结构化输出拼装成自然语言
        scene = json_data.get('scene_type', '未知')
        obs = "\\n".join(f"- {o}" for o in json_data.get('key_observations', []))
        risks = "\\n".join(f"- {r}" for r in json_data.get('risk_points', []))
        result = json_data.get('monitoring_result', '')

        reasoning = f"【场景类型】: {scene}\\n【关键观察】:\\n{obs}\\n【风险点】:\\n{risks}\\n【结论】: {result}"

    try:
        json_str = json.dumps(json_data, ensure_ascii=False, indent=2)
    except:
        json_str = str(json_data)

    return img_path, info, reasoning, json_str, f"进度: {stats['checked']}/{len(data)} (违规: {stats['violation']}, 合规: {stats['normal']})"

def save_and_next(is_violation, edited_reasoning, edited_json_str):
    global data, current_idx, stats
    if current_idx >= len(data):
        return get_current_item()

    item = data[current_idx]

    # 解析编辑后的 JSON
    try:
        new_json = json.loads(edited_json_str)
        # 强制更新 violation_detected 字段以匹配按钮意图
        new_json["violation_detected"] = is_violation
    except Exception as e:
        print(f"JSON 解析失败，保持原样: {e}")
        new_json = item.get("vlm_json", {})
        if isinstance(new_json, dict):
            new_json["violation_detected"] = is_violation

    # 如果之前没检查过，更新统计
    if not item["human_checked"]:
        stats["checked"] += 1
        if is_violation:
            stats["violation"] += 1
        else:
            stats["normal"] += 1

    # 更新当前项
    item["human_checked"] = True
    item["human_violation"] = is_violation
    item["vlm_reasoning"] = edited_reasoning
    item["vlm_json"] = new_json

    # 自动跳到下一张
    current_idx += 1
    return get_current_item()

def mark_violation(edited_reasoning, edited_json_str):
    return save_and_next(True, edited_reasoning, edited_json_str)

def mark_normal(edited_reasoning, edited_json_str):
    return save_and_next(False, edited_reasoning, edited_json_str)

def skip_item():
    global current_idx
    if current_idx < len(data):
        current_idx += 1
    return get_current_item()

def prev_item():
    global current_idx
    if current_idx > 0:
        current_idx -= 1
    return get_current_item()

def get_stats():
    global data
    violation_count = sum(1 for d in data if d.get("human_checked") and d.get("human_violation") is True)
    normal_count = sum(1 for d in data if d.get("human_checked") and d.get("human_violation") is False)
    skipped_count = sum(1 for d in data if d.get("human_checked") is False) # 未检查或被跳过
    total = len(data)
    return f"📊 **当前标注统计**: 总图数 {total} 张 | ❌ 已确认为违规: **{violation_count}** 张 | ✅ 已确认为合规: **{normal_count}** 张 | ⏭️ 未确认/跳过: **{skipped_count}** 张"

def export_dataset(balance_data=True):
    """导出数据集转换为 ms-swift 格式"""
    global data

    violations = [d for d in data if d["human_checked"] and d["human_violation"]]
    normals = [d for d in data if d["human_checked"] and not d["human_violation"]]

    if balance_data:
        # 平衡数据逻辑
        min_len = min(len(violations), len(normals))
        if min_len == 0:
            return f"❌ 数据不足以平衡！你需要至少标出 1 张违规和 1 张合规。目前违规:{len(violations)}, 合规:{len(normals)}。"

        random.seed(42)
        selected_violations = random.sample(violations, min_len)
        selected_normals = random.sample(normals, min_len)
        export_data = selected_violations + selected_normals
        mode_text = f"⚖️ **平衡导出模式** (1:1 正负样本)"
    else:
        # 全量导出逻辑
        export_data = violations + normals
        mode_text = f"📦 **全量导出模式** (包含所有已确认的正负样本)"

    if len(export_data) == 0:
        return "❌ 导出失败：没有已确认的标注数据。"

    random.shuffle(export_data)

    split_idx = int(len(export_data) * 0.9)
    train_data = export_data[:split_idx]
    val_data = export_data[split_idx:]

    def save_swift_format(dataset, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for item in dataset:
                query = "请对这张施工现场图片执行边坡/基坑安全监测，识别场景类型，提取关键观察，判断是否存在临边高坠风险及防护缺陷，并输出安全推理与 JSON 结果。"

                # 如果推理是空的（GPT 纯JSON版本），我们从 JSON 中提取并重组为结构化推理
                reasoning = item.get("vlm_reasoning", "")
                json_data = item.get("vlm_json", {})

                if not reasoning and isinstance(json_data, dict):
                    scene = json_data.get('scene_type', '未知')
                    obs = "\n".join(f"- {o}" for o in json_data.get('key_observations', []))
                    risks = "\n".join(f"- {r}" for r in json_data.get('risk_points', []))
                    result = json_data.get('monitoring_result', '')
                    reasoning = f"【场景类型】: {scene}\n【关键观察】:\n{obs}\n【风险点】:\n{risks}\n【结论】: {result}"

                json_str = json.dumps(json_data, ensure_ascii=False)
                response = f"{reasoning}\n\n```json\n{json_str}\n```"

                # >>> 修复系统 Prompt: 与最新 API 脚本的规则保持一致，不要再带偏模型 <<<
                sys_prompt = (
                    "你是一名严谨的边坡与基坑施工安全监测专家，负责基于施工现场图片输出可落地的监测结论。\n"
                    "你的任务不是泛泛描述图片，而是围绕高坠风险和临边防护进行监测判定，并输出结构化的推理过程和JSON结果。\n"
                    "【场景识别要求】：\n"
                    "- 基坑：向下开挖形成的坑槽，坑边通常需要防护隔离。\n"
                    "- 边坡：倾斜土坡、岩坡本身通常不是必须全封闭围挡的对象。但坡顶作业平台、临空边若存在坠落风险，仍需防护。\n"
                    "- 如果只是可见大面积施工坡面，不要仅凭“是边坡”就判为违规。\n"
                    "【输出要求】：\n"
                    "1. 首先，你需要详细描述图片中危险区域的状态，并进行安全推理。\n"
                    "2. 然后，你必须在回答的最后严格以 JSON 格式输出判定结果。\n"
                    "JSON 字段必须包含：violation_detected(布尔), violation_type(字符串), severity(字符串), suggestion(字符串)。"
                )

                swift_item = {
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": f"<image>{query}"},
                        {"role": "assistant", "content": response}
                    ],
                    "images": [item["image_path"]]
                }
                f.write(json.dumps(swift_item, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    save_swift_format(train_data, OUTPUT_JSONL)
    save_swift_format(val_data, OUTPUT_VAL_JSONL)

    v_count = sum(1 for d in export_data if d["vlm_json"]["violation_detected"])
    n_count = len(export_data) - v_count
    return f"✅ 导出成功！\n{mode_text}\n总导出量: {len(export_data)} (违规: {v_count}, 合规: {n_count})\n训练集: {len(train_data)} 条 -> {OUTPUT_JSONL}\n验证集: {len(val_data)} 条 -> {OUTPUT_VAL_JSONL}"

def export_balanced():
    return export_dataset(balance_data=True)

def export_all():
    return export_dataset(balance_data=False)

def build_ui():
    load_data()

    with gr.Blocks(title="边防护违规数据精细打标") as app:
        gr.Markdown("## 🚧 边防护违规数据人工筛选与修正工具")
        gr.Markdown("你可以**直接修改**大模型的推理文本和 JSON，修改后点击【确认并下一张】会自动保存你的修改内容。")

        # 统计看板
        stats_view = gr.Markdown(get_stats())
        btn_refresh_stats = gr.Button("🔄 刷新统计面板", size="sm")

        with gr.Row():
            with gr.Column(scale=1):
                # 增加 height 参数限制默认高度，但开启可缩放/预览功能
                img_view = gr.Image(type="filepath", label="现场照片 (点击右上角 🔍 按钮可放大查看细节)", height=700, interactive=False)

            with gr.Column(scale=1):
                progress_view = gr.Markdown("进度: 加载中...")
                info_view = gr.Markdown(label="基础信息")

                # 修改为可编辑的文本框
                reasoning_view = gr.Textbox(label="【可修改】大模型推理过程 (CoT)", lines=5)
                json_view = gr.Textbox(label="【可修改】最终判定 JSON", lines=8)

                with gr.Row():
                    btn_prev = gr.Button("⬅️ 上一张")
                    btn_skip = gr.Button("⏭️ 跳过 (不确定)")

                gr.Markdown("### 保存当前修改并打标 (自动下一张)")
                with gr.Row():
                    btn_normal = gr.Button("✅ 确认保存 -> 【合规】", variant="primary")
                    btn_violation = gr.Button("❌ 确认保存 -> 【违规】", variant="stop")

                gr.Markdown("---")
                with gr.Row():
                    btn_export_balance = gr.Button("📦 导出平衡数据集 (推荐, 1:1采样)", variant="primary")
                    btn_export_all = gr.Button("📂 导出全量确认数据 (不截断)", variant="secondary")
                export_result = gr.Markdown()

        # 事件绑定
        img_val, info_val, reasoning_val, json_val, prog_val = get_current_item()

        btn_refresh_stats.click(get_stats, outputs=[stats_view])

        app.load(lambda: (img_val, info_val, reasoning_val, json_val, prog_val),
                 outputs=[img_view, info_view, reasoning_view, json_view, progress_view])
        app.load(get_stats, outputs=[stats_view])

        btn_normal.click(mark_normal,
                         inputs=[reasoning_view, json_view],
                         outputs=[img_view, info_view, reasoning_view, json_view, progress_view]).then(
                         get_stats, outputs=[stats_view])

        btn_violation.click(mark_violation,
                            inputs=[reasoning_view, json_view],
                            outputs=[img_view, info_view, reasoning_view, json_view, progress_view]).then(
                            get_stats, outputs=[stats_view])

        btn_skip.click(skip_item, outputs=[img_view, info_view, reasoning_view, json_view, progress_view])
        btn_prev.click(prev_item, outputs=[img_view, info_view, reasoning_view, json_view, progress_view])

        btn_export_balance.click(export_balanced, outputs=[export_result])
        btn_export_all.click(export_all, outputs=[export_result])

    return app

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860, help="Gradio Web UI 运行的端口")
    args = parser.parse_args()

    app = build_ui()
    try:
        # Gradio 5.x 出于安全原因不允许读取当前工作目录外的文件
        # 我们必须把存放图片的 dataset 目录加入 allowed_paths
        allowed_dir = "/home/fs-ai/llama-qwen/dataset"
        app.launch(server_name="0.0.0.0", server_port=args.port, share=False, allowed_paths=[allowed_dir])
    except OSError as e:
        if "Cannot find empty port" in str(e):
            print(f"\n❌ 端口 {args.port} 已被占用！")
            print("可能是之前的 label_ui.py 还在后台运行。")
            print("请尝试更换端口运行，例如：\n  python scripts/label_ui.py --port 7861\n")
            print("或者你可以杀掉占用该端口的旧进程：\n  pkill -f label_ui.py\n")
        else:
            raise e
