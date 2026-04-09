"""
边防护违规识别 - 交互式对话测试脚本
支持：
  1. 输入图片路径进行单图检测
  2. 交互式多轮对话
  3. 对比测试（原始模型 vs 微调模型）
  4. Qwen2.5-VL 和 Qwen3-VL 模型自动适配
"""

import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
import torch
from model_utils import load_vlm, infer_vlm

# ============ 配置 ============
# 默认模型路径，可通过 --model_path 覆盖
MODEL_PATH = "/home/fs-ai/llama-qwen/models/Qwen/Qwen3-VL-2B-Instruct"
# 使用最佳 checkpoint

SYSTEM_PROMPT = (
    "你是一名严谨的边坡与基坑施工安全监测专家，负责基于施工现场图片输出可落地的监测结论。\n"
    "你的任务不是泛泛描述图片，而是围绕高坠风险和临边防护进行监测判定，并输出结构化的推理过程和JSON结果。\n"
    "【场景识别要求】：\n"
    "- 基坑：向下开挖形成的坑槽，坑边通常需要防护隔离。\n"
    "- 边坡：倾斜土坡、岩坡本身通常不是必须全封闭围挡的对象。但坡顶作业平台、临空边若存在坠落风险，仍需防护。\n"
    "- 如果只是可见大面积施工坡面，不要仅凭'是边坡'或'是基坑'或就判为违规。\n"
    "【输出要求】：\n"
    "1. 首先，你需要详细描述图片中危险区域的状态，并进行安全推理。\n"
    "2. 然后，你必须在回答的最后严格以 JSON 格式输出判定结果。\n"
    "JSON 字段必须包含：violation_detected(布尔), violation_type(字符串), severity(字符串), suggestion(字符串)。\n"
    "3. 如果检测到违规(violation_detected=true)，你必须额外输出 violation_boxes 字段，它是一个数组，"
    "每个元素包含：label(违规简述), bbox(四个整数的数组，格式为[x_min, y_min, x_max, y_max]，"
    "坐标为相对于图片宽高的千分比坐标，范围0-1000)。"
    "例如：\"violation_boxes\": [{\"label\": \"围栏缺口\", \"bbox\": [120, 300, 450, 680]}]\n"
    "4. 如果合规(violation_detected=false)，violation_boxes 为空数组 []。"
)

DEFAULT_QUERY = "请对这张施工现场图片执行边坡/基坑安全监测，识别场景类型，提取关键观察，判断是否存在临边高坠风险及防护缺陷，并输出安全推理与 JSON 结果。如果存在违规，请用 violation_boxes 标出违规区域的位置坐标。"


# ============ 可视化配置 ============
BOX_COLOR_BGR = (0, 0, 255)       # 红色 (BGR for cv2)
BOX_COLOR_RGB = (255, 0, 0)       # 红色 (RGB for PIL)
BOX_THICKNESS = 3
LABEL_BG_COLOR = (200, 0, 0)     # 深红背景 (RGB)
LABEL_TEXT_COLOR = (255, 255, 255)  # 白色文字

# 中文字体路径
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CJK_FONT_SIZE = 22


def draw_violation_boxes(image_path, boxes, output_path=None):
    """在图片上绘制违规区域的 bounding box 并保存（支持中文标签）"""
    pil_img = PILImage.open(image_path).convert("RGB")
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)

    # 加载中文字体
    try:
        font = ImageFont.truetype(CJK_FONT_PATH, CJK_FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for box_info in boxes:
        label = box_info.get("label", "违规区域")
        bbox = box_info.get("bbox", [])
        if len(bbox) != 4:
            continue

        # 千分比坐标 -> 像素坐标
        x_min = int(bbox[0] / 1000 * w)
        y_min = int(bbox[1] / 1000 * h)
        x_max = int(bbox[2] / 1000 * w)
        y_max = int(bbox[3] / 1000 * h)

        # 裁剪到图片范围内
        x_min = max(0, min(x_min, w - 1))
        y_min = max(0, min(y_min, h - 1))
        x_max = max(0, min(x_max, w - 1))
        y_max = max(0, min(y_max, h - 1))

        # 画矩形框
        draw.rectangle([x_min, y_min, x_max, y_max], outline=BOX_COLOR_RGB, width=BOX_THICKNESS)

        # 计算标签尺寸
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        # 标签放在框的上方
        label_x = x_min
        label_y = max(y_min - th - 8, 0)
        draw.rectangle([label_x, label_y, label_x + tw + 8, label_y + th + 6], fill=LABEL_BG_COLOR)
        draw.text((label_x + 4, label_y + 2), label, fill=LABEL_TEXT_COLOR, font=font)

    # 确定输出路径
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_annotated{ext}"

    pil_img.save(output_path, quality=95)
    print(f"  标注图片已保存: {output_path}")
    return output_path


GROUNDING_PROMPT = (
    "请观察这张施工现场图片，标出存在安全隐患的区域。"
    "包括：围栏防护不完整的区域、临边缺少防护的区域、存在高坠风险的区域等。"
    "请直接输出JSON数组，每个元素包含 label(问题简述) 和 bbox([x_min, y_min, x_max, y_max]，"
    "坐标为相对于图片宽高的千分比，范围0-1000)。"
    "只输出JSON数组，不要输出其他内容。"
    '例如: [{"label": "防护缺失区域", "bbox": [120, 300, 450, 680]}]'
)


def _normalize_boxes(boxes, image_path):
    """自动检测坐标是像素坐标还是千分比坐标，统一转为千分比(0-1000)"""
    from PIL import Image as PILImage
    try:
        img = PILImage.open(image_path)
        w, h = img.size
    except Exception:
        return boxes

    for box in boxes:
        bbox = box.get("bbox", [])
        if len(bbox) != 4:
            continue
        if any(v > 1000 for v in bbox):
            box["bbox"] = [
                int(bbox[0] / w * 1000),
                int(bbox[1] / h * 1000),
                int(bbox[2] / w * 1000),
                int(bbox[3] / h * 1000),
            ]
    return boxes


def locate_violations(model, processor, model_family, image_path):
    """第二阶段：让模型定位违规区域，输出 bbox"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": GROUNDING_PROMPT},
            ],
        },
    ]

    response = infer_vlm(model, processor, model_family, messages, max_new_tokens=512)

    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    # 修复被截断的 JSON 数组
    if cleaned.startswith("[") and not cleaned.rstrip().endswith("]"):
        last_brace = cleaned.rfind("}")
        if last_brace != -1:
            cleaned = cleaned[:last_brace + 1] + "]"

    try:
        boxes = json.loads(cleaned)
        if isinstance(boxes, list):
            return _normalize_boxes(boxes, image_path)
    except json.JSONDecodeError:
        arr_start = cleaned.find("[")
        arr_end = cleaned.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            try:
                boxes = json.loads(cleaned[arr_start:arr_end + 1])
                if isinstance(boxes, list):
                    return _normalize_boxes(boxes, image_path)
            except json.JSONDecodeError:
                pass

    print(f"  定位阶段未能解析出 bbox，原始输出:\n  {response[:300]}")
    return []


def load_model(use_lora=True):
    """加载模型（自动适配 Qwen2.5-VL / Qwen3-VL）"""
    lora = LORA_PATH if (use_lora and LORA_PATH) else None
    model, processor, family = load_vlm(MODEL_PATH, lora_path=lora)
    return model, processor, family


def chat(model, processor, model_family, image_path, query=None):
    """单次对话推理"""
    if query is None:
        query = DEFAULT_QUERY

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": query},
            ],
        },
    ]

    start = time.time()
    response = infer_vlm(model, processor, model_family, messages, max_new_tokens=768)
    elapsed = time.time() - start

    return response, elapsed


import re

def _extract_json_object(response: str):
    """从回复中提取JSON对象，并兼容模型将 JSON 切散或格式不标准的情况"""
    # 1. 尝试匹配 markdown 的 json 代码块
    json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
    if json_block_match:
        candidate = json_block_match.group(1)
        try:
            return json.loads(candidate), candidate, response.replace(json_block_match.group(0), "").strip()
        except json.JSONDecodeError:
            pass

    # 2. 尝试从后往前寻找能够成功 parse 的完整 JSON 对象
    end = response.rfind("}")
    parsed_json = None
    json_candidate = None
    json_start_index = -1
    
    if end != -1:
        start = response.rfind("{", 0, end)
        while start != -1:
            candidate = response[start:end+1]
            try:
                # 兼容模型偶尔使用单引号或格式不规范的情况
                candidate_fixed = candidate.replace("'", '"')
                parsed_json = json.loads(candidate_fixed)
                json_candidate = candidate_fixed
                json_start_index = start
                break # 找到了最外层或者说第一个能解开的
            except json.JSONDecodeError:
                start = response.rfind("{", 0, start)
                
    # 如果找到了 JSON，并且它包含了关键的状态字段
    if parsed_json is not None and ("violation_detected" in parsed_json or "violation_type" in parsed_json):
        # 检查是否缺失 boxes
        if parsed_json.get("violation_detected") is True and not parsed_json.get("violation_boxes"):
            # 尝试在整个文本中搜索散落的 boxes，并补充进去
            box_objects = re.findall(r'\{\s*["\']label["\']\s*:\s*["\'](.*?)["\']\s*,\s*["\']bbox["\']\s*:\s*(\[[^\]]+\])\s*\}', response)
            if box_objects:
                parsed_boxes = []
                for label, bbox_str in box_objects:
                    try:
                        bbox = json.loads(bbox_str)
                        parsed_boxes.append({"label": label, "bbox": bbox})
                    except:
                        pass
                if parsed_boxes:
                    parsed_json["violation_boxes"] = parsed_boxes
        
        return parsed_json, json_candidate, response[:json_start_index].strip() if json_start_index > 0 else response.strip()
                
    # 3. 兜底策略：如果模型没有输出标准 JSON，或者将 violation_boxes 打印在了 JSON 外面
    # 尝试手动通过正则提取关键字段构建 JSON
    fallback_json = {}
    
    # 提取 violation_detected
    if "violation_detected: true" in response.lower() or '"violation_detected": true' in response.lower():
        fallback_json["violation_detected"] = True
    elif "violation_detected: false" in response.lower() or '"violation_detected": false' in response.lower():
        fallback_json["violation_detected"] = False
        
    # 如果找到了布尔值，说明模型意图输出结构化数据，继续提取其他字段
    if "violation_detected" in fallback_json:
        # 提取类型、严重程度、建议等
        type_match = re.search(r'violation_type:\s*([^\n]+)', response)
        if type_match: fallback_json["violation_type"] = type_match.group(1).strip().strip('",')
        
        severity_match = re.search(r'severity:\s*([^\n]+)', response)
        if severity_match: fallback_json["severity"] = severity_match.group(1).strip().strip('",')
        
        suggestion_match = re.search(r'suggestion:\s*([^\n]+)', response)
        if suggestion_match: fallback_json["suggestion"] = suggestion_match.group(1).strip().strip('",')
        
        # 尝试提取独立的 violation_boxes 数组
        # 模型可能输出: violation_boxes: [{"label":...}] [{"label":...}] 这种多个列表散落的情况
        # 或者输出正常的: violation_boxes: [{"label":...}, {"label":...}]
        if "violation_boxes" not in fallback_json:
            # 找到所有长得像 {"label": "...", "bbox": [...]} 的对象
            box_objects = re.findall(r'\{\s*["\']label["\']\s*:\s*["\'](.*?)["\']\s*,\s*["\']bbox["\']\s*:\s*(\[[^\]]+\])\s*\}', response)
            if box_objects:
                parsed_boxes = []
                for label, bbox_str in box_objects:
                    try:
                        bbox = json.loads(bbox_str)
                        parsed_boxes.append({"label": label, "bbox": bbox})
                    except:
                        pass
                if parsed_boxes:
                    fallback_json["violation_boxes"] = parsed_boxes
                    
        # 还有一种情况：模型输出了两段 JSON，一段是状态，一段是 boxes (整块数组)
        if "violation_boxes" not in fallback_json:
            all_arrays = re.findall(r'\[\s*\{.*?"label".*?"bbox".*?\}\s*\]', response, re.DOTALL)
            if all_arrays:
                try:
                    fallback_json["violation_boxes"] = json.loads(all_arrays[-1])
                except json.JSONDecodeError:
                    pass
                    
        return fallback_json, json.dumps(fallback_json, ensure_ascii=False), response.strip()

    return None, None, response.strip()


def print_result(response, elapsed):
    """格式化输出结果，返回解析后的 JSON dict"""
    print(f"\n{'='*60}")
    print(f"推理耗时: {elapsed:.2f}s")
    print(f"{'='*60}")

    result, json_str, reasoning = _extract_json_object(response)
    if reasoning and json_str:
        print("【推理过程】:")
        print(reasoning)
        print("-" * 60)

    try:
        if result is None:
            raise json.JSONDecodeError("No valid JSON object found", response, 0)

        violation = result.get("violation_detected", False)  # 默认False防报错
        vtype = result.get("violation_type", "N/A")
        severity = result.get("severity", "N/A")
        suggestion = result.get("suggestion", "N/A")

        status = "!! 违规 !!" if violation else "合规"
        print(f"  状态: {status}")
        print(f"  类型: {vtype}")
        print(f"  严重程度: {severity}")
        print(f"  建议: {suggestion}")

        # 打印 violation_boxes
        boxes = result.get("violation_boxes", [])
        if boxes:
            print(f"  违规区域数量: {len(boxes)}")
            for i, b in enumerate(boxes):
                print(f"    [{i+1}] {b.get('label', '?')} -> bbox: {b.get('bbox', [])}")

        # 打印其他附加字段
        skip_keys = {"violation_detected", "violation_type", "severity", "suggestion", "violation_boxes"}
        for k, v in result.items():
            if k not in skip_keys:
                print(f"  {k}: {v}")

        print(f"{'='*60}\n")

        if isinstance(violation, str) and violation.lower() == "true":
            result["violation_detected"] = True
        elif isinstance(violation, str) and violation.lower() == "false":
            result["violation_detected"] = False

        return result
    except json.JSONDecodeError:
        print(f"  [Error] 无法解析输出的 JSON，原始输出:\n{response}")
        print(f"{'='*60}\n")
        return None


def interactive_mode(model, processor, model_family, visualize=False, output_dir=None):
    """交互式对话模式"""
    print("\n" + "="*60)
    print("  边防护违规识别系统 - 交互式测试")
    family_label = "Qwen3-VL" if model_family == "qwen3-vl" else "Qwen2.5-VL"
    print(f"  模型: {family_label} ({os.path.basename(MODEL_PATH)})")
    if visualize:
        print(f"  可视化模式已开启，标注图保存至: {output_dir}")
    print("="*60)
    print("  使用方法:")
    print("    输入图片路径 → 使用默认问题检测")
    print("    输入 q 或 quit → 退出")
    print("    输入 help → 显示帮助")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("请输入图片路径 (或 q 退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("q", "quit", "exit"):
            print("退出。")
            break
        if user_input.lower() == "help":
            print("  直接输入图片路径即可进行边防护检测")
            print("  示例: /home/fs-ai/llama-qwen/dataset/1-100/DJI_0001.JPG")
            continue

        # 去除可能的引号
        image_path = user_input.strip("'\"")

        if not os.path.exists(image_path):
            print(f"  文件不存在: {image_path}")
            continue

        # 可选自定义问题
        custom_query = input("自定义提问 (回车使用默认): ").strip()
        query = custom_query if custom_query else None

        print(f"\n正在分析图片: {image_path}")
        response, elapsed = chat(model, processor, model_family, image_path, query)
        result = print_result(response, elapsed)

        # 增加宽容度，如果没解析出 result，但原始回复包含特定关键字，也可以触发定位
        is_violation = False
        boxes = []
        if result and result.get("violation_detected"):
            is_violation = True
            boxes = result.get("violation_boxes", [])
        elif not result:
            # 如果没输出 JSON 或是解析失败，从纯文本中寻找线索
            lower_res = response.lower()
            if "violation_detected: true" in lower_res or "存在安全隐患" in response or "高坠风险" in response:
                is_violation = True

        if visualize and is_violation:
            if not boxes:
                print("  警告: 未在初始回答中检测到 violation_boxes，正在尝试从备用逻辑获取或进行第二阶段定位...")
                # 兼容旧版本模型：如果解析出来没有框，自动进行一次定位兜底
                boxes = locate_violations(model, processor, model_family, image_path)
            else:
                boxes = _normalize_boxes(boxes, image_path)

            if boxes:
                out_name = os.path.basename(image_path).rsplit(".", 1)[0] + "_annotated.jpg"
                out_path = os.path.join(output_dir, out_name)
                draw_violation_boxes(image_path, boxes, out_path)
            else:
                print("  未能定位到具体违规区域，跳过可视化。")


def batch_test(model, processor, model_family, test_jsonl):
    """批量测试模式：在测试集上运行"""
    print(f"\n正在加载测试数据: {test_jsonl}")
    samples = []
    with open(test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"共 {len(samples)} 条测试数据\n")

    correct = 0
    total = 0

    for i, sample in enumerate(samples):
        image_path = sample.get("images", [None])[0]
        if not image_path or not os.path.exists(image_path):
            print(f"[{i+1}] 跳过 - 图片不存在")
            continue

        expected_raw = sample["messages"][-1]["content"]
        expected_json_str = expected_raw
        expected_json_start = expected_raw.rfind("{")
        expected_json_end = expected_raw.rfind("}")
        if expected_json_start != -1 and expected_json_end != -1 and expected_json_end > expected_json_start:
            expected_json_str = expected_raw[expected_json_start:expected_json_end+1]

        try:
            expected = json.loads(expected_json_str)
        except json.JSONDecodeError:
            continue

        response, elapsed = chat(model, processor, model_family, image_path)

        predicted_json_str = response
        predicted_json_start = response.rfind("{")
        predicted_json_end = response.rfind("}")
        if predicted_json_start != -1 and predicted_json_end != -1 and predicted_json_end > predicted_json_start:
            predicted_json_str = response[predicted_json_start:predicted_json_end+1]

        try:
            predicted = json.loads(predicted_json_str)
        except json.JSONDecodeError:
            predicted = {}

        exp_v = expected.get("violation_detected")
        pred_v = predicted.get("violation_detected")
        match = exp_v == pred_v
        if match:
            correct += 1
        total += 1

        status = "OK" if match else "WRONG"
        print(f"[{i+1}/{len(samples)}] {status} | 期望: {exp_v} | 预测: {pred_v} | 耗时: {elapsed:.1f}s")

    if total > 0:
        print(f"\n准确率: {correct}/{total} = {correct/total:.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="边防护违规识别 - 测试对话")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径（覆盖默认值）")
    parser.add_argument("--no-lora", action="store_true", help="不加载 LoRA，使用原始模型对比")
    parser.add_argument("--image", type=str, help="直接测试单张图片")
    parser.add_argument("--batch-test", action="store_true", help="在测试集上批量测试")
    parser.add_argument("--test-data", type=str, default="/home/fs-ai/llama-qwen/data/test.jsonl")
    parser.add_argument("--query", type=str, default=None, help="自定义提问")
    parser.add_argument("--lora_path",type=str,default=None,help="指定模型LoRA参数权重文件路径")
    parser.add_argument("--visualize", action="store_true", help="检测到违规时自动画框并保存标注图片")
    parser.add_argument("--output-dir", type=str, default="/home/fs-ai/llama-qwen/outputs/visualized", help="标注图片保存目录")

    args = parser.parse_args()
    LORA_PATH = args.lora_path
    if args.model_path:
        MODEL_PATH = args.model_path
    use_lora = not args.no_lora
    
    # 校验 LoRA 路径
    if use_lora:
        if not LORA_PATH:
            print("[Error] 未指定 --lora_path 且未使用 --no-lora。请提供有效的 LoRA 路径，或者使用 --no-lora 来测试原始基座模型。")
            sys.exit(1)
        if not os.path.exists(LORA_PATH):
            print(f"[Error] 指定的 LoRA 路径不存在: {LORA_PATH}")
            print("请检查路径是否正确，或者使用 --no-lora 来测试原始基座模型。")
            sys.exit(1)
            
    model, processor, model_family = load_model(use_lora=use_lora)

    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.image:
        # 单图测试
        response, elapsed = chat(model, processor, model_family, args.image, args.query)
        result = print_result(response, elapsed)
        if args.visualize and result and result.get("violation_detected"):
            boxes = result.get("violation_boxes", [])
            if not boxes:
                print("  警告: 未在初始回答中检测到 violation_boxes，正在尝试从备用逻辑获取或进行第二阶段定位...")
                # 兼容旧版本模型：如果解析出来没有框，自动进行一次定位兜底
                boxes = locate_violations(model, processor, model_family, args.image)
            else:
                boxes = _normalize_boxes(boxes, args.image)
            if boxes:
                out_name = os.path.basename(args.image).rsplit(".", 1)[0] + "_annotated.jpg"
                out_path = os.path.join(args.output_dir, out_name)
                draw_violation_boxes(args.image, boxes, out_path)
            else:
                print("  未能定位到具体违规区域，跳过可视化。")
    elif args.batch_test:
        # 批量测试
        batch_test(model, processor, model_family, args.test_data)
    else:
        # 交互模式
        interactive_mode(model, processor, model_family, visualize=args.visualize, output_dir=args.output_dir)
