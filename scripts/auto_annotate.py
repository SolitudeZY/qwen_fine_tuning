import os
import json
import glob
import time
from tqdm import tqdm

import torch
from qwen_vl_utils import process_vision_info
from model_utils import load_vlm, infer_vlm

# ============ 核心配置 ============
MODEL_PATH = "/home/fs-ai/llama-qwen/models/Qwen/Qwen3-VL-2B-Instruct"
DATASET_DIR = "/home/fs-ai/llama-qwen/dataset"
OUTPUT_JSON = "/home/fs-ai/llama-qwen/outputs/auto_annotations_cot.json"

def load_model():
    """加载 4-bit 量化的 VLM 模型（自动适配 Qwen2.5-VL / Qwen3-VL）"""
    model, processor, family = load_vlm(MODEL_PATH)
    return model, processor, family

def parse_labelme_to_qwen_boxes(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth", 1)
    img_h = data.get("imageHeight", 1)
    shapes = data.get("shapes", [])

    qwen_boxes_str = ""
    extracted_labels = []

    for shape in shapes:
        label = shape["label"]
        extracted_labels.append(label)

        points = shape["points"]
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        # 裁剪并归一化 (0-1000)
        x_min = max(0, min(xs))
        y_min = max(0, min(ys))
        x_max = min(img_w, max(xs))
        y_max = min(img_h, max(ys))

        ymin_norm = int(y_min / img_h * 1000)
        xmin_norm = int(x_min / img_w * 1000)
        ymax_norm = int(y_max / img_h * 1000)
        xmax_norm = int(x_max / img_w * 1000)

        box_str = f"<box_start>({ymin_norm},{xmin_norm}),({ymax_norm},{xmax_norm})<box_end>{label}\n"
        qwen_boxes_str += box_str

    img_name = data.get("imagePath", "")
    img_path = os.path.abspath(os.path.join(os.path.dirname(json_path), img_name))

    return {
        "image_path": img_path,
        "json_path": os.path.abspath(json_path),
        "qwen_boxes_prompt": qwen_boxes_str.strip(),
        "labels": list(set(extracted_labels)),
    }

def build_cot_prompt(parsed: dict) -> list:
    sys_prompt = (
        "你是一个极其严苛的工业安全检查专家。你的任务是分析施工现场图片，判定【边防护设施】是否合规。\n"
        "【核心规则】：只有在危险区域（如基坑、高处边坡、孔洞边缘）明确存在物理隔离（围栏、护栏网片等）时才算合规。如果是正在建设、挖掘的边坡或基坑且没有任何护栏遮挡，必须判定为违规！\n"
        "【输出格式要求】：\n"
        "1. 首先，你需要详细描述图片中危险区域的状态，并进行安全推理（不超过200字）。\n"
        "2. 然后，你必须在回答的最后严格以 JSON 格式输出判定结果。不要用 Markdown 块包裹 JSON，只要纯 JSON。\n"
        "JSON 字段必须包含：violation_detected(布尔), violation_type(字符串), severity(字符串), suggestion(字符串)。\n"
        "违规类型可选：normal(合规), fence_damaged(破损), no_edge_protection(无防护)。"
    )

    boxes = parsed["qwen_boxes_prompt"]
    if boxes:
        user_query = (
            f"前置系统已在图片中检测到以下目标（坐标格式为<box_start>(ymin,xmin),(ymax,xmax)<box_end>）：\n{boxes}\n\n"
            "请结合这些位置信息仔细观察：这些区域（尤其是基坑、边坡等危险区域）是否存在防护缺失、护栏不全的情况？请先输出你的推理过程，最后输出 JSON 结果。"
        )
    else:
        user_query = "请仔细观察图片中是否存在高处边坡或基坑等危险区域，它们边缘的物理防护隔离（围栏等）是否缺失？请先输出你的推理过程，最后输出 JSON 结果。"

    messages = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{parsed['image_path']}"},
                {"type": "text", "text": user_query},
            ],
        },
    ]
    return messages

def parse_vlm_output(output_text: str) -> dict:
    cot_text = ""
    json_result = {}

    json_start = output_text.rfind("{")
    json_end = output_text.rfind("}")

    if json_start != -1 and json_end != -1 and json_end > json_start:
        cot_text = output_text[:json_start].strip()
        json_str = output_text[json_start:json_end+1]
        try:
            json_result = json.loads(json_str)
        except:
            json_result = {"error": "解析JSON失败", "raw": json_str}
    else:
        cot_text = output_text

    return {"cot_reasoning": cot_text, "json_output": json_result, "raw_response": output_text}

def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    print("Parsing LabelMe JSON files...")
    json_files = sorted(glob.glob(os.path.join(DATASET_DIR, "**/*.json"), recursive=True))
    parsed_data = []

    for jf in json_files:
        pd = parse_labelme_to_qwen_boxes(jf)
        if os.path.exists(pd["image_path"]):
            parsed_data.append(pd)

    print(f"Found {len(parsed_data)} valid image-annotation pairs.")

    model, processor, model_family = load_model()

    print("\nStarting Auto-Annotation...")
    results = []
    start_time = time.time()

    for i, data in enumerate(tqdm(parsed_data)):
        messages = build_cot_prompt(data)

        response_text = infer_vlm(
            model, processor, model_family, messages,
            max_new_tokens=768, do_sample=True, temperature=0.3,
        )

        vlm_parsed = parse_vlm_output(response_text)

        record = {
            "image_path": data["image_path"],
            "original_labels": data["labels"],
            "vlm_reasoning": vlm_parsed["cot_reasoning"],
            "vlm_json": vlm_parsed["json_output"],
            "raw_response": vlm_parsed["raw_response"]
        }
        results.append(record)

        if (i + 1) % 5 == 0:
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"  Saved {i+1}/{len(parsed_data)} annotations.")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nAuto-Annotation completed in {time.time() - start_time:.1f}s")
    print(f"Results saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
