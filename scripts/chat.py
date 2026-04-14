"""
施工现场安全监测 - 交互式推理脚本

用法：
  python scripts/chat.py                                      # 交互模式
  python scripts/chat.py --image /path/to/img.jpg            # 单图推理
  python scripts/chat.py --image /path/to/img.jpg --visualize # 单图+可视化
  python scripts/chat.py --tiled                             # 启用分块推理
  python scripts/chat.py --lora_path outputs/.../checkpoint-X
  python scripts/chat.py --no-lora                           # 不加 LoRA
"""

import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_vlm, infer_vlm
from tiled_infer import tiled_chat
from prompts import SYSTEM_PROMPT, DEFAULT_QUERY

# ── 模型路径 ──────────────────────────────────────────────────────────────────
# 注意：两阶段训练时基座为 Stage 1 合并模型，一阶段训练时改为原始基座
MODEL_PATH = "/home/fs-ai/llama-qwen/outputs/stage1_grounding/v0-20260414-133809/checkpoint-105-merged"
LORA_PATH  = None   # 通过 --lora_path 传入，或在下方硬编码

TILED_PIXEL_THRESHOLD = 4_000_000   # 超过此像素数时大图提示（需 --tiled 才自动分块）

# ── 可视化配置 ────────────────────────────────────────────────────────────────
BOX_COLOR_RGB  = (255, 50, 50)
LABEL_BG_COLOR = (200, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255)
BOX_THICKNESS  = 3
CJK_FONT_PATH  = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CJK_FONT_SIZE  = 22


# ── JSON 提取 ─────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> tuple[dict | None, str, str]:
    """
    从模型输出中提取 JSON。
    返回 (parsed_dict, json_str, remaining_text)
    """
    text_clean = re.sub(r"```(?:json)?", "", text).strip()
    start = text_clean.find("{")
    end = text_clean.rfind("}") + 1
    if start == -1 or end == 0:
        return None, "", text
    json_str = text_clean[start:end]
    try:
        d = json.loads(json_str)
        if "violation_detection" in d and "violation_detected" not in d:
            d["violation_detected"] = d.pop("violation_detection")
        remaining = text_clean[:start] + text_clean[end:]
        return d, json_str, remaining.strip()
    except Exception:
        return None, "", text


# ── 可视化 ────────────────────────────────────────────────────────────────────
def draw_violation_boxes(image_path: str, boxes: list, output_path: str | None = None) -> str:
    """在图片上绘制违规框，返回输出路径。"""
    img = PILImage.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    try:
        font = ImageFont.truetype(CJK_FONT_PATH, CJK_FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        bbox = box.get("bbox", [])
        label = box.get("label", "违规")
        if len(bbox) != 4:
            continue
        x0 = int(bbox[0] / 1000 * w)
        y0 = int(bbox[1] / 1000 * h)
        x1 = int(bbox[2] / 1000 * w)
        y1 = int(bbox[3] / 1000 * h)

        for t in range(BOX_THICKNESS):
            draw.rectangle([x0-t, y0-t, x1+t, y1+t], outline=BOX_COLOR_RGB)

        text_bbox = draw.textbbox((x0, max(0, y0 - CJK_FONT_SIZE - 4)), label, font=font)
        draw.rectangle(text_bbox, fill=LABEL_BG_COLOR)
        draw.text((text_bbox[0], text_bbox[1]), label, fill=LABEL_TEXT_COLOR, font=font)

    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_annotated{ext}"
    img.save(output_path)
    return output_path


# ── 单图推理 ──────────────────────────────────────────────────────────────────
def infer_image(
    model, processor, family,
    image_path: str,
    use_tiled: bool = False,
    query: str = DEFAULT_QUERY,
) -> tuple[dict | None, str, float, bool]:
    """
    推理单张图片。
    返回 (parsed_json, raw_response, elapsed_seconds, is_tiled)
    """
    t0 = time.time()
    try:
        w, h = PILImage.open(image_path).size
        is_large = w * h > TILED_PIXEL_THRESHOLD
    except Exception:
        is_large = False

    is_tiled = use_tiled and is_large

    if is_tiled:
        print("  [大图] 启用分块推理...")
        parsed = tiled_chat(
            model, processor, family, image_path,
            infer_fn=infer_vlm,
            system_prompt=SYSTEM_PROMPT,
            query=query,
            extract_fn=lambda t: (_extract_json(t)[0], None, None),
        )
        raw = json.dumps(parsed, ensure_ascii=False) if parsed else ""
    else:
        if is_large:
            print("  [提示] 大图检测到，使用 --tiled 可启用分块推理提升精度")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": query},
            ]},
        ]
        raw = infer_vlm(model, processor, family, messages, max_new_tokens=1024)
        parsed, _, _ = _extract_json(raw)

    elapsed = time.time() - t0
    return parsed, raw, elapsed, is_tiled


# ── 结果打印 ──────────────────────────────────────────────────────────────────
def print_result(parsed: dict | None, raw: str, elapsed: float, is_tiled: bool):
    mode = "分块" if is_tiled else "单路"
    print(f"\n[推理完成] 耗时 {elapsed:.1f}s ({mode})")
    if not parsed:
        print("[!] 无法解析 JSON，原始输出：")
        print(raw[:500])
        return

    detected = parsed.get("violation_detected", False)
    status = "🚨 违规" if detected else "✅ 合规"
    print(f"结论: {status}")
    if detected:
        print(f"类型: {parsed.get('violation_type', '')}")
        print(f"严重度: {parsed.get('severity', '')}")
        print(f"建议: {parsed.get('suggestion', '')}")
        boxes = parsed.get("violation_boxes", [])
        if boxes:
            print(f"违规框 ({len(boxes)} 个):")
            for b in boxes:
                print(f"  - {b.get('label')}  bbox={b.get('bbox')}")
    else:
        print(f"建议: {parsed.get('suggestion', '')}")


# ── 交互模式 ──────────────────────────────────────────────────────────────────
def interactive_mode(model, processor, family, use_tiled: bool, visualize: bool, output_dir: str | None):
    family_label = "Qwen3-VL" if "qwen3" in family else "Qwen2.5-VL"
    print(f"\n{'='*50}")
    print(f"  施工现场安全监测  [{family_label}]")
    print(f"  推理模式: {'分块' if use_tiled else '快速单路'}")
    print(f"  输入图片路径，输入 q 退出")
    print(f"{'='*50}\n")

    while True:
        try:
            img_path = input("图片路径> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if img_path.lower() in ("q", "quit", "exit"):
            break
        if not img_path:
            continue
        if not os.path.exists(img_path):
            print(f"[!] 文件不存在: {img_path}")
            continue

        parsed, raw, elapsed, is_tiled = infer_image(
            model, processor, family, img_path, use_tiled
        )
        print_result(parsed, raw, elapsed, is_tiled)

        if visualize and parsed and parsed.get("violation_boxes"):
            out_dir = output_dir or os.path.dirname(img_path)
            out_name = os.path.basename(img_path).rsplit(".", 1)[0] + "_annotated.jpg"
            out_path = os.path.join(out_dir, out_name)
            saved = draw_violation_boxes(img_path, parsed["violation_boxes"], out_path)
            print(f"可视化已保存: {saved}")


# ── 入口 ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="单图推理路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA checkpoint 路径")
    parser.add_argument("--model_path", type=str, default=None, help="基座模型路径（覆盖默认）")
    parser.add_argument("--no-lora", action="store_true", help="不加载 LoRA")
    parser.add_argument("--tiled", action="store_true", help="大图启用分块推理")
    parser.add_argument("--visualize", action="store_true", help="保存可视化结果")
    parser.add_argument("--output_dir", type=str, default=None, help="可视化输出目录")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.model_path:
        MODEL_PATH = args.model_path
    if args.lora_path:
        LORA_PATH = args.lora_path

    lora = None if args.no_lora else LORA_PATH
    model, processor, family = load_vlm(MODEL_PATH, lora_path=lora)

    if args.image:
        parsed, raw, elapsed, is_tiled = infer_image(
            model, processor, family, args.image, args.tiled
        )
        print_result(parsed, raw, elapsed, is_tiled)
        if args.visualize and parsed and parsed.get("violation_boxes"):
            out_dir = args.output_dir or os.path.dirname(args.image)
            out_name = os.path.basename(args.image).rsplit(".", 1)[0] + "_annotated.jpg"
            saved = draw_violation_boxes(
                args.image, parsed["violation_boxes"],
                os.path.join(out_dir, out_name)
            )
            print(f"可视化已保存: {saved}")
    else:
        interactive_mode(model, processor, family, args.tiled, args.visualize, args.output_dir)
