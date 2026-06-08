"""
Qwen3.5 施工现场安全监测推理脚本

特点:
  1. 仅面向 Qwen3.5 多模态模型，不复用旧的 model_utils 检测逻辑
  2. 默认不启用 4bit 量化，避免 bitsandbytes 依赖导致推理失败
  3. 支持单图推理、交互式推理、LoRA 加载、违规框可视化

用法:
  python scripts/qwen3.5-train/chat_qwen35.py \
      --image /path/to/img.jpg \
      --model-path /path/to/stage1_merged \
      --lora-path /path/to/stage2_checkpoint \
      --visualize
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from PIL import Image as PILImage, ImageDraw, ImageFont
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prompts import DEFAULT_QUERY, SYSTEM_PROMPT  # noqa: E402


DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "models" / "Qwen" / "Qwen3.5-2B")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "visualized" / "qwen3.5")

BOX_COLOR_RGB = (255, 50, 50)
LABEL_BG_COLOR = (200, 0, 0)
LABEL_TEXT_COLOR = (255, 255, 255)
BOX_THICKNESS = 3
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
CJK_FONT_SIZE = 22


def _fix_malformed_json(text: str) -> str:
    """尽量修复模型输出里的常见 JSON 小问题。"""

    def _fix_bbox_value(match: re.Match[str]) -> str:
        nums = re.findall(r"\d+", match.group(1))
        if len(nums) == 4:
            return f'"bbox": [{", ".join(nums)}]'
        return match.group(0)

    text = re.sub(r'"bbox"\s*:\s*"([^"]*)"', _fix_bbox_value, text)

    result = []
    stack = []
    in_str = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            result.append(ch)
            continue
        if ch == "\\":
            escape = True
            result.append(ch)
            continue
        if ch == '"':
            in_str = not in_str
            result.append(ch)
            continue
        if in_str:
            result.append(ch)
            continue
        if ch in "{[":
            stack.append(ch)
            result.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
                result.append(ch)
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
                result.append(ch)
        else:
            result.append(ch)

    for opener in reversed(stack):
        result.append("}" if opener == "{" else "]")
    return "".join(result)


def extract_json(text: str) -> tuple[dict | None, str]:
    """从模型输出中提取 JSON。"""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end <= 0:
        return None, ""
    json_str = _fix_malformed_json(cleaned[start:end])
    try:
        data = json.loads(json_str)
    except Exception:
        return None, json_str
    if "violation_detection" in data and "violation_detected" not in data:
        data["violation_detected"] = data.pop("violation_detection")
    return data, json_str


def draw_violation_boxes(image_path: str, boxes: list[dict], output_path: str) -> str:
    """绘制 0-1000 千分比坐标的违规框。"""
    img = PILImage.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    try:
        font = ImageFont.truetype(CJK_FONT_PATH, CJK_FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for box in boxes:
        bbox = box.get("bbox", [])
        label = box.get("label", "违规")
        if len(bbox) != 4:
            continue

        x0 = int(bbox[0] / 1000 * width)
        y0 = int(bbox[1] / 1000 * height)
        x1 = int(bbox[2] / 1000 * width)
        y1 = int(bbox[3] / 1000 * height)

        for t in range(BOX_THICKNESS):
            draw.rectangle([x0 - t, y0 - t, x1 + t, y1 + t], outline=BOX_COLOR_RGB)

        text_box = draw.textbbox((x0, max(0, y0 - CJK_FONT_SIZE - 4)), label, font=font)
        draw.rectangle(text_box, fill=LABEL_BG_COLOR)
        draw.text((text_box[0], text_box[1]), label, fill=LABEL_TEXT_COLOR, font=font)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    return output_path


def load_qwen35(
    model_path: str,
    lora_path: str | None = None,
    load_in_4bit: bool = False,
    device_map: str = "auto",
):
    """加载 Qwen3.5 模型和可选 LoRA。"""
    print(f"[chat_qwen35] 加载模型: {model_path}")
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": device_map,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)

    if lora_path:
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA 路径不存在: {lora_path}")
        print(f"[chat_qwen35] 加载 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    model.eval()
    attn_impl = getattr(model.config, "_attn_implementation", None)
    print(
        f"[chat_qwen35] 运行配置: device={model.device}, "
        f"attn_impl={attn_impl}, load_in_4bit={load_in_4bit}"
    )
    return model, processor


def infer_qwen35(
    model,
    processor,
    image_path: str,
    query: str,
    system_prompt: str,
    max_new_tokens: int = 1024,
) -> tuple[dict | None, str, str, float]:
    """对单张图片推理，返回 parsed_json/json_str/raw_text/elapsed。"""
    start = time.time()
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": query},
            ],
        },
    ]

    chat_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        chat_kwargs["enable_thinking"] = False
        text = processor.apply_chat_template(messages, **chat_kwargs)
    except TypeError:
        chat_kwargs.pop("enable_thinking", None)
        text = processor.apply_chat_template(messages, **chat_kwargs)

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.2,
        )

    generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    raw = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    parsed, json_str = extract_json(raw)
    return parsed, json_str, raw, time.time() - start


def print_result(parsed: dict | None, json_str: str, raw: str, elapsed: float, show_raw_json: bool) -> None:
    print(f"\n[推理完成] 耗时 {elapsed:.1f}s")
    if not parsed:
        print("[!] 无法解析 JSON，原始输出如下：")
        print(raw[:1000])
        return

    detected = parsed.get("violation_detected", False)
    print(f"结论: {'违规' if detected else '合规'}")
    print(f"类型: {parsed.get('violation_type', '')}")
    print(f"严重度: {parsed.get('severity', '')}")
    print(f"建议: {parsed.get('suggestion', '')}")
    boxes = parsed.get("violation_boxes", [])
    if boxes:
        print(f"违规框 ({len(boxes)} 个):")
        for box in boxes:
            print(f"  - {box.get('label', '违规')} bbox={box.get('bbox')}")
    if show_raw_json and json_str:
        print("\n原始 JSON:")
        print(json_str)


def infer_single_image(
    model,
    processor,
    image_path: str,
    query: str,
    system_prompt: str,
    max_new_tokens: int,
    visualize: bool,
    output_dir: str,
    show_raw_json: bool,
) -> None:
    parsed, json_str, raw, elapsed = infer_qwen35(
        model,
        processor,
        image_path=image_path,
        query=query,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )
    print_result(parsed, json_str, raw, elapsed, show_raw_json)

    if visualize and parsed and parsed.get("violation_boxes"):
        out_name = Path(image_path).stem + "_annotated.jpg"
        out_path = os.path.join(output_dir, out_name)
        saved = draw_violation_boxes(image_path, parsed["violation_boxes"], out_path)
        print(f"可视化已保存: {saved}")


def interactive_mode(
    model,
    processor,
    query: str,
    system_prompt: str,
    max_new_tokens: int,
    visualize: bool,
    output_dir: str,
    show_raw_json: bool,
) -> None:
    print("\n" + "=" * 54)
    print("  Qwen3.5 施工现场安全监测")
    print("  输入图片路径，输入 q 退出")
    print("=" * 54 + "\n")

    while True:
        try:
            image_path = input("图片路径> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出")
            break

        if image_path.lower() in {"q", "quit", "exit"}:
            break
        if not image_path:
            continue
        if not os.path.exists(image_path):
            print(f"[!] 文件不存在: {image_path}")
            continue

        infer_single_image(
            model,
            processor,
            image_path=image_path,
            query=query,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            visualize=visualize,
            output_dir=output_dir,
            show_raw_json=show_raw_json,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="单图推理路径")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Qwen3.5 基座或合并模型路径")
    parser.add_argument("--lora-path", type=str, default=None, help="Stage 2 LoRA checkpoint 路径")
    parser.add_argument("--visualize", action="store_true", help="保存可视化结果")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="可视化输出目录")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成长度")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="用户查询")
    parser.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT, help="系统提示词")
    parser.add_argument("--device-map", type=str, default="auto", help="transformers device_map")
    parser.add_argument("--load-in-4bit", action="store_true", help="显式启用 4bit 量化")
    parser.add_argument("--show-raw-json", action="store_true", help="打印提取到的原始 JSON 字符串")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model, processor = load_qwen35(
        model_path=args.model_path,
        lora_path=args.lora_path,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )

    if args.image:
        infer_single_image(
            model,
            processor,
            image_path=args.image,
            query=args.query,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            visualize=args.visualize,
            output_dir=args.output_dir,
            show_raw_json=args.show_raw_json,
        )
    else:
        interactive_mode(
            model,
            processor,
            query=args.query,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            visualize=args.visualize,
            output_dir=args.output_dir,
            show_raw_json=args.show_raw_json,
        )


if __name__ == "__main__":
    main()
