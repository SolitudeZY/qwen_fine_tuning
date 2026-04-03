"""
Qwen2.5-VL-7B 边防护违规识别 - 推理脚本
支持加载 QLoRA 微调后的权重进行单图/批量推理
"""

import argparse
import json
import os
import time

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import torch


def load_model(model_path: str, lora_path: str = None):
    """加载模型（可选加载 LoRA 权重）"""
    print(f"Loading base model from: {model_path}")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )

    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def infer_single(model, processor, image_path: str, query: str = None) -> dict:
    """单张图片推理"""
    if query is None:
        query = "请检查这张施工现场图片中的边防护设施是否合规，是否存在围栏缺失、防护不全等违规情况。"

    system_prompt = (
        "你是一个专业的工业安全检查AI助手，专注于施工现场的边防护设施安全检查。"
        "你需要分析图片中的施工现场，重点关注临边防护设施（围栏、护栏、防护网、安全围挡等）的完整性和合规性。"
        "请以JSON格式输出检查结果。"
    )

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

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    start_time = time.time()
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    # 去掉输入部分，只保留生成的token
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    elapsed = time.time() - start_time

    # 尝试解析为 JSON
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError:
        result = {"raw_output": output_text}

    result["_inference_time"] = f"{elapsed:.2f}s"
    result["_image_path"] = image_path
    return result


def main():
    parser = argparse.ArgumentParser(description="边防护违规识别推理")
    parser.add_argument("--model_path", type=str, default="/home/fs-ai/llama-qwen/models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 权重路径（微调后）")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--image_dir", type=str, help="图片目录（批量推理）")
    parser.add_argument("--output", type=str, default="inference_results.json", help="输出结果文件")
    parser.add_argument("--query", type=str, default=None, help="自定义提问文本")
    args = parser.parse_args()

    model, processor = load_model(args.model_path, args.lora_path)

    results = []
    if args.image:
        result = infer_single(model, processor, args.image, query=args.query)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.image_dir:
        import glob
        images = sorted(
            glob.glob(os.path.join(args.image_dir, "*.jpg"))
            + glob.glob(os.path.join(args.image_dir, "*.JPG"))
            + glob.glob(os.path.join(args.image_dir, "*.png"))
        )
        print(f"Found {len(images)} images in {args.image_dir}")
        for i, img in enumerate(images):
            print(f"\n[{i+1}/{len(images)}] Processing: {img}")
            result = infer_single(model, processor, img, query=args.query)
            results.append(result)
            violation = result.get("violation_detected", "N/A")
            vtype = result.get("violation_type", "N/A")
            print(f"  Violation: {violation}, Type: {vtype}")
    else:
        print("Please specify --image or --image_dir")
        return

    # 保存结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
