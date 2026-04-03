"""
Qwen2.5-VL-7B 边防护违规识别 - 评估脚本
在测试集上评估模型性能
"""

import argparse
import json
import os
import sys

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def load_model(model_path: str, lora_path: str = None):
    """加载模型"""
    print(f"Loading model: {model_path}")
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
        print(f"Loading LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def evaluate(model, processor, test_data: list) -> dict:
    """评估模型在测试集上的表现"""
    total = len(test_data)
    correct_violation = 0  # 违规判定正确数
    correct_type = 0  # 违规类型正确数
    results = []

    for i, sample in enumerate(test_data):
        messages = sample["messages"]
        system_message = next((m for m in messages if m["role"] == "system"), None)
        user_message = next((m for m in messages if m["role"] == "user"), None)
        if system_message is None or user_message is None:
            print(f"  [{i+1}] Missing system/user message, skipping")
            continue
        user_content = user_message.get("content")
        if isinstance(user_content, str) and "<image>" in user_content:
            image_path = sample.get("images", [None])[0]
            if image_path is None or (not os.path.exists(image_path)):
                print(f"  [{i+1}] Missing image path, skipping")
                continue
            text_query = user_content.replace("<image>", "").strip()
            user_content = [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": text_query},
            ]
        input_messages = [
            {"role": "system", "content": system_message["content"]},
            {"role": "user", "content": user_content},
        ]
        expected_raw = messages[-1]["content"]  # assistant 的预期回复

        try:
            expected = json.loads(expected_raw)
        except json.JSONDecodeError:
            print(f"  [{i+1}] Cannot parse expected JSON, skipping")
            continue

        # 推理
        text = processor.apply_chat_template(
            input_messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(input_messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        try:
            predicted = json.loads(output_text)
        except json.JSONDecodeError:
            predicted = {"raw_output": output_text, "violation_detected": None}

        # 比较
        exp_violation = expected.get("violation_detected")
        pred_violation = predicted.get("violation_detected")
        exp_type = expected.get("violation_type")
        pred_type = predicted.get("violation_type")

        violation_match = exp_violation == pred_violation
        type_match = exp_type == pred_type

        if violation_match:
            correct_violation += 1
        if type_match:
            correct_type += 1

        status = "OK" if violation_match else "WRONG"
        print(f"  [{i+1}/{total}] {status} | Expected: {exp_violation}({exp_type}) | Predicted: {pred_violation}({pred_type})")

        results.append({
            "expected_violation": exp_violation,
            "predicted_violation": pred_violation,
            "expected_type": exp_type,
            "predicted_type": pred_type,
            "violation_match": violation_match,
            "type_match": type_match,
        })

    metrics = {
        "total": total,
        "evaluated": len(results),
        "violation_accuracy": correct_violation / len(results) if results else 0,
        "type_accuracy": correct_type / len(results) if results else 0,
    }

    print(f"\n=== Evaluation Results ===")
    print(f"Total: {metrics['total']}, Evaluated: {metrics['evaluated']}")
    print(f"Violation Detection Accuracy: {metrics['violation_accuracy']:.2%}")
    print(f"Violation Type Accuracy: {metrics['type_accuracy']:.2%}")

    return {"metrics": metrics, "details": results}


def main():
    parser = argparse.ArgumentParser(description="评估边防护违规识别模型")
    parser.add_argument("--model_path", type=str, default="/home/fs-ai/llama-qwen/models/Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--test_data", type=str, default="/home/fs-ai/llama-qwen/data/test.jsonl")
    parser.add_argument("--output", type=str, default="/home/fs-ai/llama-qwen/outputs/eval_results.json")
    args = parser.parse_args()

    # 加载测试数据
    test_data = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test samples")

    model, processor = load_model(args.model_path, args.lora_path)
    results = evaluate(model, processor, test_data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
