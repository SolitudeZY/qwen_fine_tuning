"""
VLM 边防护违规识别 - 评估脚本
支持 Qwen2.5-VL 和 Qwen3-VL 系列模型
在测试集上评估模型性能
"""

import argparse
import json
import os

import torch
from model_utils import load_vlm, infer_vlm


def evaluate(model, processor, model_family: str, test_data: list) -> dict:
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

        # 推理（使用统一接口）
        output_text = infer_vlm(
            model, processor, model_family, input_messages, max_new_tokens=512
        )

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
        print(
            f"  [{i+1}/{total}] {status} | Expected: {exp_violation}({exp_type})"
            f" | Predicted: {pred_violation}({pred_type})"
        )

        results.append(
            {
                "expected_violation": exp_violation,
                "predicted_violation": pred_violation,
                "expected_type": exp_type,
                "predicted_type": pred_type,
                "violation_match": violation_match,
                "type_match": type_match,
            }
        )

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


def main() -> None:
    parser = argparse.ArgumentParser(description="评估边防护违规识别模型")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/fs-ai/llama-qwen/models/Qwen/Qwen3-VL-2B-Instruct",
        help="模型路径（支持 Qwen2.5-VL 和 Qwen3-VL）",
    )
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument(
        "--test_data",
        type=str,
        default="/home/fs-ai/llama-qwen/data/test.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/fs-ai/llama-qwen/outputs/eval_results.json",
    )
    args = parser.parse_args()

    # 加载测试数据
    test_data = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line))
    print(f"Loaded {len(test_data)} test samples")

    model, processor, model_family = load_vlm(args.model_path, args.lora_path)
    results = evaluate(model, processor, model_family, test_data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
