"""
从 annotate_cache.json 随机抽样评估 Qwen3.5 模型。

默认行为:
  - 从 cache 中随机抽取 50 张合规图片和 50 张违规图片
  - 调用 Qwen3.5 模型逐张推理
  - 输出 precision / recall / 平均耗时 / 总耗时 / 混淆矩阵

用法示例:
  python scripts/qwen3.5-train/test_from_cache_qwen35.py \
      --model-path /home/fs-ai/llama-qwen/outputs/stage1_merged/qwen3.5/merged-20260526-120650 \
      --lora-path /home/fs-ai/llama-qwen/outputs/stage2_json/qwen3.5/v0-20260526-120748/checkpoint-808
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chat_qwen35 import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    DEFAULT_QUERY,
    infer_qwen35,
    load_qwen35,
)


DEFAULT_CACHE_PATH = "/home/fs-ai/llama-qwen/data/annotate_cache.json"
DEFAULT_SAMPLE_SIZE = 50


def load_cache(cache_path: str) -> dict:
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ground_truth(entry: dict) -> bool:
    """True 表示违规，False 表示合规。"""
    gc = entry.get("global_conclusion", {})
    if "violation_detected" in gc:
        return bool(gc["violation_detected"])
    return not bool(entry.get("is_compliant", True))


def filter_candidates(cache: dict, target_positive: bool, reviewed_only: bool) -> list[dict]:
    candidates = []
    for image_path, entry in cache.items():
        if reviewed_only and not entry.get("reviewed", False):
            continue
        if not os.path.exists(image_path):
            continue
        if get_ground_truth(entry) != target_positive:
            continue
        candidates.append(entry)
    return candidates


def sample_entries(candidates: list[dict], sample_size: int, rng: random.Random) -> list[dict]:
    if len(candidates) <= sample_size:
        return list(candidates)
    return rng.sample(candidates, sample_size)


def evaluate_samples(
    model,
    processor,
    samples: list[dict],
    query: str,
    system_prompt: str,
    max_new_tokens: int,
) -> tuple[list[dict], dict]:
    results = []
    tp = fp = tn = fn = 0
    total_elapsed = 0.0
    parse_failures = 0

    for idx, entry in enumerate(samples, start=1):
        image_path = entry["image"]
        gt_positive = get_ground_truth(entry)
        print(f"[{idx}/{len(samples)}] 测试: {os.path.basename(image_path)}")

        parsed, json_str, raw, elapsed = infer_qwen35(
            model,
            processor,
            image_path=image_path,
            query=query,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )
        total_elapsed += elapsed

        pred_positive = False
        if parsed is None:
            parse_failures += 1
        else:
            pred_positive = bool(parsed.get("violation_detected", False))

        if gt_positive and pred_positive:
            tp += 1
        elif (not gt_positive) and pred_positive:
            fp += 1
        elif (not gt_positive) and (not pred_positive):
            tn += 1
        else:
            fn += 1

        results.append(
            {
                "image": image_path,
                "ground_truth_violation": gt_positive,
                "predicted_violation": pred_positive,
                "elapsed_seconds": round(elapsed, 4),
                "parse_success": parsed is not None,
                "parsed_json": parsed,
                "raw_json": json_str,
                "raw_text": raw if parsed is None else "",
            }
        )

    total = len(samples)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    avg_elapsed = total_elapsed / total if total else 0.0

    metrics = {
        "total_samples": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "total_elapsed_seconds": total_elapsed,
        "avg_elapsed_seconds": avg_elapsed,
        "parse_failures": parse_failures,
    }
    return results, metrics


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 56)
    print("评估结果")
    print("=" * 56)
    print(f"样本数:     {metrics['total_samples']}")
    print(f"TP / FP:    {metrics['tp']} / {metrics['fp']}")
    print(f"TN / FN:    {metrics['tn']} / {metrics['fn']}")
    print(f"Precision:  {metrics['precision']:.4f}")
    print(f"Recall:     {metrics['recall']:.4f}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"总耗时:     {metrics['total_elapsed_seconds']:.2f}s")
    print(f"平均耗时:   {metrics['avg_elapsed_seconds']:.2f}s/张")
    print(f"解析失败:   {metrics['parse_failures']}")


def resolve_output_report_path(output_report: str) -> str:
    """兼容传入目录或完整文件路径两种形式。"""
    output_path = os.path.abspath(output_report)
    if not os.path.splitext(output_path)[1]:
        if os.path.isfile(output_path):
            return f"{output_path}.json"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_path, f"eval-report-{timestamp}.json")
    elif os.path.isdir(output_path):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = os.path.join(output_path, f"eval-report-{timestamp}.json")
    return output_path


def build_report_payload(args, metrics: dict, results: list[dict]) -> dict:
    """构建更适合排查性能问题的评估报告结构。"""
    per_image_timings = []
    for item in results:
        per_image_timings.append(
            {
                "image": item["image"],
                "image_name": os.path.basename(item["image"]),
                "elapsed_seconds": item["elapsed_seconds"],
                "ground_truth_violation": item["ground_truth_violation"],
                "predicted_violation": item["predicted_violation"],
                "parse_success": item["parse_success"],
            }
        )

    timing_values = [item["elapsed_seconds"] for item in per_image_timings]
    slowest_images = sorted(
        per_image_timings,
        key=lambda item: item["elapsed_seconds"],
        reverse=True,
    )[:10]

    timing_summary = {
        "count": len(timing_values),
        "min_elapsed_seconds": min(timing_values) if timing_values else 0.0,
        "max_elapsed_seconds": max(timing_values) if timing_values else 0.0,
        "avg_elapsed_seconds": metrics["avg_elapsed_seconds"],
        "total_elapsed_seconds": metrics["total_elapsed_seconds"],
        "slowest_images_top10": slowest_images,
    }

    return {
        "config": {
            "cache_path": args.cache_path,
            "model_path": args.model_path,
            "lora_path": args.lora_path,
            "sample_size_per_class": args.sample_size,
            "seed": args.seed,
            "reviewed_only": args.reviewed_only,
            "max_new_tokens": args.max_new_tokens,
            "device_map": args.device_map,
            "load_in_4bit": args.load_in_4bit,
        },
        "metrics": metrics,
        "timing_summary": timing_summary,
        "per_image_timings": per_image_timings,
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", type=str, default=DEFAULT_CACHE_PATH, help="annotate_cache.json 路径")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Qwen3.5 基座或合并模型路径")
    parser.add_argument("--lora-path", type=str, default=None, help="Stage 2 LoRA checkpoint 路径")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="每类抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--reviewed-only", action="store_true", help="仅使用 reviewed=true 的样本")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="用户查询")
    parser.add_argument("--system-prompt", type=str, default=None, help="系统提示词；为空时使用模型脚本默认值")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成长度")
    parser.add_argument("--device-map", type=str, default="auto", help="transformers device_map")
    parser.add_argument("--load-in-4bit", action="store_true", help="显式启用 4bit 量化")
    parser.add_argument("--output-report", type=str, default=None, help="可选：保存详细测试报告 JSON")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cache = load_cache(args.cache_path)
    rng = random.Random(args.seed)

    positive_candidates = filter_candidates(cache, target_positive=True, reviewed_only=args.reviewed_only)
    negative_candidates = filter_candidates(cache, target_positive=False, reviewed_only=args.reviewed_only)

    sampled_positive = sample_entries(positive_candidates, args.sample_size, rng)
    sampled_negative = sample_entries(negative_candidates, args.sample_size, rng)
    samples = sampled_positive + sampled_negative
    rng.shuffle(samples)

    print(f"违规候选: {len(positive_candidates)}，抽样: {len(sampled_positive)}")
    print(f"合规候选: {len(negative_candidates)}，抽样: {len(sampled_negative)}")
    print(f"总测试数: {len(samples)}")

    if not samples:
        raise RuntimeError("没有可用测试样本，请检查 cache 路径、图片是否存在，或去掉 --reviewed-only")

    model, processor = load_qwen35(
        model_path=args.model_path,
        lora_path=args.lora_path,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )

    if args.system_prompt is None:
        from chat_qwen35 import SYSTEM_PROMPT  # 延迟导入，保持参数优先级清晰

        system_prompt = SYSTEM_PROMPT
    else:
        system_prompt = args.system_prompt

    results, metrics = evaluate_samples(
        model,
        processor,
        samples=samples,
        query=args.query,
        system_prompt=system_prompt,
        max_new_tokens=args.max_new_tokens,
    )
    print_metrics(metrics)

    if args.output_report:
        report = build_report_payload(args, metrics, results)
        output_path = resolve_output_report_path(args.output_report)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n详细报告已保存: {output_path}")


if __name__ == "__main__":
    main()
