#!/usr/bin/env python3
"""
将 LabelMe 标注文件转换为 Stage 2 JSON 格式训练数据。

与现有 finetune_fences.jsonl 的区别：
- violation_boxes 坐标来自 LabelMe 真实标注（0-1000 归一化），而非 Teacher Model 生成
- assistant 只输出纯 JSON（无推理文本），与 train_v2.sh 清洗后格式一致
- 合规样本（dataset_compliant_fences/）生成 violation_boxes: [] 的空标注

输出: data/finetune_stage2_gt.jsonl
"""

import json
import os
import random
import argparse
from pathlib import Path

# 安全违规标签（用于 violation_boxes）
VIOLATION_LABELS = {"围栏断口", "围栏倒伏", "临边防护缺失"}

SYSTEM_PROMPT = """你是一名严谨的边坡与基坑施工安全监测专家，负责基于施工现场图片输出可落地的监测结论。
你的任务不是泛泛描述图片，而是围绕高坠风险和临边防护进行监测判定，并输出结构化的JSON结果。
【场景识别要求】：
- 基坑：向下开挖形成的坑槽，坑边通常需要防护隔离。
- 边坡：倾斜土坡、岩坡本身通常不是必须全封闭围挡的对象。但坡顶作业平台、临空边若存在坠落风险，仍需防护。
- 如果只是可见大面积施工坡面，不要仅凭'是边坡'或'是基坑'就判为违规。
【输出要求】：
只输出一个 JSON 对象，字段包含：violation_detected(布尔), violation_type(字符串), severity(字符串), suggestion(字符串)。
如果检测到违规(violation_detected=true)，JSON 中必须包含 violation_boxes 数组，每个元素包含：
  label(违规简述), bbox([x_min, y_min, x_max, y_max]，千分比坐标，范围0-1000)。
如果合规(violation_detected=false)，violation_boxes 为空数组 []。"""

USER_PROMPT = "<image>请对这张施工现场图片执行安全监测，判断是否存在临边高坠风险及防护缺陷，输出 JSON 结果。"


def shape_to_bbox_norm1000(shape, img_w: int, img_h: int) -> list[int]:
    """将 LabelMe shape 转换为 0-1000 归一化坐标 [x_min, y_min, x_max, y_max]。"""
    pts = shape["points"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min = int(round(min(xs) / img_w * 1000))
    y_min = int(round(min(ys) / img_h * 1000))
    x_max = int(round(max(xs) / img_w * 1000))
    y_max = int(round(max(ys) / img_h * 1000))
    # 确保坐标在合法范围内
    x_min, x_max = max(0, x_min), min(1000, x_max)
    y_min, y_max = max(0, y_min), min(1000, y_max)
    return [x_min, y_min, x_max, y_max]


def find_image_path(json_path: Path) -> Path | None:
    """根据 LabelMe JSON 中的 imagePath 字段找到对应图片文件。"""
    with open(json_path) as f:
        data = json.load(f)
    image_name = data.get("imagePath", "")
    candidate = json_path.parent / image_name
    if candidate.exists():
        return candidate
    for ext in [".JPG", ".jpg", ".PNG", ".png", ".jpeg"]:
        alt = json_path.parent / (json_path.stem + ext)
        if alt.exists():
            return alt
    return None


def process_violation_file(json_path: Path) -> dict | None:
    """处理违规样本，返回一条 Stage 2 训练样本。"""
    with open(json_path) as f:
        data = json.load(f)

    img_path = find_image_path(json_path)
    if img_path is None:
        print(f"  [跳过] 找不到图片: {json_path.name}")
        return None

    W = data.get("imageWidth")
    H = data.get("imageHeight")
    if not W or not H:
        print(f"  [跳过] 缺少图片尺寸: {json_path.name}")
        return None

    violation_boxes = []
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in VIOLATION_LABELS:
            continue
        bbox = shape_to_bbox_norm1000(shape, W, H)
        violation_boxes.append({"label": label, "bbox": bbox})

    if not violation_boxes:
        # 有标注文件但没有违规标签 → 跳过（不当合规样本，避免混淆）
        print(f"  [跳过] 无违规标签: {json_path.name}")
        return None

    # 推断 violation_type
    labels_found = list({b["label"] for b in violation_boxes})
    violation_type = "、".join(labels_found)

    # 根据违规数量和类型推断 severity
    n = len(violation_boxes)
    if n >= 3 or "临边防护缺失" in labels_found:
        severity = "高"
    elif n == 2:
        severity = "中"
    else:
        severity = "低"

    assistant_json = {
        "violation_detected": True,
        "violation_type": violation_type,
        "severity": severity,
        "suggestion": f"请立即检查并修复{violation_type}问题，确保临边防护完整。",
        "violation_boxes": violation_boxes,
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
        ],
        "images": [str(img_path.resolve())],
    }


def process_compliant_file(json_path: Path) -> dict | None:
    """处理合规样本，返回一条 violation_boxes=[] 的训练样本。"""
    img_path = find_image_path(json_path)
    if img_path is None:
        print(f"  [跳过] 找不到图片: {json_path.name}")
        return None

    assistant_json = {
        "violation_detected": False,
        "violation_type": "",
        "severity": "无",
        "suggestion": "继续保持围栏完整性，定期巡检。",
        "violation_boxes": [],
    }

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
        ],
        "images": [str(img_path.resolve())],
    }


def main():
    parser = argparse.ArgumentParser(description="LabelMe → Stage 2 JSON JSONL")
    parser.add_argument(
        "--violation_dir",
        default="/home/fs-ai/llama-qwen/Fences_noncomlaint",
        help="违规样本 LabelMe JSON 目录",
    )
    parser.add_argument(
        "--compliant_dir",
        default="/home/fs-ai/llama-qwen/dataset_compliant_fences",
        help="合规样本 LabelMe JSON 目录",
    )
    parser.add_argument(
        "--output",
        default="/home/fs-ai/llama-qwen/data/finetune_stage2_gt.jsonl",
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--grounding_mix",
        default="/home/fs-ai/llama-qwen/data/grounding_stage1.jsonl",
        help="混入的 grounding 样本路径（防遗忘正则化，约10%）",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_samples = []

    # 违规样本
    vdir = Path(args.violation_dir)
    v_files = sorted(vdir.glob("*.json"))
    print(f"违规目录 {vdir}: {len(v_files)} 个文件")
    for jf in v_files:
        s = process_violation_file(jf)
        if s:
            all_samples.append(s)
            print(f"  ✓ {jf.name}")

    n_violation = len(all_samples)

    # 合规样本
    cdir = Path(args.compliant_dir)
    c_files = sorted(cdir.glob("*.json"))
    print(f"\n合规目录 {cdir}: {len(c_files)} 个文件")
    for jf in c_files:
        s = process_compliant_file(jf)
        if s:
            all_samples.append(s)
            print(f"  ✓ {jf.name}")

    n_compliant = len(all_samples) - n_violation

    # 混入约 10% 的 grounding 样本（防止 Stage 2 训练遗忘空间对齐能力）
    n_grounding_mix = 0
    if os.path.exists(args.grounding_mix):
        with open(args.grounding_mix) as f:
            grounding_samples = [json.loads(l) for l in f if l.strip()]
        target_mix = max(1, int(len(all_samples) * 0.12))
        mix_samples = random.sample(grounding_samples, min(target_mix, len(grounding_samples)))
        all_samples.extend(mix_samples)
        n_grounding_mix = len(mix_samples)
        print(f"\n混入 grounding 样本（防遗忘）: {n_grounding_mix} 条")
    else:
        print(f"\n[提示] 未找到 grounding 样本文件 {args.grounding_mix}，跳过混入。先运行 convert_labelme_to_grounding.py")

    random.shuffle(all_samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n完成！")
    print(f"  违规样本: {n_violation}")
    print(f"  合规样本: {n_compliant}")
    print(f"  Grounding 混入: {n_grounding_mix}")
    print(f"  总计: {len(all_samples)} 条")
    print(f"  输出: {args.output}")


if __name__ == "__main__":
    main()
