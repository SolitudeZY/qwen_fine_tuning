#!/usr/bin/env python3
"""
将 LabelMe 标注文件转换为 ms-swift grounding 格式 JSONL（Stage 1 训练数据）。

输出格式（ms-swift objects 字段）：
{
  "messages": [
    {"role": "user", "content": "<image>请找出图中所有围栏断口的位置"},
    {"role": "assistant", "content": "<ref-object><bbox>"}
  ],
  "images": ["/abs/path/image.jpg"],
  "objects": {
    "ref": ["围栏断口"],
    "bbox": [[x1, y1, x2, y2]],   # 像素坐标，ms-swift 根据 bbox_type 自动归一化
    "bbox_type": "real",
    "width": [W],
    "height": [H]
  }
}

多个同类标注 → 一条样本（ref/bbox 列表）
多种违规类型 → 每种类型生成一条独立样本
"""

import json
import os
import random
import argparse
from pathlib import Path

# 只关注安全违规相关标签，其余标签忽略
VIOLATION_LABELS = {"围栏断口", "围栏倒伏", "临边防护缺失"}

# 每种违规类型的多样化提问模板
GROUNDING_PROMPTS = {
    "围栏断口": [
        "请找出图中所有围栏断口的位置",
        "标注图中围栏断口区域",
        "图中哪些位置存在围栏断口？请标出来",
        "请定位图中的围栏断口",
    ],
    "围栏倒伏": [
        "请找出图中所有围栏倒伏的位置",
        "标注图中倒伏的围栏区域",
        "图中哪些围栏发生了倒伏？请标出来",
        "请定位图中倒伏的围栏",
    ],
    "临边防护缺失": [
        "请找出图中临边防护缺失的位置",
        "标注图中缺少防护的临边区域",
        "图中哪些临边位置缺少防护？请标出来",
        "请定位图中临边防护缺失的区域",
    ],
}


def shape_to_bbox(shape):
    """将 LabelMe shape（polygon 或 rectangle）转换为 [x_min, y_min, x_max, y_max] 像素坐标。"""
    pts = shape["points"]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def find_image_path(json_path: Path) -> Path | None:
    """根据 LabelMe JSON 中的 imagePath 字段找到对应图片文件。"""
    with open(json_path) as f:
        data = json.load(f)
    image_name = data.get("imagePath", "")
    # 先在同目录下找
    candidate = json_path.parent / image_name
    if candidate.exists():
        return candidate
    # 尝试大小写变体（.JPG / .jpg）
    for ext in [".JPG", ".jpg", ".PNG", ".png", ".jpeg"]:
        alt = json_path.parent / (json_path.stem + ext)
        if alt.exists():
            return alt
    return None


def process_json_file(json_path: Path) -> list[dict]:
    """处理单个 LabelMe JSON 文件，返回若干 grounding 样本。"""
    with open(json_path) as f:
        data = json.load(f)

    img_path = find_image_path(json_path)
    if img_path is None:
        print(f"  [跳过] 找不到图片: {json_path.name}")
        return []

    W = data.get("imageWidth")
    H = data.get("imageHeight")
    if not W or not H:
        print(f"  [跳过] 缺少图片尺寸: {json_path.name}")
        return []

    # 按违规类型分组收集 bbox
    label_bboxes: dict[str, list] = {}
    for shape in data.get("shapes", []):
        label = shape.get("label", "")
        if label not in VIOLATION_LABELS:
            continue
        bbox = shape_to_bbox(shape)
        label_bboxes.setdefault(label, []).append(bbox)

    if not label_bboxes:
        return []

    samples = []
    for label, bboxes in label_bboxes.items():
        prompt = random.choice(GROUNDING_PROMPTS[label])
        # 多个同类标注 → ref/bbox 列表，assistant 中重复占位符
        ref_list = [label] * len(bboxes)
        assistant_content = "\n".join(["<ref-object><bbox>"] * len(bboxes))

        sample = {
            "messages": [
                {"role": "user", "content": f"<image>{prompt}"},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [str(img_path.resolve())],
            "objects": {
                "ref": ref_list,
                "bbox": bboxes,
                "bbox_type": "real",
                "width": [W],
                "height": [H],
            },
        }
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description="LabelMe → ms-swift grounding JSONL")
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        default=["/home/fs-ai/llama-qwen/Fences_noncomlaint"],
        help="包含 LabelMe JSON 的目录列表",
    )
    parser.add_argument(
        "--output",
        default="/home/fs-ai/llama-qwen/data/grounding_stage1.jsonl",
        help="输出 JSONL 路径",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_samples = []
    for dir_path in args.input_dirs:
        p = Path(dir_path)
        json_files = sorted(p.glob("*.json"))
        print(f"扫描 {p}: {len(json_files)} 个 JSON 文件")
        for jf in json_files:
            samples = process_json_file(jf)
            if samples:
                all_samples.extend(samples)
                print(f"  {jf.name}: {len(samples)} 条样本")

    random.shuffle(all_samples)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 统计
    label_counts: dict[str, int] = {}
    for s in all_samples:
        for ref in s["objects"]["ref"]:
            label_counts[ref] = label_counts.get(ref, 0) + 1

    print(f"\n完成！共生成 {len(all_samples)} 条 grounding 样本")
    print("标签分布：")
    for k, v in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"输出: {args.output}")


if __name__ == "__main__":
    main()
