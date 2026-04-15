"""
为大图生成分块视角的 Stage 1 grounding 训练样本。

问题背景：
  训练时模型看到完整大图（4032×3024），bbox 坐标相对于全图。
  分块推理时模型看到 2×2 裁剪块，坐标系完全不同，导致定位失准。

解决方案：
  对 stage1_grounding.jsonl 中的大图样本，按照与 tiled_infer.py 相同的
  分块参数（2×2，20% 重叠）生成裁剪块，将 bbox 转换为块内坐标，
  追加到训练数据中。

用法：
  python scripts/2_stage_train/gen_tiled_stage1.py
  python scripts/2_stage_train/gen_tiled_stage1.py --min_pixels 2000000 --output data/stage1_tiled.jsonl
"""

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image

# 与 tiled_infer.py 保持一致
TILE_ROWS = 2
TILE_COLS = 2
OVERLAP = 0.20

# 大图阈值：超过此像素数才生成分块样本
DEFAULT_MIN_PIXELS = 2_000_000

PROMPTS = {
    "围栏断口": ["请定位图中围栏断口的位置", "找出图中围栏开口或缺口区域", "标注图中围栏断裂处"],
    "围栏倒伏": ["请定位图中倒伏的围栏", "找出图中倒塌的围栏区域"],
    "临边防护缺失": ["请定位图中临边防护缺失的区域", "找出图中缺少防护的临边位置", "标注图中高坠风险区域"],
    "临边开口未防护": ["请定位图中未防护的临边开口", "找出图中缺少防护门或隔离措施的开口位置"],
}


def compute_tiles(W: int, H: int) -> list[dict]:
    """计算分块参数，与 tiled_infer._tile_image 逻辑一致。"""
    tile_w = int(W / (TILE_COLS - OVERLAP * (TILE_COLS - 1)))
    tile_h = int(H / (TILE_ROWS - OVERLAP * (TILE_ROWS - 1)))
    step_x = int(tile_w * (1 - OVERLAP))
    step_y = int(tile_h * (1 - OVERLAP))

    tiles = []
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            x0 = col * step_x
            y0 = row * step_y
            x1 = min(x0 + tile_w, W)
            y1 = min(y0 + tile_h, H)
            tiles.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                           "w": x1 - x0, "h": y1 - y0})
    return tiles


def clip_bbox_to_tile(bbox_px: list[int], tile: dict, min_overlap: float = 0.3) -> list[int] | None:
    """
    将原图像素 bbox 裁剪到 tile 范围内，转换为 tile 内坐标。
    若 bbox 与 tile 的交集面积 < bbox 面积 * min_overlap，返回 None（框主体不在此块内）。
    """
    bx0, by0, bx1, by1 = bbox_px
    ix0 = max(bx0, tile["x0"])
    iy0 = max(by0, tile["y0"])
    ix1 = min(bx1, tile["x1"])
    iy1 = min(by1, tile["y1"])

    if ix1 <= ix0 or iy1 <= iy0:
        return None

    inter_area = (ix1 - ix0) * (iy1 - iy0)
    bbox_area = max(1, (bx1 - bx0) * (by1 - by0))
    if inter_area / bbox_area < min_overlap:
        return None

    # 转为 tile 内坐标
    return [ix0 - tile["x0"], iy0 - tile["y0"],
            ix1 - tile["x0"], iy1 - tile["y0"]]


def save_tile_image(image_path: str, tile: dict, out_dir: str, tile_idx: int) -> str:
    """裁剪并保存子图，返回保存路径。"""
    img = Image.open(image_path).convert("RGB")
    crop = img.crop((tile["x0"], tile["y0"], tile["x1"], tile["y1"]))
    stem = Path(image_path).stem
    out_path = os.path.join(out_dir, f"{stem}_tile{tile_idx}.jpg")
    crop.save(out_path, quality=95)
    return out_path


def process_sample(sample: dict, tile_img_dir: str, min_pixels: int) -> list[dict]:
    """
    对一条 stage1 样本生成分块版本。
    若图片不是大图则返回空列表。
    """
    W = sample["objects"]["width"][0]
    H = sample["objects"]["height"][0]
    if W * H < min_pixels:
        return []

    image_path = sample["images"][0]
    if not os.path.exists(image_path):
        print(f"  [跳过] 图片不存在: {image_path}")
        return []

    # 按 label 分组原始 bbox（像素坐标）
    from collections import defaultdict
    groups: dict[str, list[list[int]]] = defaultdict(list)
    refs = sample["objects"]["ref"]
    bboxes = sample["objects"]["bbox"]
    for label, bbox in zip(refs, bboxes):
        groups[label].append(bbox)

    tiles = compute_tiles(W, H)
    new_samples = []

    for tile_idx, tile in enumerate(tiles):
        tile_path = save_tile_image(image_path, tile, tile_img_dir, tile_idx)
        tw, th = tile["w"], tile["h"]

        # 对每个 label 找落在此 tile 内的框
        tile_groups: dict[str, list[list[int]]] = defaultdict(list)
        for label, bboxes_px in groups.items():
            for bbox_px in bboxes_px:
                clipped = clip_bbox_to_tile(bbox_px, tile)
                if clipped is not None:
                    tile_groups[label].append(clipped)

        if not tile_groups:
            continue  # 此 tile 内没有违规框，跳过

        for label, tile_bboxes in tile_groups.items():
            prompt = random.choice(PROMPTS.get(label, ["请定位图中违规区域"]))
            new_samples.append({
                "messages": [
                    {"role": "user", "content": f"<image>{prompt}"},
                    {"role": "assistant", "content": "<ref-object><bbox>"},
                ],
                "images": [tile_path],
                "objects": {
                    "ref": [label] * len(tile_bboxes),
                    "bbox": tile_bboxes,
                    "bbox_type": "real",
                    "width": [tw] * len(tile_bboxes),
                    "height": [th] * len(tile_bboxes),
                },
            })

    return new_samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/stage1_grounding.jsonl")
    parser.add_argument("--output", default="data/stage1_tiled.jsonl")
    parser.add_argument("--tile_img_dir", default="data/tiled_crops",
                        help="分块子图保存目录")
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS,
                        help="大图阈值（像素数），超过此值才生成分块样本")
    parser.add_argument("--merge", action="store_true",
                        help="将分块样本追加到原始 stage1_grounding.jsonl（而非单独输出）")
    args = parser.parse_args()

    os.makedirs(args.tile_img_dir, exist_ok=True)

    with open(args.input) as f:
        samples = [json.loads(line) for line in f]

    large_count = sum(
        1 for s in samples
        if s["objects"]["width"][0] * s["objects"]["height"][0] >= args.min_pixels
    )
    print(f"共 {len(samples)} 条样本，其中大图 {large_count} 条")

    all_new = []
    for s in samples:
        new = process_sample(s, args.tile_img_dir, args.min_pixels)
        all_new.extend(new)

    print(f"生成分块样本 {len(all_new)} 条")

    if args.merge:
        out_path = args.input
        with open(out_path, "a") as f:
            for s in all_new:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"已追加到 {out_path}，总计 {len(samples) + len(all_new)} 条")
    else:
        with open(args.output, "w") as f:
            for s in all_new:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"已写入 {args.output}")
        print(f"若确认无误，运行以下命令合并到训练集：")
        print(f"  cat {args.output} >> {args.input}")


if __name__ == "__main__":
    main()
