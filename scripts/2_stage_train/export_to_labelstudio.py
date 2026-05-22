"""
从 data/annotate_cache.json 导出已人工审查的标注到 Label Studio 导入格式。

输出结构（默认 export/labelstudio/）：
  export/labelstudio/
    images/                 # 拷贝的图片文件
    labelstudio_import.json # Label Studio Import JSON（含 predictions）
    manifest.json           # 导出元信息

用法：
  python scripts/2_stage_train/export_to_labelstudio.py \
      --out export/labelstudio \
      [--only violation|compliant|all] \
      [--image-prefix /data/local-files/?d=images]

注意：
  - 仅导出 reviewed=true 的条目
  - bbox_1000 (x_min,y_min,x_max,y_max, 千分位) → LS percentage (0-100)
  - rectanglelabels 取 violations[*].label
  - 整体合规标签写入 image-level Choices（is_compliant -> 合规/违规）
"""
import argparse
import json
import shutil
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = ROOT / "data" / "annotate_cache.json"

VALID_LABELS = ["临边防护缺失"]


def bbox_to_ls(bbox_1000, img_w, img_h):
    """千分位 [x0,y0,x1,y1] → LS 百分比 {x,y,width,height}."""
    x0, y0, x1, y1 = bbox_1000
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    x0 = max(0, min(1000, x0)); x1 = max(0, min(1000, x1))
    y0 = max(0, min(1000, y0)); y1 = max(0, min(1000, y1))
    return {
        "x": x0 / 10.0,
        "y": y0 / 10.0,
        "width": (x1 - x0) / 10.0,
        "height": (y1 - y0) / 10.0,
        "rotation": 0,
    }


def build_task(entry, image_url):
    img_w = entry.get("width", 1024)
    img_h = entry.get("height", 1024)
    results = []

    for v in entry.get("violations", []):
        bbox = v.get("bbox_1000")
        label = v.get("label")
        if not bbox or len(bbox) != 4 or label not in VALID_LABELS:
            continue
        rect = bbox_to_ls(bbox, img_w, img_h)
        results.append({
            "id": uuid.uuid4().hex[:10],
            "type": "rectanglelabels",
            "from_name": "label",
            "to_name": "image",
            "original_width": img_w,
            "original_height": img_h,
            "image_rotation": 0,
            "value": {**rect, "rectanglelabels": [label]},
        })

    # image-level choice：合规 / 违规
    compliant_choice = "合规" if entry.get("is_compliant") else "违规"
    results.append({
        "id": uuid.uuid4().hex[:10],
        "type": "choices",
        "from_name": "compliance",
        "to_name": "image",
        "value": {"choices": [compliant_choice]},
    })

    gc = entry.get("global_conclusion", {})
    task = {
        "data": {
            "image": image_url,
            "scene": entry.get("scene", ""),
            "suggestion": gc.get("suggestion", ""),
            "violation_type": gc.get("violation_type", ""),
            "severity": gc.get("severity", ""),
        },
        "predictions": [{
            "model_version": entry.get("model", "qwen-vl") + "+human-review",
            "result": results,
        }],
    }
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "export" / "labelstudio"))
    ap.add_argument("--only", choices=["all", "violation", "compliant"], default="all")
    ap.add_argument("--image-prefix", default="/data/local-files/?d=images",
                    help="Label Studio 中图片 URL 前缀，最终为 prefix/<filename>")
    ap.add_argument("--copy-images", action="store_true", default=True)
    ap.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    args = ap.parse_args()

    if not CACHE_FILE.exists():
        raise SystemExit(f"未找到 {CACHE_FILE}")

    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    if args.copy_images:
        img_dir.mkdir(exist_ok=True)

    tasks = []
    n_violation = 0
    n_compliant = 0
    n_skipped_no_review = 0
    n_missing_image = 0
    seen_names = {}

    for key, entry in cache.items():
        if not entry.get("reviewed"):
            n_skipped_no_review += 1
            continue
        is_compliant = bool(entry.get("is_compliant"))
        if args.only == "violation" and is_compliant:
            continue
        if args.only == "compliant" and not is_compliant:
            continue

        src = Path(entry.get("image", key))
        if not src.exists():
            n_missing_image += 1
            continue

        # 处理重名
        name = src.name
        if name in seen_names:
            seen_names[name] += 1
            stem, suf = src.stem, src.suffix
            name = f"{stem}__{seen_names[src.name]}{suf}"
        else:
            seen_names[name] = 0

        if args.copy_images:
            shutil.copy2(src, img_dir / name)

        image_url = f"{args.image_prefix.rstrip('/')}/{name}"
        tasks.append(build_task(entry, image_url))

        if is_compliant:
            n_compliant += 1
        else:
            n_violation += 1

    out_json = out_dir / "labelstudio_import.json"
    out_json.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "total_tasks": len(tasks),
        "violation": n_violation,
        "compliant": n_compliant,
        "skipped_unreviewed": n_skipped_no_review,
        "skipped_missing_image": n_missing_image,
        "only_filter": args.only,
        "image_prefix": args.image_prefix,
        "label_config_hint": (
            "<View>\n"
            "  <Image name=\"image\" value=\"$image\"/>\n"
            "  <RectangleLabels name=\"label\" toName=\"image\">\n"
            "    <Label value=\"临边防护缺失\" background=\"#CC00FF\"/>\n"
            "  </RectangleLabels>\n"
            "  <Choices name=\"compliance\" toName=\"image\" choice=\"single\">\n"
            "    <Choice value=\"合规\"/>\n"
            "    <Choice value=\"违规\"/>\n"
            "  </Choices>\n"
            "</View>"
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"导出完成：{out_json}")
    print(f"  任务数: {len(tasks)}（违规 {n_violation} / 合规 {n_compliant}）")
    print(f"  跳过未审查: {n_skipped_no_review}，图片缺失: {n_missing_image}")
    if args.copy_images:
        print(f"  图片输出: {img_dir}")
    print(f"  Label Studio 标签配置见 manifest.json 的 label_config_hint")


if __name__ == "__main__":
    main()
