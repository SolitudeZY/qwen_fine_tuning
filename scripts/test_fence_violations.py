"""
围栏违规检测评估脚本
测试指标：
  - 召回率 (Recall)：违规图片中检出违规的比例
  - 精确率 (Precision)：合规图片中判为合规的比例
  - IoU：预测框与 LabelMe 标注框的平均 IoU

用法：
  python scripts/test_fence_violations.py              # 默认快速模式
  python scripts/test_fence_violations.py --tiled      # 分块推理（更准但慢）
  python scripts/test_fence_violations.py --base-only  # 不加 LoRA，测基座
  python scripts/test_fence_violations.py --count 30   # 每类最多测 30 张
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from PIL import Image
from tqdm import tqdm

# 把 scripts/ 加入路径
sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_vlm, infer_vlm
from prompts import (
    SYSTEM_PROMPT, DEFAULT_QUERY,
    NON_COMPLIANT_DIR, COMPLIANT_IMAGE_DIRS, VALID_LABELS,
)
from tiled_infer import tiled_chat

# ── 模型路径 ──────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/fs-ai/llama-qwen/outputs/stage1_grounding/v0-20260414-133809/checkpoint-105-merged"
# 一阶段直接训练时: MODEL_PATH = "/home/fs-ai/llama-qwen/models/Qwen/Qwen3-VL-2B-Instruct"
LORA_PATH  = "/home/fs-ai/llama-qwen/outputs/stage2_json/v0-20260414-135237/checkpoint-285"

TILED_PIXEL_THRESHOLD = 4_000_000
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def get_violation_images(max_count: int) -> list[dict]:
    """从 NON_COMPLIANT_DIR 加载有 LabelMe 标注的违规图片。"""
    items = []
    if not os.path.isdir(NON_COMPLIANT_DIR):
        print(f"[!] 找不到违规目录: {NON_COMPLIANT_DIR}")
        return items
    for fname in os.listdir(NON_COMPLIANT_DIR):
        if os.path.splitext(fname)[1] not in IMG_EXTS:
            continue
        img_path = os.path.join(NON_COMPLIANT_DIR, fname)
        json_path = os.path.splitext(img_path)[0] + ".json"
        if not os.path.exists(json_path):
            continue
        try:
            label_data = json.load(open(json_path, encoding="utf-8"))
            gt_boxes = [
                s for s in label_data.get("shapes", [])
                if s.get("label") in VALID_LABELS
            ]
            if gt_boxes:
                items.append({"path": img_path, "gt_boxes": gt_boxes,
                               "img_w": label_data.get("imageWidth", 0),
                               "img_h": label_data.get("imageHeight", 0)})
        except Exception:
            pass
        if len(items) >= max_count:
            break
    return items


def get_compliant_images(max_count: int) -> list[dict]:
    """从 COMPLIANT_IMAGE_DIRS 加载合规图片（无需标注文件）。"""
    items = []
    for d in COMPLIANT_IMAGE_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if os.path.splitext(fname)[1] not in IMG_EXTS:
                continue
            items.append({"path": os.path.join(d, fname), "gt_boxes": []})
            if len(items) >= max_count:
                return items
    return items


# ── 推理 ──────────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict | None:
    """从模型输出中提取 JSON 对象。"""
    import re
    # 去掉 markdown 代码块
    text = re.sub(r"```(?:json)?", "", text).strip()
    # 找最外层 {}
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        d = json.loads(text[start:end])
        # 兼容 violation_detection typo
        if "violation_detection" in d and "violation_detected" not in d:
            d["violation_detected"] = d.pop("violation_detection")
        return d
    except Exception:
        return None


def infer_single(model, processor, family, img_path: str, use_tiled: bool) -> dict | None:
    """单张图推理，返回解析后的 JSON dict。"""
    try:
        w, h = Image.open(img_path).size
        is_large = w * h > TILED_PIXEL_THRESHOLD
    except Exception:
        is_large = False

    if use_tiled and is_large:
        return tiled_chat(
            model, processor, family, img_path,
            infer_fn=infer_vlm,
            system_prompt=SYSTEM_PROMPT,
            query=DEFAULT_QUERY,
            extract_fn=lambda t: (_extract_json(t), None, None),
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(img_path)}"},
            {"type": "text", "text": DEFAULT_QUERY},
        ]},
    ]
    raw = infer_vlm(model, processor, family, messages, max_new_tokens=1024)
    return _extract_json(raw)


# ── IoU 计算 ──────────────────────────────────────────────────────────────────
def _labelme_to_norm1000(points: list, img_w: int, img_h: int) -> list:
    """LabelMe polygon/rectangle points → [x0,y0,x1,y1] 千分比。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [
        int(min(xs) / img_w * 1000),
        int(min(ys) / img_h * 1000),
        int(max(xs) / img_w * 1000),
        int(max(ys) / img_h * 1000),
    ]


def _iou(a: list, b: list) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def compute_iou_stats(pred_boxes: list, gt_boxes: list, img_w: int, img_h: int) -> dict:
    """
    计算预测框与 GT 框的匹配 IoU。
    返回 {mean_iou, matched, total_gt, total_pred}
    """
    if not gt_boxes or not pred_boxes:
        return {"mean_iou": 0.0, "matched": 0,
                "total_gt": len(gt_boxes), "total_pred": len(pred_boxes)}

    gt_norm = [_labelme_to_norm1000(s["points"], img_w, img_h) for s in gt_boxes]
    pred_norm = [b["bbox"] for b in pred_boxes if len(b.get("bbox", [])) == 4]

    matched_ious = []
    used_pred = set()
    for gt in gt_norm:
        best_iou, best_j = 0.0, -1
        for j, pred in enumerate(pred_norm):
            if j in used_pred:
                continue
            iou = _iou(gt, pred)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou > 0:
            matched_ious.append(best_iou)
            used_pred.add(best_j)

    return {
        "mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "matched": len(matched_ious),
        "total_gt": len(gt_norm),
        "total_pred": len(pred_norm),
    }


# ── 主测试逻辑 ────────────────────────────────────────────────────────────────
def run_test(use_lora: bool, use_tiled: bool, count: int):
    lora = LORA_PATH if use_lora else None
    mode = "微调模型" if use_lora else "基座模型"
    infer_mode = "分块推理" if use_tiled else "快速单路"
    print(f"\n模型: {mode}  推理模式: {infer_mode}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    if lora:
        print(f"LORA_PATH:  {lora}")

    violation_items = get_violation_images(count)
    compliant_items = get_compliant_images(count)
    print(f"违规测试集: {len(violation_items)} 张  合规测试集: {len(compliant_items)} 张")

    model, processor, family = load_vlm(MODEL_PATH, lora_path=lora)

    # ── 违规图测试（召回率 + IoU）────────────────────────────────────────────
    print(f"\n[1/2] 违规图测试（召回率 + IoU）...")
    recall_correct = 0
    iou_stats_all = []
    violation_results = []
    t0 = time.time()

    for item in tqdm(violation_items):
        img_path = item["path"]
        t1 = time.time()
        parsed = infer_single(model, processor, family, img_path, use_tiled)
        elapsed = time.time() - t1

        detected = False
        iou_stat = {"mean_iou": 0.0, "matched": 0,
                    "total_gt": len(item["gt_boxes"]), "total_pred": 0}

        if parsed:
            detected = bool(parsed.get("violation_detected", False))
            if isinstance(detected, str):
                detected = detected.lower() == "true"
            # 有框但 detected=false，修正
            if not detected and parsed.get("violation_boxes"):
                detected = True

            if detected and item["gt_boxes"] and item["img_w"]:
                iou_stat = compute_iou_stats(
                    parsed.get("violation_boxes", []),
                    item["gt_boxes"],
                    item["img_w"], item["img_h"],
                )
                iou_stats_all.append(iou_stat["mean_iou"])

        if detected:
            recall_correct += 1

        violation_results.append({
            "image": os.path.basename(img_path),
            "detected": detected,
            "violation_type": parsed.get("violation_type", "") if parsed else "",
            "iou": iou_stat,
            "elapsed": round(elapsed, 2),
        })

    recall = recall_correct / len(violation_items) if violation_items else 0
    mean_iou = sum(iou_stats_all) / len(iou_stats_all) if iou_stats_all else 0

    # ── 合规图测试（精确率）──────────────────────────────────────────────────
    print(f"\n[2/2] 合规图测试（精确率）...")
    precision_correct = 0
    compliant_results = []

    for item in tqdm(compliant_items):
        img_path = item["path"]
        t1 = time.time()
        parsed = infer_single(model, processor, family, img_path, use_tiled)
        elapsed = time.time() - t1

        is_compliant = True
        if parsed:
            detected = bool(parsed.get("violation_detected", False))
            if isinstance(detected, str):
                detected = detected.lower() == "true"
            if not detected and parsed.get("violation_boxes"):
                detected = True
            is_compliant = not detected

        if is_compliant:
            precision_correct += 1

        compliant_results.append({
            "image": os.path.basename(img_path),
            "is_compliant": is_compliant,
            "violation_type": parsed.get("violation_type", "") if parsed else "",
            "elapsed": round(elapsed, 2),
        })

    precision = precision_correct / len(compliant_items) if compliant_items else 0
    total_time = time.time() - t0

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  测试报告  [{mode} / {infer_mode}]")
    print("=" * 60)
    print(f"  召回率  (Recall)   : {recall*100:.1f}%  ({recall_correct}/{len(violation_items)})")
    print(f"  精确率  (Precision): {precision*100:.1f}%  ({precision_correct}/{len(compliant_items)})")
    print(f"  平均 IoU           : {mean_iou:.3f}  (基于 {len(iou_stats_all)} 张有框图)")
    print(f"  总耗时             : {total_time:.1f}s  (平均 {total_time/(len(violation_items)+len(compliant_items)):.1f}s/张)")
    print("=" * 60)

    # 前10条违规结果
    print("\n违规图前10条：")
    for r in violation_results[:10]:
        icon = "✅" if r["detected"] else "❌"
        iou_str = f"IoU={r['iou']['mean_iou']:.2f}" if r["iou"]["total_gt"] > 0 else ""
        print(f"  {icon} {r['image']}  类型:{r['violation_type']}  {iou_str}  {r['elapsed']}s")

    print("\n合规图前10条：")
    for r in compliant_results[:10]:
        icon = "✅" if r["is_compliant"] else "❌"
        print(f"  {icon} {r['image']}  {r['elapsed']}s")

    # 保存 JSON 报告
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "lora" if use_lora else "base"
    out_path = f"outputs/{prefix}_test_{ts}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "model": mode, "infer_mode": infer_mode,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "mean_iou": round(mean_iou, 4),
                "total_time": round(total_time, 1),
            },
            "violation_results": violation_results,
            "compliant_results": compliant_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true", help="不加载 LoRA")
    parser.add_argument("--tiled", action="store_true", help="启用分块推理（慢但准）")
    parser.add_argument("--count", type=int, default=50, help="每类最多测试张数")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_test(
        use_lora=not args.base_only,
        use_tiled=args.tiled,
        count=args.count,
    )
