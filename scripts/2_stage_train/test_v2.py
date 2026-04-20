"""
两阶段训练模型评估脚本 v2
数据源：data/annotate_cache.json（人工审查后的 canonical annotation）

测试指标：
  - 召回率 (Recall)：违规图中检出违规的比例
  - 精确率 (Precision)：合规图中判为合规的比例
  - IoU：预测框与 cache GT 框的平均 IoU（均为 0-1000 千分比，无需转换）

用法：
  python scripts/test_v2.py                  # 快速单路
  python scripts/test_v2.py --tiled          # 分块推理
  python scripts/test_v2.py --count 30       # 每类最多 30 张
  python scripts/test_v2.py --base-only      # 不加 LoRA
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.append(str(Path(__file__).parent.parent)) # 这里要使用str将Path()返回的对象转换成字符串，不然sys.path解析不了
# 而且print打印的时候会自动调用__str__方法进行隐式转换

from model_utils import load_vlm, infer_vlm
from prompts import SYSTEM_PROMPT, DEFAULT_QUERY
from tiled_infer import tiled_chat

# ── 模型路径 ──────────────────────────────────────────────────────────────────
MODEL_PATH = "/home/fs-ai/llama-qwen/outputs/stage1_grounding/v0-20260414-133809/checkpoint-105-merged"
LORA_PATH  = "/home/fs-ai/llama-qwen/outputs/stage2_json/v4-20260420-090209/checkpoint-285"
PROJECT_ROOT = "/home/fs-ai/llama-qwen"
CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "annotate_cache.json"
TILED_PIXEL_THRESHOLD = 4_000_000
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_test_data(max_count: int) -> tuple[list, list]:
    """从 annotate_cache.json 加载合规/违规测试集。"""
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    violation_items, compliant_items = [], []
    for ann in cache.values():
        img_path = ann.get("image", "")
        if not os.path.exists(img_path):
            continue
        if ann.get("is_compliant"):
            compliant_items.append({
                "path": img_path,
                "gt_boxes": [],
            })
        else:
            violation_items.append({
                "path": img_path,
                "gt_boxes": ann.get("violations", []),   # bbox_1000 格式
                "width": ann.get("width", 0),
                "height": ann.get("height", 0),
            })

    # 截断到 max_count
    violation_items = violation_items[:max_count]
    compliant_items = compliant_items[:max_count]
    return violation_items, compliant_items


# ── JSON 提取 ─────────────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict | None:
    import re
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        d = json.loads(text[start:end])
        if "violation_detection" in d and "violation_detected" not in d:
            d["violation_detected"] = d.pop("violation_detection")
        return d
    except Exception:
        return None


# ── 推理 ──────────────────────────────────────────────────────────────────────
def infer_single(model, processor, family, img_path: str, use_tiled: bool) -> dict | None:
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


# ── IoU（均为 0-1000 千分比，直接比较）────────────────────────────────────────
def _iou(a: list, b: list) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


def compute_iou(pred_boxes: list, gt_violations: list) -> dict:
    """
    pred_boxes: [{"label":..., "bbox":[x0,y0,x1,y1]}]  来自模型输出
    gt_violations: [{"label":..., "bbox_1000":[x0,y0,x1,y1]}]  来自 cache
    """
    gt_norm = [v["bbox_1000"] for v in gt_violations if len(v.get("bbox_1000", [])) == 4]
    pred_norm = [b["bbox"] for b in pred_boxes if len(b.get("bbox", [])) == 4]

    if not gt_norm or not pred_norm:
        return {"mean_iou": 0.0, "matched": 0,
                "total_gt": len(gt_norm), "total_pred": len(pred_norm)}

    matched_ious, used_pred = [], set()
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
    print(f"MODEL_PATH : {MODEL_PATH}")
    if lora:
        print(f"LORA_PATH  : {lora}")
    print(f"数据源     : {CACHE_FILE}")

    violation_items, compliant_items = load_test_data(count)
    print(f"违规测试集 : {len(violation_items)} 张")
    print(f"合规测试集 : {len(compliant_items)} 张")

    if not violation_items and not compliant_items:
        print("[Error] 没有可用测试数据，请检查 annotate_cache.json 和图片路径")
        return

    model, processor, family = load_vlm(MODEL_PATH, lora_path=lora)
    t_start = time.time()

    # ── 违规图测试 ────────────────────────────────────────────────────────────
    print(f"\n[1/2] 违规图测试（召回率 + IoU）...")
    recall_correct = 0
    iou_list = []
    violation_results = []

    for item in tqdm(violation_items):
        t1 = time.time()
        parsed = infer_single(model, processor, family, item["path"], use_tiled)
        elapsed = time.time() - t1

        detected = False
        iou_stat = {"mean_iou": 0.0, "matched": 0,
                    "total_gt": len(item["gt_boxes"]), "total_pred": 0}

        if parsed:
            detected = bool(parsed.get("violation_detected", False))
            if isinstance(detected, str):
                detected = detected.lower() == "true"
            if not detected and parsed.get("violation_boxes"):
                detected = True
            if detected and item["gt_boxes"]:
                iou_stat = compute_iou(
                    parsed.get("violation_boxes", []),
                    item["gt_boxes"],
                )
                if iou_stat["total_gt"] > 0:
                    iou_list.append(iou_stat["mean_iou"])

        if detected:
            recall_correct += 1

        violation_results.append({
            "image": item["path"],
            "detected": detected,
            "detection_result": "GT为违规，检测为违规" if detected else "GT为违规，检测为合规",
            "violation_type": parsed.get("violation_type", "") if parsed else "",
            "pred_labels": [b.get("label") for b in (parsed or {}).get("violation_boxes", [])],
            "gt_labels": [v.get("label") for v in item["gt_boxes"]],
            "iou": iou_stat,
            "elapsed": round(elapsed, 2),
        })

    recall = recall_correct / len(violation_items) if violation_items else 0
    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0

    # ── 合规图测试 ────────────────────────────────────────────────────────────
    print(f"\n[2/2] 合规图测试（精确率）...")
    precision_correct = 0
    compliant_results = []

    for item in tqdm(compliant_items):
        t1 = time.time()
        parsed = infer_single(model, processor, family, item["path"], use_tiled)
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
            "image": item["path"],
            "is_compliant": is_compliant,
            "detection_result": "GT为合规，检测为合规" if is_compliant else "GT为合规，检测为违规",
            "violation_type": parsed.get("violation_type", "") if parsed else "",
            "elapsed": round(elapsed, 2),
        })

    precision = precision_correct / len(compliant_items) if compliant_items else 0
    total_time = time.time() - t_start
    avg_time = total_time / (len(violation_items) + len(compliant_items))

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  评估报告  [{mode} / {infer_mode}]")
    print("=" * 60)
    print(f"  召回率  (Recall)   : {recall*100:.1f}%  ({recall_correct}/{len(violation_items)})")
    print(f"  精确率  (Precision): {precision*100:.1f}%  ({precision_correct}/{len(compliant_items)})")
    print(f"  平均 IoU           : {mean_iou:.3f}  (基于 {len(iou_list)} 张有框图)")
    print(f"  总耗时             : {total_time:.1f}s  (平均 {avg_time:.1f}s/张)")
    print("=" * 60)

    # 漏检列表
    missed = [r for r in violation_results if not r["detected"]]
    if missed:
        print(f"\n漏检 ({len(missed)} 张):")
        for r in missed:
            print(f"  ❌ {r['image']}  GT标签:{r['gt_labels']}")

    # 误报列表
    false_alarms = [r for r in compliant_results if not r["is_compliant"]]
    if false_alarms:
        print(f"\n误报 ({len(false_alarms)} 张):")
        for r in false_alarms:
            print(f"  ❌ {r['image']}  预测类型:{r['violation_type']}")

    # IoU 分布
    if iou_list:
        buckets = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-1.0": 0}
        for v in iou_list:
            if v < 0.3: buckets["0.0-0.3"] += 1
            elif v < 0.5: buckets["0.3-0.5"] += 1
            elif v < 0.7: buckets["0.5-0.7"] += 1
            else: buckets["0.7-1.0"] += 1
        print(f"\nIoU 分布: {buckets}")

    # 保存报告
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "lora" if use_lora else "base"
    out_path = f"{PROJECT_ROOT}/outputs/{prefix}_test_v2_{ts}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "model": mode, "infer_mode": infer_mode,
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "mean_iou": round(mean_iou, 4),
                "total_time": round(total_time, 1),
                "avg_time_per_image": round(avg_time, 2),
            },
            "violation_results": violation_results,
            "compliant_results": compliant_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-only", action="store_true", help="不加载 LoRA，测基座")
    parser.add_argument("--tiled", action="store_true", help="大图启用分块推理")
    parser.add_argument("--count", type=int, default=50, help="每类最多测试张数")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)
    run_test(
        use_lora=not args.base_only,
        use_tiled=args.tiled,
        count=args.count,
    )
