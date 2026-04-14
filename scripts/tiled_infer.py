"""
大图分块推理工具（滑动窗口）

将高分辨率图片（如 DJI 4032×3024）切成 2×2 重叠子图分别推理，
再将各子图的 violation_boxes 坐标映射回原图坐标并去重合并。

用法：
    from tiled_infer import tiled_chat

    result = tiled_chat(model, processor, model_family, image_path)
    # result: {"violation_detected": bool, "violation_type": str,
    #          "severity": str, "suggestion": str, "violation_boxes": [...]}
"""

import os
from PIL import Image


# 子图网格：2 行 × 2 列，重叠比例 20%
TILE_ROWS = 2
TILE_COLS = 2
OVERLAP = 0.20
# IoU 阈值：两个框重叠超过此值则合并（去重）
IOU_THRESHOLD = 0.3


def _tile_image(image_path: str) -> list[dict]:
    """
    将原图切成 TILE_ROWS × TILE_COLS 个重叠子图。
    返回列表，每项包含：
        pil_crop  : PIL.Image 子图
        x_offset  : 子图左上角在原图中的 x 像素坐标
        y_offset  : 子图左上角在原图中的 y 像素坐标
        crop_w    : 子图宽度（像素）
        crop_h    : 子图高度（像素）
    """
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

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
            crop = img.crop((x0, y0, x1, y1))
            tiles.append({
                "pil_crop": crop,
                "x_offset": x0,
                "y_offset": y0,
                "crop_w": x1 - x0,
                "crop_h": y1 - y0,
            })
    return tiles, W, H


def _box_norm1000_to_pixel(bbox: list, w: int, h: int) -> list:
    """千分比坐标 → 像素坐标"""
    return [
        int(bbox[0] / 1000 * w),
        int(bbox[1] / 1000 * h),
        int(bbox[2] / 1000 * w),
        int(bbox[3] / 1000 * h),
    ]


def _box_pixel_to_norm1000(bbox: list, w: int, h: int) -> list:
    """像素坐标 → 千分比坐标"""
    return [
        int(bbox[0] / w * 1000),
        int(bbox[1] / h * 1000),
        int(bbox[2] / w * 1000),
        int(bbox[3] / h * 1000),
    ]


def _iou(a: list, b: list) -> float:
    """计算两个像素坐标框的 IoU"""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _merge_boxes(all_boxes_pixel: list, orig_w: int, orig_h: int) -> list:
    """
    对所有子图映射回原图的像素坐标框做 NMS 去重，
    返回千分比坐标格式的框列表。
    """
    if not all_boxes_pixel:
        return []

    kept = []
    for box in all_boxes_pixel:
        merged = False
        for k in kept:
            if k["label"] == box["label"] and _iou(k["bbox_px"], box["bbox_px"]) > IOU_THRESHOLD:
                # 合并：取两框的外接矩形
                k["bbox_px"] = [
                    min(k["bbox_px"][0], box["bbox_px"][0]),
                    min(k["bbox_px"][1], box["bbox_px"][1]),
                    max(k["bbox_px"][2], box["bbox_px"][2]),
                    max(k["bbox_px"][3], box["bbox_px"][3]),
                ]
                merged = True
                break
        if not merged:
            kept.append({"label": box["label"], "bbox_px": box["bbox_px"]})

    # 转回千分比
    result = []
    for k in kept:
        result.append({
            "label": k["label"],
            "bbox": _box_pixel_to_norm1000(k["bbox_px"], orig_w, orig_h),
        })
    return result


def tiled_chat(
    model,
    processor,
    model_family: str,
    image_path: str,
    infer_fn,
    system_prompt: str,
    query: str,
    extract_fn,
    tmp_dir: str = "/tmp/tiled_infer",
) -> dict:
    """
    对大图做「全图低分辨率 + 分块高分辨率」双路推理，合并结果。

    全图推理：能看到整体布局，识别大面积缺失区域
    分块推理：能看到细节，识别小目标断口

    参数：
        infer_fn    : infer_vlm 函数引用
        system_prompt / query : 推理用的 prompt
        extract_fn  : _extract_json_object 函数引用
        tmp_dir     : 子图临时保存目录

    返回：合并后的结果 dict（与单图推理格式相同）
    """
    import os
    os.makedirs(tmp_dir, exist_ok=True)

    tiles, orig_w, orig_h = _tile_image(image_path)

    all_boxes_pixel = []
    violation_detected = False
    violation_types = []
    severities = []
    suggestions = []

    severity_rank = {"无": 0, "低": 1, "low": 1, "medium": 2, "中": 2, "high": 3, "高": 3, "critical": 4, "极高": 4}

    def _collect_result(parsed, tile=None):
        """从单次推理结果中收集违规信息，tile=None 表示全图推理"""
        nonlocal violation_detected
        if not parsed:
            return
        if "violation_detection" in parsed and "violation_detected" not in parsed:
            parsed["violation_detected"] = parsed.pop("violation_detection")
        if not parsed.get("violation_detected"):
            return

        violation_detected = True
        vt = parsed.get("violation_type", "")
        if isinstance(vt, list):
            vt = "、".join(str(v) for v in vt)
        if vt:
            violation_types.append(vt)
        sev = parsed.get("severity", "")
        if sev:
            severities.append(sev)
        sug = parsed.get("suggestion", "")
        if sug:
            suggestions.append(sug)

        for box in parsed.get("violation_boxes", []):
            bbox_norm = box.get("bbox", [])
            if len(bbox_norm) != 4:
                continue
            if tile is None:
                bx0 = int(bbox_norm[0] / 1000 * orig_w)
                by0 = int(bbox_norm[1] / 1000 * orig_h)
                bx1 = int(bbox_norm[2] / 1000 * orig_w)
                by1 = int(bbox_norm[3] / 1000 * orig_h)
            else:
                bx0 = int(bbox_norm[0] / 1000 * tile["crop_w"]) + tile["x_offset"]
                by0 = int(bbox_norm[1] / 1000 * tile["crop_h"]) + tile["y_offset"]
                bx1 = int(bbox_norm[2] / 1000 * tile["crop_w"]) + tile["x_offset"]
                by1 = int(bbox_norm[3] / 1000 * tile["crop_h"]) + tile["y_offset"]

            bx0 = max(0, min(bx0, orig_w))
            by0 = max(0, min(by0, orig_h))
            bx1 = max(0, min(bx1, orig_w))
            by1 = max(0, min(by1, orig_h))

            label = box.get("label", "违规区域")
            # 启发式修正：框面积超过原图 8% 时，"围栏断口" 更可能是大面积缺失
            box_area = (bx1 - bx0) * (by1 - by0)
            orig_area = orig_w * orig_h
            if label == "围栏断口" and box_area > orig_area * 0.08:
                label = "临边防护缺失"

            all_boxes_pixel.append({
                "label": label,
                "bbox_px": [bx0, by0, bx1, by1],
            })

    # ── 路径1：全图低分辨率推理（识别大面积缺失）──
    print("    [全图推理] 识别整体布局...")
    full_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": query},
            ],
        },
    ]
    full_response = infer_fn(model, processor, model_family, full_messages, max_new_tokens=1024)
    full_parsed, _, _ = extract_fn(full_response)
    _collect_result(full_parsed, tile=None)

    # ── 路径2：分块高分辨率推理（识别小目标断口）──
    print("    [分块推理] 识别细节目标...")
    for i, tile in enumerate(tiles):
        # 保存子图到临时文件
        tmp_path = os.path.join(tmp_dir, f"tile_{i}.jpg")
        tile["pil_crop"].save(tmp_path, quality=95)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(tmp_path)}"},
                    {"type": "text", "text": query},
                ],
            },
        ]

        response = infer_fn(model, processor, model_family, messages, max_new_tokens=1024)
        parsed, _, _ = extract_fn(response)
        _collect_result(parsed, tile=tile)

    # 合并去重
    merged_boxes = _merge_boxes(all_boxes_pixel, orig_w, orig_h)

    # 取最高 severity
    best_severity = "无"
    for s in severities:
        if severity_rank.get(s, 0) > severity_rank.get(best_severity, 0):
            best_severity = s

    return {
        "violation_detected": violation_detected,
        "violation_type": "、".join(list(dict.fromkeys(str(v) for v in violation_types))) if violation_types else "",
        "severity": best_severity,
        "suggestion": suggestions[0] if suggestions else "",
        "violation_boxes": merged_boxes,
    }
