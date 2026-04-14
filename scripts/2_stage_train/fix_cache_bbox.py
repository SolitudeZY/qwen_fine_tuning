"""
离线清洗 annotate_cache.json 中的越界 bbox_1000
- clamp 到 0-1000
- 修正 x0>x1 / y0>y1
- 过滤过小框（边长 < 10）
用法：python scripts/2_stage_train/fix_cache_bbox.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
CACHE_FILE = ROOT / "data" / "annotate_cache.json"


def norm_bbox(b):
    x0, y0, x1, y1 = [max(0, min(1000, c)) for c in b]
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def is_dirty(b):
    return any(v < 0 or v > 1000 for v in b) or b[0] > b[2] or b[1] > b[3]


cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

total_boxes = fixed = dropped = 0

for key, ann in cache.items():
    cleaned = []
    for v in ann.get("violations", []):
        b = v.get("bbox_1000", [])
        if len(b) != 4:
            dropped += 1
            continue
        total_boxes += 1
        if is_dirty(b):
            fixed += 1
        nb = norm_bbox(b)
        if nb[2] - nb[0] < 10 or nb[3] - nb[1] < 10:
            dropped += 1
            continue
        v["bbox_1000"] = nb
        cleaned.append(v)
    ann["violations"] = cleaned

CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"处理 {len(cache)} 条标注，共 {total_boxes} 个框")
print(f"修正越界框：{fixed}，过滤过小框：{dropped}")
print("cache 已更新")
