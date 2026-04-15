"""
将 LabelMe 标注数据转换为 ms-swift Qwen2.5-VL 微调所需的对话格式。

针对边防护违规识别场景：
- 有临边防护标注 → 标注了防护设施的位置，说明有防护
- 无临边防护标注但有基坑/高处作业相关标注 → 可能存在防护缺失
- 整体场景描述 + 违规判定
"""

import json
import glob
import os
import random
from pathlib import Path
from collections import Counter

# ============ 配置 ============
DATASET_DIR = "/home/fs-ai/llama-qwen/dataset"
OUTPUT_DIR = "/home/fs-ai/llama-qwen/data"
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
TEST_RATIO = 0.05
SEED = 42

# 边防护相关标签
EDGE_PROTECTION_LABELS = {"临边防护", "临边防护网片", "钢管护栏", "钻孔灌注桩孔口防护", "安全围挡", "基坑围挡", "桥台围挡"}

# 高风险区域标签（需要有防护的区域）
HIGH_RISK_LABELS = {"基坑", "桩基孔口", "钻孔灌注桩孔口", "孔口井盖", "钻孔灌注桩"}

# 系统提示
SYSTEM_PROMPT = (
    "你是一个专业的工业安全检查AI助手，专注于施工现场的边防护设施安全检查。"
    "你需要分析图片中的施工现场，重点关注临边防护设施（围栏、护栏、防护网、安全围挡等）的完整性和合规性。"
    "请以JSON格式输出检查结果。"
)

# 用户提问模板（随机选择增加多样性）
USER_QUERIES = [
    "请检查这张施工现场图片中的边防护设施是否合规，是否存在围栏缺失、防护不全等违规情况。",
    "请对这张图片进行边防护安全检查，分析临边防护设施的状态。",
    "检查图片中的施工现场，判断边防护措施是否到位，是否存在安全隐患。",
    "请分析这张施工现场照片的边防护设施情况，输出安全检查结果。",
    "对图片中的施工区域进行边防护违规检测，识别是否存在防护缺失问题。",
]


def parse_annotation(json_path: str) -> dict:
    """解析 LabelMe 标注文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    labels = [s["label"] for s in shapes]
    label_counts = Counter(labels)

    # 图片路径
    image_name = data.get("imagePath", "")
    image_dir = os.path.dirname(json_path)
    image_path = os.path.join(image_dir, image_name)

    # 统计边防护相关标注
    edge_protections = []
    high_risk_areas = []
    other_objects = []

    for shape in shapes:
        label = shape["label"]
        points = shape["points"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = {
            "x_min": round(min(xs)),
            "y_min": round(min(ys)),
            "x_max": round(max(xs)),
            "y_max": round(max(ys)),
        }

        if label in EDGE_PROTECTION_LABELS:
            edge_protections.append({"label": label, "bbox": bbox})
        elif label in HIGH_RISK_LABELS:
            high_risk_areas.append({"label": label, "bbox": bbox})
        else:
            other_objects.append(label)

    return {
        "image_path": image_path,
        "image_name": image_name,
        "edge_protections": edge_protections,
        "high_risk_areas": high_risk_areas,
        "other_objects": list(set(other_objects)),
        "all_labels": label_counts,
    }


def generate_response(parsed: dict) -> str:
    """根据标注信息生成检查结果 JSON"""
    has_protection = len(parsed["edge_protections"]) > 0
    has_risk_areas = len(parsed["high_risk_areas"]) > 0
    protection_count = len(parsed["edge_protections"])

    # 构建场景描述
    all_objs = parsed["other_objects"]
    scene_elements = []
    if "液压起重机" in all_objs or "起重机" in all_objs:
        scene_elements.append("起重机械")
    if "人员" in all_objs:
        scene_elements.append("施工人员")
    if "挖掘机" in all_objs or "装载机" in all_objs:
        scene_elements.append("施工机械")
    if any("墩" in o for o in all_objs):
        scene_elements.append("桥墩结构")
    if any("梁" in o for o in all_objs):
        scene_elements.append("梁体结构")

    scene_desc = "、".join(scene_elements) if scene_elements else "施工设备和设施"

    # 构建防护描述
    protection_labels = [p["label"] for p in parsed["edge_protections"]]
    protection_types = list(set(protection_labels))
    protection_desc = "、".join(protection_types) if protection_types else ""

    if has_protection and not has_risk_areas:
        # 有防护设施，无明显高风险区域暴露 → 合规
        result = {
            "violation_detected": False,
            "violation_type": "normal",
            "severity": "none",
            "edge_protection_count": protection_count,
            "edge_protection_types": protection_types,
            "description": f"施工现场可见{scene_desc}，已设置{protection_desc}共{protection_count}处，边防护设施基本到位。",
            "suggestion": "继续保持现有防护措施，定期检查防护设施的完整性和稳固性。",
        }
    elif has_protection and has_risk_areas:
        # 有防护也有高风险区 → 需进一步检查
        risk_labels = list(set([r["label"] for r in parsed["high_risk_areas"]]))
        risk_desc = "、".join(risk_labels)
        result = {
            "violation_detected": False,
            "violation_type": "normal",
            "severity": "low",
            "edge_protection_count": protection_count,
            "edge_protection_types": protection_types,
            "risk_areas": risk_labels,
            "description": f"施工现场可见{scene_desc}，已设置{protection_desc}共{protection_count}处。现场存在{risk_desc}等高风险区域，目前已有防护措施覆盖。",
            "suggestion": f"注意检查{risk_desc}周围的防护设施是否完整牢固，确保防护范围覆盖所有风险区域。",
        }
    elif not has_protection and has_risk_areas:
        # 无防护但有高风险区域 → 违规
        risk_labels = list(set([r["label"] for r in parsed["high_risk_areas"]]))
        risk_desc = "、".join(risk_labels)
        result = {
            "violation_detected": True,
            "violation_type": "no_edge_protection",
            "severity": "high",
            "edge_protection_count": 0,
            "risk_areas": risk_labels,
            "description": f"施工现场可见{scene_desc}，存在{risk_desc}等高风险区域，但未发现明确的临边防护设施，存在较大安全隐患。",
            "suggestion": f"需要立即在{risk_desc}周围设置临边防护栏杆、安全网或围挡，防止人员坠落或误入危险区域。",
        }
    else:
        # 无防护也无高风险区 → 需关注但非严重
        result = {
            "violation_detected": False,
            "violation_type": "normal",
            "severity": "none",
            "edge_protection_count": 0,
            "description": f"施工现场可见{scene_desc}，当前画面中未发现需要临边防护的高风险区域。",
            "suggestion": "当前区域无明显临边防护需求，建议关注其他安全事项。",
        }

    return json.dumps(result, ensure_ascii=False, indent=2)


def build_conversation(parsed: dict) -> dict:
    """构建 ms-swift 4.x 对话格式（content 必须为字符串，图片用 <image> 占位符）"""
    query = random.choice(USER_QUERIES)
    response = generate_response(parsed)
    abs_image_path = os.path.abspath(parsed["image_path"])

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{query}"},
            {"role": "assistant", "content": response},
        ],
        "images": [abs_image_path],
    }


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 收集所有标注文件
    json_files = sorted(glob.glob(f"{DATASET_DIR}/**/*.json", recursive=True))
    print(f"Found {len(json_files)} annotation files")

    # 解析并构建对话
    conversations = []
    stats = {"total": 0, "with_protection": 0, "with_risk": 0, "violation": 0}

    for jf in json_files:
        parsed = parse_annotation(jf)

        # 检查图片是否存在
        if not os.path.exists(parsed["image_path"]):
            print(f"  WARNING: Image not found: {parsed['image_path']}")
            continue

        conv = build_conversation(parsed)
        conversations.append(conv)

        # 统计
        stats["total"] += 1
        if parsed["edge_protections"]:
            stats["with_protection"] += 1
        if parsed["high_risk_areas"]:
            stats["with_risk"] += 1
        resp = json.loads(conv["messages"][2]["content"])
        if resp["violation_detected"]:
            stats["violation"] += 1

    print(f"\n=== Dataset Statistics ===")
    print(f"Total samples: {stats['total']}")
    print(f"With edge protection: {stats['with_protection']}")
    print(f"With high-risk areas: {stats['with_risk']}")
    print(f"Violations detected: {stats['violation']}")

    # 打乱并划分数据集
    random.shuffle(conversations)
    n = len(conversations)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_data = conversations[:n_train]
    val_data = conversations[n_train : n_train + n_val]
    test_data = conversations[n_train + n_val :]

    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 保存
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {output_path} ({len(data)} samples)")

    # 额外生成 ms-swift jsonl 格式（每行一个 JSON）
    for name, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        output_path = os.path.join(OUTPUT_DIR, f"{name}.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
