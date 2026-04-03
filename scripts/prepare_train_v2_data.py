"""
清洗 train_balanced/val_balanced 数据，生成更适合 Qwen2.5-VL SFT 的纯 JSON 训练集。
"""

import argparse
import json
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = (
    "你是一名严谨的边坡与基坑施工安全监测专家，负责基于施工现场图片输出可落地的监测结论。\n"
    "你的任务不是泛泛描述图片，而是围绕高坠风险和临边防护进行监测判定，并输出结构化结果。\n"
    "如果图片中危险边缘、人员可达性或防护设施状态看不清，请优先输出 uncertain，不要为了给出结论而过度推断。\n"
    "【场景识别要求】\n"
    "1. 先判断监测对象更接近：foundation_pit、slope、platform_or_edge、mixed、unknown。\n"
    "2. 必须区分边坡坡面本体与需要防护的危险临边。\n"
    "3. 如果只是可见大面积施工坡面，且看不到明确临边作业平台或坠落边缘，不要仅凭是边坡就判为违规。\n"
    "【输出要求】\n"
    "只输出一个 JSON 对象，不要输出 markdown，不要输出代码块，不要输出额外解释。\n"
    "JSON 必须包含字段：scene_type, monitoring_content, monitoring_result, key_observations, risk_points, compliance_status, violation_detected, violation_type, severity, confidence, suggestion。\n"
    "【一致性要求】\n"
    "1. compliance_status=compliant 时，violation_detected=false, violation_type=normal。\n"
    "2. compliance_status=non_compliant 时，violation_detected=true。\n"
    "3. compliance_status=uncertain 时，violation_detected=null, violation_type=uncertain, severity=unknown。"
)

USER_PROMPT = (
    "请对这张施工现场图片执行边坡/基坑安全监测，识别场景类型，提取关键观察，"
    "判断是否存在临边高坠风险及防护缺陷，并严格只输出 JSON 对象。"
)

REQUIRED_JSON_KEYS = {
    "scene_type": "unknown",
    "monitoring_content": "",
    "monitoring_result": "",
    "key_observations": [],
    "risk_points": [],
    "compliance_status": "uncertain",
    "violation_detected": None,
    "violation_type": "uncertain",
    "severity": "unknown",
    "confidence": "low",
    "suggestion": "请结合现场复核图像中危险边缘、防护设施连续性和人员可达性。",
}

SCENE_TYPES = {"foundation_pit", "slope", "platform_or_edge", "mixed", "unknown"}
COMPLIANCE_STATUSES = {"compliant", "non_compliant", "uncertain"}
VIOLATION_TYPES = {
    "normal",
    "no_edge_protection",
    "fence_damaged",
    "guardrail_deformed",
    "warning_missing",
    "unsafe_access",
    "multiple_issues",
    "uncertain",
}
SEVERITIES = {"low", "medium", "high", "critical", "unknown"}
CONFIDENCES = {"low", "medium", "high"}


def extract_json_dict(text: str) -> dict:
    cleaned_text = text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`").strip()
        if cleaned_text.startswith("json"):
            cleaned_text = cleaned_text[4:].strip()
    try:
        return json.loads(cleaned_text)
    except Exception:
        start = cleaned_text.find("{")
        end = cleaned_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("未找到可解析的 JSON 对象")
        return json.loads(cleaned_text[start:end + 1])


def ensure_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def build_default_texts(data: dict) -> tuple[str, str]:
    scene_type = data["scene_type"]
    compliance_status = data["compliance_status"]
    violation_type = data["violation_type"]
    scene_text = {
        "foundation_pit": "基坑区域",
        "slope": "边坡区域",
        "platform_or_edge": "高处平台或临边区域",
        "mixed": "边坡与基坑混合区域",
        "unknown": "施工监测区域",
    }[scene_type]
    if compliance_status == "compliant":
        result = "现场防护措施基本到位，未发现明显临边防护缺陷。"
    elif compliance_status == "non_compliant":
        if violation_type == "multiple_issues":
            result = "现场存在多项临边防护问题，存在较明显安全风险。"
        elif violation_type == "no_edge_protection":
            result = "现场存在缺失临边防护的情况，存在高坠风险。"
        else:
            result = "现场存在临边防护缺陷，存在安全风险。"
    else:
        result = "图像证据不足，当前无法可靠判断是否存在临边防护问题。"
    content = f"{scene_text}的临边防护与高坠风险监测。"
    return content, result


def normalize_record(data: dict) -> dict:
    normalized = dict(REQUIRED_JSON_KEYS)
    normalized.update(data if isinstance(data, dict) else {})

    normalized["scene_type"] = normalized["scene_type"] if normalized["scene_type"] in SCENE_TYPES else "unknown"
    normalized["compliance_status"] = (
        normalized["compliance_status"] if normalized["compliance_status"] in COMPLIANCE_STATUSES else "uncertain"
    )
    normalized["violation_type"] = (
        normalized["violation_type"] if normalized["violation_type"] in VIOLATION_TYPES else "uncertain"
    )
    normalized["severity"] = normalized["severity"] if normalized["severity"] in SEVERITIES else "unknown"
    normalized["confidence"] = normalized["confidence"] if normalized["confidence"] in CONFIDENCES else "low"
    normalized["key_observations"] = ensure_list(normalized.get("key_observations"))
    normalized["risk_points"] = ensure_list(normalized.get("risk_points"))

    if normalized["compliance_status"] == "compliant":
        normalized["violation_detected"] = False
        normalized["violation_type"] = "normal"
        if normalized["severity"] == "unknown":
            normalized["severity"] = "low"
    elif normalized["compliance_status"] == "non_compliant":
        normalized["violation_detected"] = True
        if normalized["violation_type"] in {"normal", "uncertain"}:
            normalized["violation_type"] = "multiple_issues" if len(normalized["risk_points"]) > 1 else "no_edge_protection"
        if normalized["severity"] == "unknown":
            normalized["severity"] = "medium"
    else:
        normalized["violation_detected"] = None
        normalized["violation_type"] = "uncertain"
        normalized["severity"] = "unknown"

    if not str(normalized["monitoring_content"]).strip() or not str(normalized["monitoring_result"]).strip():
        default_content, default_result = build_default_texts(normalized)
        if not str(normalized["monitoring_content"]).strip():
            normalized["monitoring_content"] = default_content
        if not str(normalized["monitoring_result"]).strip():
            normalized["monitoring_result"] = default_result

    if not str(normalized["suggestion"]).strip():
        normalized["suggestion"] = REQUIRED_JSON_KEYS["suggestion"]

    return normalized


def convert_item(item: dict) -> tuple[dict, bool]:
    assistant_text = item["messages"][-1]["content"]
    raw_data = extract_json_dict(assistant_text)
    normalized = normalize_record(raw_data)
    was_changed = normalized != raw_data or assistant_text.strip().startswith("【") or "```json" in assistant_text
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<image>{USER_PROMPT}"},
            {"role": "assistant", "content": json.dumps(normalized, ensure_ascii=False)},
        ],
        "images": item["images"],
    }, was_changed


def process_file(input_path: Path, output_path: Path) -> None:
    records = []
    stats = Counter()
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            converted, was_changed = convert_item(item)
            stats["total"] += 1
            if was_changed:
                stats["changed"] += 1
            label = json.loads(converted["messages"][-1]["content"])
            stats[f"status::{label['compliance_status']}"] += 1
            stats[f"scene::{label['scene_type']}"] += 1
            records.append(converted)

    with output_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{input_path.name} -> {output_path.name}")
    print(f"  total={stats['total']} changed={stats['changed']}")
    print(
        "  compliance_status:",
        {
            "compliant": stats["status::compliant"],
            "non_compliant": stats["status::non_compliant"],
            "uncertain": stats["status::uncertain"],
        },
    )
    print(
        "  scene_type:",
        {
            "foundation_pit": stats["scene::foundation_pit"],
            "slope": stats["scene::slope"],
            "platform_or_edge": stats["scene::platform_or_edge"],
            "mixed": stats["scene::mixed"],
            "unknown": stats["scene::unknown"],
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/home/fs-ai/llama-qwen/data"))
    parser.add_argument("--train-name", type=str, default="train_balanced.jsonl")
    parser.add_argument("--val-name", type=str, default="val_balanced.jsonl")
    parser.add_argument("--train-output", type=str, default="train_balanced_clean.jsonl")
    parser.add_argument("--val-output", type=str, default="val_balanced_clean.jsonl")
    args = parser.parse_args()

    input_dir = args.input_dir
    process_file(input_dir / args.train_name, input_dir / args.train_output)
    process_file(input_dir / args.val_name, input_dir / args.val_output)


if __name__ == "__main__":
    main()
