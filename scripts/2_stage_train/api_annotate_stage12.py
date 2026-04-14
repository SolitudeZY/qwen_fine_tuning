"""
API 打标脚本：Qwen-VL-Max → canonical annotation → Stage1 + Stage2 JSONL

流程：
  1. 读取 data/unique_images.txt
  2. 对每张图调用 Qwen-VL-Max，生成 canonical annotation（中间真值结构）
  3. 断点续传：已标注的写入 data/annotate_cache.json，跳过重复
  4. 失败/低置信度：写入 data/annotate_skip.json
  5. 全部完成后派生：
       data/stage1_grounding.jsonl   — ms-swift grounding 格式（像素坐标）
       data/stage2_json.jsonl        — ms-swift JSON 精调格式（0-1000 千分比）

用法：
  export DASHSCOPE_API_KEY=your_key
  python scripts/2_stage_train/api_annotate_stage12.py [--threshold 0.5] [--dry-run]
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
UNIQUE_IMAGES_TXT = ROOT / "data" / "unique_images.txt"
CACHE_FILE = ROOT / "data" / "annotate_cache.json"
SKIP_FILE = ROOT / "data" / "annotate_skip.json"
STAGE1_OUT = ROOT / "data" / "stage1_grounding.jsonl"
STAGE2_OUT = ROOT / "data" / "stage2_json.jsonl"

# ── 标注协议 ──────────────────────────────────────────────────────────────────
VALID_LABELS = {
    "围栏断口",        # 围栏本体存在破损或缺口
    "围栏倒伏",        # 围栏整体或局部倒塌
    "临边防护缺失",    # 危险临边完全没有任何防护
    "临边开口未防护",  # 出入口/开口处无防护门或警示隔离
}
VALID_SCENES = {"基坑", "边坡", "泥浆池", "平台临边", "其他"}

SYSTEM_PROMPT = """你是施工现场安全监测专家。请分析图片，输出严格的 JSON，不要输出任何其他文字。

图片来源：无人机俯拍施工现场。

输出格式：
{
  "scene": "基坑|边坡|泥浆池|平台临边|其他",
  "is_compliant": true/false,
  "confidence": "high|medium|low",
  "violations": [
    {
      "label": "围栏断口|围栏倒伏|临边防护缺失|临边开口未防护",
      "bbox_1000": [x_min, y_min, x_max, y_max],
      "severity": "低|中|高",
      "evidence": "简短描述违规证据，说明从俯拍视角看到了什么"
    }
  ],
  "global_conclusion": {
    "violation_detected": true/false,
    "violation_type": "违规类型，多类型用顿号连接，合规时为空字符串",
    "severity": "低|中|高|无",
    "suggestion": "整改建议，合规时写巡检建议"
  }
}

标签定义（只能用这四个）：
- 围栏断口：围栏本体存在破损或缺口，但周围仍有围栏结构
- 围栏倒伏：围栏整体或局部倒塌，从俯拍可见围栏不再直立
- 临边防护缺失：危险临边完全没有任何围栏或防护结构
- 临边开口未防护：施工出入口或开口处缺少防护门、警示带等隔离措施

规则：
- is_compliant=true 时 violations 必须为空数组
- is_compliant=false 时 violations 必须非空
- bbox_1000 是 0-1000 千分比坐标 [x_min, y_min, x_max, y_max]
- 边坡坡面本体不是违规，只有危险临边才需要防护
- 无法从俯拍视角判断的缺陷（如防护高度、踢脚板）不要标注
- 证据不足时 confidence 填 low，is_compliant 填 true"""

USER_PROMPT = "请对这张施工现场图片进行安全监测，输出 JSON 结果。"

STAGE2_SYSTEM = """你是一名严谨的边坡与基坑施工安全监测专家，负责基于施工现场图片输出可落地的监测结论。
你的任务不是泛泛描述图片，而是围绕高坠风险和临边防护进行监测判定，并输出结构化的JSON结果。
【场景识别要求】：
- 基坑：向下开挖形成的坑槽，坑边通常需要防护隔离。
- 边坡：倾斜土坡、岩坡本身通常不是必须全封闭围挡的对象。但坡顶作业平台、临空边若存在坠落风险，仍需防护。
- 如果只是可见大面积施工坡面，不要仅凭'是边坡'或'是基坑'就判为违规。
【违规标签说明】：
- 围栏断口：围栏本体存在破损或缺口
- 围栏倒伏：围栏整体或局部倒塌
- 临边防护缺失：危险临边完全没有任何防护结构
- 临边开口未防护：出入口或开口处缺少防护门、警示带等隔离措施
【输出要求】：
只输出一个 JSON 对象，字段包含：violation_detected(布尔), violation_type(字符串), severity(字符串), suggestion(字符串)。
如果检测到违规(violation_detected=true)，JSON 中必须包含 violation_boxes 数组，每个元素包含：
  label(违规简述), bbox([x_min, y_min, x_max, y_max]，千分比坐标，范围0-1000)。
如果合规(violation_detected=false)，violation_boxes 为空数组 []。"""

# ─────────────────────────────────────────────────────────────────────────────


def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_skip() -> list:
    if SKIP_FILE.exists():
        with open(SKIP_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_skip(skip: list):
    with open(SKIP_FILE, "w", encoding="utf-8") as f:
        json.dump(skip, f, ensure_ascii=False, indent=2)


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_qwen_vl_max(image_path: Path, api_key: str, retries: int = 3) -> tuple[str, str]:
    """返回 (raw_response_text, error_msg)，失败时 raw_response_text 为空。"""
    import urllib.request
    import urllib.error

    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    b64 = image_to_base64(image_path)
    data_url = f"data:{mime};base64,{b64}"

    payload = json.dumps({
        "model": "qwen-vl-max",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "max_tokens": 1024,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                return result["choices"][0]["message"]["content"], ""
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "", str(e)
    return "", "max retries exceeded"


def parse_canonical(raw: str, image_path: Path, api_key: str) -> tuple[dict | None, str]:
    """从 API 响应中解析 canonical annotation，返回 (annotation, error)。"""
    # 提取 JSON（模型可能包裹在 ```json ... ``` 里）
    text = raw.strip()
    if "```" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        text = text[start:end] if start != -1 else text

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

    # 基本字段校验
    required = {"scene", "is_compliant", "confidence", "violations", "global_conclusion"}
    missing = required - data.keys()
    if missing:
        return None, f"missing fields: {missing}"

    # 标签合法性校验
    for v in data.get("violations", []):
        if v.get("label") not in VALID_LABELS:
            v["label"] = "临边防护缺失"  # 降级处理

    # 一致性校验：violation_detected ↔ violations
    gc = data["global_conclusion"]
    has_violations = len(data["violations"]) > 0
    if gc.get("violation_detected") and not has_violations:
        return None, "inconsistent: violation_detected=true but violations=[]"
    if not gc.get("violation_detected") and has_violations:
        data["violations"] = []  # 以 global_conclusion 为准

    img = Image.open(image_path)
    annotation = {
        "image": str(image_path.resolve()),
        "width": img.width,
        "height": img.height,
        "scene": data.get("scene", "其他"),
        "is_compliant": data["is_compliant"],
        "confidence": data.get("confidence", "medium"),
        "violations": data["violations"],
        "global_conclusion": gc,
        "raw_response": raw,
        "model": "qwen-vl-max",
    }
    return annotation, ""


def to_stage1_sample(ann: dict) -> list[dict]:
    """一张图可能有多个违规框，每个框生成一条 Stage 1 样本。"""
    if not ann["violations"]:
        return []

    samples = []
    w, h = ann["width"], ann["height"]

    # 按 label 分组，同 label 的框合并为一条样本
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for v in ann["violations"]:
        groups[v["label"]].append(v["bbox_1000"])

    prompts = {
        "围栏断口": ["请定位图中围栏断口的位置", "找出图中围栏开口或缺口区域", "标注图中围栏断裂处"],
        "围栏倒伏": ["请定位图中倒伏的围栏", "找出图中倒塌的围栏区域"],
        "临边防护缺失": ["请定位图中临边防护缺失的区域", "找出图中缺少防护的临边位置", "标注图中高坠风险区域"],
        "临边开口未防护": ["请定位图中未防护的临边开口", "找出图中缺少防护门或隔离措施的开口位置"],
    }
    import random

    for label, bboxes_1000 in groups.items():
        # 0-1000 → 像素坐标
        pixel_bboxes = []
        for b in bboxes_1000:
            x0 = int(b[0] / 1000 * w)
            y0 = int(b[1] / 1000 * h)
            x1 = int(b[2] / 1000 * w)
            y1 = int(b[3] / 1000 * h)
            pixel_bboxes.append([x0, y0, x1, y1])

        prompt = random.choice(prompts.get(label, ["请定位图中违规区域"]))
        sample = {
            "messages": [
                {"role": "user", "content": f"<image>{prompt}"},
                {"role": "assistant", "content": "<ref-object><bbox>"},
            ],
            "images": [ann["image"]],
            "objects": {
                "ref": [label] * len(pixel_bboxes),
                "bbox": pixel_bboxes,
                "bbox_type": "real",
                "width": [w] * len(pixel_bboxes),
                "height": [h] * len(pixel_bboxes),
            },
        }
        samples.append(sample)
    return samples


def to_stage2_sample(ann: dict) -> dict:
    """生成 Stage 2 JSON 精调样本。"""
    gc = ann["global_conclusion"]
    assistant_json = {
        "violation_detected": gc["violation_detected"],
        "violation_type": gc.get("violation_type", ""),
        "severity": gc.get("severity", "无"),
        "suggestion": gc.get("suggestion", ""),
        "violation_boxes": [
            {"label": v["label"], "bbox": v["bbox_1000"]}
            for v in ann["violations"]
        ],
    }
    return {
        "messages": [
            {"role": "system", "content": STAGE2_SYSTEM},
            {"role": "user", "content": "<image>请对这张施工现场图片执行安全监测，判断是否存在临边高坠风险及防护缺陷，输出 JSON 结果。"},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)},
        ],
        "images": [ann["image"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="低于此置信度跳过（high=1, medium=0.5, low=0）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印，不调用 API，不写文件")
    parser.add_argument("--derive-only", action="store_true",
                        help="跳过 API 调用，直接从 cache 派生 JSONL")
    args = parser.parse_args()

    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key and not args.dry_run and not args.derive_only:
        print("错误：请设置 DASHSCOPE_API_KEY 环境变量")
        sys.exit(1)

    if not UNIQUE_IMAGES_TXT.exists():
        print(f"错误：{UNIQUE_IMAGES_TXT} 不存在，请先运行 dedup_dataset.py")
        sys.exit(1)

    images = [ROOT / p.strip() for p in UNIQUE_IMAGES_TXT.read_text().splitlines() if p.strip()]
    print(f"待标注图片：{len(images)} 张")

    cache = load_cache()
    skip = load_skip()
    skip_keys = {s["image"] for s in skip}

    # ── API 打标 ──────────────────────────────────────────────────────────────
    if not args.derive_only:
        conf_score = {"high": 1.0, "medium": 0.5, "low": 0.0}
        todo = [p for p in images if str(p.resolve()) not in cache and str(p.resolve()) not in skip_keys]
        print(f"已缓存：{len(cache)}，跳过列表：{len(skip)}，本次需标注：{len(todo)}")

        for i, img_path in enumerate(todo):
            print(f"[{i+1}/{len(todo)}] {img_path.name} ...", end=" ", flush=True)

            if args.dry_run:
                print("(dry-run 跳过)")
                continue

            raw, err = call_qwen_vl_max(img_path, api_key)
            if err:
                print(f"API 失败: {err}")
                skip.append({"image": str(img_path.resolve()), "reason": f"api_error: {err}"})
                save_skip(skip)
                continue

            ann, err = parse_canonical(raw, img_path, api_key)
            if err:
                print(f"解析失败: {err}")
                skip.append({"image": str(img_path.resolve()), "reason": f"parse_error: {err}", "raw": raw})
                save_skip(skip)
                continue

            # 低置信度处理
            score = conf_score.get(ann["confidence"], 0.5)
            if score < args.threshold:
                print(f"低置信度({ann['confidence']})，跳过")
                skip.append({"image": str(img_path.resolve()), "reason": f"low_confidence: {ann['confidence']}", "annotation": ann})
                save_skip(skip)
                continue

            cache[str(img_path.resolve())] = ann
            save_cache(cache)
            status = "违规" if not ann["is_compliant"] else "合规"
            conf = ann["confidence"]
            print(f"OK [{status}, confidence={conf}]")

            time.sleep(0.3)  # 避免限速

    # ── 派生 JSONL ────────────────────────────────────────────────────────────
    print(f"\n从 {len(cache)} 条 canonical annotation 派生 JSONL ...")

    stage1_samples = []
    stage2_samples = []

    for ann in cache.values():
        s1 = to_stage1_sample(ann)
        stage1_samples.extend(s1)
        stage2_samples.append(to_stage2_sample(ann))

    if not args.dry_run:
        with open(STAGE1_OUT, "w", encoding="utf-8") as f:
            for s in stage1_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        with open(STAGE2_OUT, "w", encoding="utf-8") as f:
            for s in stage2_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Stage 1 grounding: {len(stage1_samples)} 条 → {STAGE1_OUT}")
    print(f"Stage 2 JSON:      {len(stage2_samples)} 条 → {STAGE2_OUT}")
    print(f"跳过列表:          {len(skip)} 条 → {SKIP_FILE}")

    # 低置信度统计
    low_conf = [a for a in cache.values() if a.get("confidence") == "low"]
    if low_conf:
        print(f"\n低置信度（需人工复查）：{len(low_conf)} 条")
        for a in low_conf:
            print(f"  {a['image']}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
