"""
API 打标脚本（NVIDIA 版）：通过 NVIDIA API 调用 Qwen3.5 进行施工安全标注

注意：qwen/qwen3.5-397b-a17b 是纯文本 MoE 模型，可能不支持图片输入。
     若 API 返回不支持图片的错误，请更换 --model 为 NVIDIA 上的视觉模型。

流程与 api_annotate_stage12.py 相同：
  1. 读取 data/unique_images.txt
  2. 对每张图调用 NVIDIA API
  3. 断点续传：已标注写入 data/annotate_cache_nvidia.json
  4. 失败/低置信度写入 data/annotate_skip_nvidia.json
  5. 派生 Stage1 + Stage2 JSONL

用法：
  export NVIDIA_API_KEY=nvapi-xxx
  python scripts/2_stage_train/api_annotate_nvidia.py [--model MODEL] [--derive-only]
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image

# ── 路径配置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
UNIQUE_IMAGES_TXT = ROOT / "data" / "unique_images.txt"
CACHE_FILE = ROOT / "data" / "annotate_cache_nvidia.json"
SKIP_FILE = ROOT / "data" / "annotate_skip_nvidia.json"
STAGE1_OUT = ROOT / "data" / "stage1_grounding_nvidia.jsonl"
STAGE2_OUT = ROOT / "data" / "stage2_json_nvidia.jsonl"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "2_stage_train"))
from prompts import SYSTEM_PROMPT as STAGE2_SYSTEM, VALID_LABELS
from api_annotate_stage12 import (
    parse_canonical,
    to_stage1_sample,
    to_stage2_sample,
    image_to_base64,
    SYSTEM_PROMPT,
    USER_PROMPT,
)

# ── NVIDIA API 配置 ──────────────────────────────────────────────────────────
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.5-397b-a17b"


# ── 缓存 ─────────────────────────────────────────────────────────────────────

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_skip() -> list:
    if SKIP_FILE.exists():
        with open(SKIP_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_skip(skip: list) -> None:
    with open(SKIP_FILE, "w", encoding="utf-8") as f:
        json.dump(skip, f, ensure_ascii=False, indent=2)


def strip_thinking(text: str) -> str:
    """去除 <think>...</think> 思考过程，只保留最终输出。"""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── NVIDIA API 调用 ──────────────────────────────────────────────────────────

def call_nvidia_api(
    image_path: Path,
    api_key: str,
    model: str = DEFAULT_MODEL,
    enable_thinking: bool = True,
    retries: int = 3,
) -> tuple[str, str]:
    """调用 NVIDIA API，返回 (response_text, error_msg)。"""
    import requests

    ext = image_path.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    b64 = image_to_base64(image_path)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    # 构建 user content — 图片 + 文本
    user_content: list[dict] = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": USER_PROMPT},
    ]

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 4096,
        "temperature": 0.6,
        "top_p": 0.95,
        "stream": False,
    }
    if enable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": True}

    for attempt in range(retries):
        try:
            resp = requests.post(
                NVIDIA_URL, headers=headers, json=payload, timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            msg = result["choices"][0]["message"]
            # 优先取 content，忽略 reasoning_content（如有）
            content = msg.get("content", "")
            content = strip_thinking(content)
            return content, ""
        except requests.exceptions.HTTPError as e:
            body = ""
            if e.response is not None:
                body = e.response.text[:500]
            err = f"HTTP {e.response.status_code if e.response else '?'}: {body}"
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return "", err
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                return "", str(e)
    return "", "max retries exceeded"



# ── 主流程 ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="NVIDIA API 标注脚本")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="NVIDIA 模型名称")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="低于此置信度跳过（high=1, medium=0.5, low=0）")
    parser.add_argument("--dry-run", action="store_true", help="只打印，不调用 API")
    parser.add_argument("--derive-only", action="store_true",
                        help="跳过 API 调用，直接从 cache 派生 JSONL")
    parser.add_argument("--no-thinking", action="store_true",
                        help="禁用模型思考模式")
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key and not args.dry_run and not args.derive_only:
        print("错误：请设置 NVIDIA_API_KEY 环境变量")
        sys.exit(1)

    if not UNIQUE_IMAGES_TXT.exists():
        print(f"错误：{UNIQUE_IMAGES_TXT} 不存在，请先运行 dedup_dataset.py")
        sys.exit(1)

    images = [ROOT / p.strip() for p in UNIQUE_IMAGES_TXT.read_text().splitlines() if p.strip()]
    print(f"模型: {args.model}")
    print(f"待标注图片：{len(images)} 张")

    cache = load_cache()
    skip = load_skip()
    skip_keys = {s["image"] for s in skip}

    # ── API 打标 ──────────────────────────────────────────────────────────────
    if not args.derive_only:
        conf_score = {"high": 1.0, "medium": 0.5, "low": 0.0}
        todo = [
            p for p in images
            if str(p.resolve()) not in cache and str(p.resolve()) not in skip_keys
        ]
        print(f"已缓存：{len(cache)}，跳过列表：{len(skip)}，本次需标注：{len(todo)}")

        for i, img_path in enumerate(todo):
            print(f"[{i+1}/{len(todo)}] {img_path.name} ...", end=" ", flush=True)

            if args.dry_run:
                print("(dry-run 跳过)")
                continue

            raw, err = call_nvidia_api(
                img_path, api_key,
                model=args.model,
                enable_thinking=not args.no_thinking,
            )
            if err:
                print(f"API 失败: {err}")
                skip.append({"image": str(img_path.resolve()), "reason": f"api_error: {err}"})
                save_skip(skip)
                continue

            ann, err = parse_canonical(raw, img_path, api_key)
            if err:
                print(f"解析失败: {err}")
                skip.append({
                    "image": str(img_path.resolve()),
                    "reason": f"parse_error: {err}",
                    "raw": raw,
                })
                save_skip(skip)
                continue

            score = conf_score.get(ann["confidence"], 0.5)
            if score < args.threshold:
                print(f"低置信度({ann['confidence']})，跳过")
                skip.append({
                    "image": str(img_path.resolve()),
                    "reason": f"low_confidence: {ann['confidence']}",
                    "annotation": ann,
                })
                save_skip(skip)
                continue

            # 记录实际使用的模型
            ann["model"] = args.model
            cache[str(img_path.resolve())] = ann
            save_cache(cache)
            status = "违规" if not ann["is_compliant"] else "合规"
            print(f"OK [{status}, confidence={ann['confidence']}]")

            time.sleep(0.5)  # NVIDIA API 限速

    # ── 派生 JSONL ────────────────────────────────────────────────────────────
    print(f"\n从 {len(cache)} 条 canonical annotation 派生 JSONL ...")

    stage1_samples: list[dict] = []
    stage2_samples: list[dict] = []

    for ann in cache.values():
        stage1_samples.extend(to_stage1_sample(ann))
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

    low_conf = [a for a in cache.values() if a.get("confidence") == "low"]
    if low_conf:
        print(f"\n低置信度（需人工复查）：{len(low_conf)} 条")
        for a in low_conf:
            print(f"  {a['image']}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
