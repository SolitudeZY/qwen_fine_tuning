"""
图片去重脚本：精确去重（MD5）+ 感知去重（pHash/DCT）
输出：
  data/unique_images.txt     — 保留的图片路径列表
  data/dedup_report.json     — 详细去重报告
  data/dedup_report.txt      — 人类可读摘要
"""

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# ── 配置 ──────────────────────────────────────────────────────────────────────
DATASET_DIR = Path("dataset")
OUTPUT_DIR = Path("data")
PHASH_THRESHOLD = 10          # Hamming 距离 ≤ 此值视为感知重复（64位hash，10≈15%差异）
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# ─────────────────────────────────────────────────────────────────────────────


def md5_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path, hash_size: int = 8) -> int:
    """DCT 感知哈希，返回 64 位整数。"""
    img = Image.open(path).convert("L").resize(
        (hash_size * 4, hash_size * 4), Image.LANCZOS
    )
    pixels = np.array(img, dtype=np.float32)

    # 2D DCT via separable 1D DCT
    from scipy.fft import dct
    dct_rows = dct(pixels, axis=1, norm="ortho")
    dct_2d = dct(dct_rows, axis=0, norm="ortho")

    low_freq = dct_2d[:hash_size, :hash_size]
    mean = low_freq.mean()
    bits = (low_freq > mean).flatten()
    # pack 64 bits into one int
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def collect_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.suffix in IMAGE_EXTS
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    images = collect_images(DATASET_DIR)
    total = len(images)
    print(f"扫描到 {total} 张图片")

    # ── Stage 1: 精确去重（MD5）────────────────────────────────────────────
    print("\n[1/2] 精确去重（MD5）...")
    md5_map: dict[str, Path] = {}          # md5 → 首次出现路径
    exact_dupes: list[dict] = []           # 被删除的精确重复
    after_exact: list[Path] = []

    for p in images:
        h = md5_hash(p)
        if h in md5_map:
            exact_dupes.append({
                "removed": str(p),
                "duplicate_of": str(md5_map[h]),
                "reason": "exact_md5",
                "md5": h,
            })
        else:
            md5_map[h] = p
            after_exact.append(p)

    print(f"  精确重复：{len(exact_dupes)} 张  →  剩余 {len(after_exact)} 张")

    # ── Stage 2: 感知去重（pHash）──────────────────────────────────────────
    print(f"\n[2/2] 感知去重（pHash，Hamming 阈值={PHASH_THRESHOLD}）...")
    hashes: list[tuple[int, Path]] = []    # (hash, path) 已保留
    perceptual_dupes: list[dict] = []
    kept: list[Path] = []

    for i, p in enumerate(after_exact):
        if i % 50 == 0:
            print(f"  处理 {i}/{len(after_exact)} ...", end="\r")
        try:
            h = phash(p)
        except Exception as e:
            print(f"\n  警告：{p} 无法计算 pHash（{e}），跳过")
            kept.append(p)
            continue

        # 与已保留图片比较
        matched = None
        for kept_h, kept_p in hashes:
            if hamming(h, kept_h) <= PHASH_THRESHOLD:
                matched = kept_p
                break

        if matched:
            perceptual_dupes.append({
                "removed": str(p),
                "duplicate_of": str(matched),
                "reason": "perceptual_phash",
                "hamming_distance": hamming(h, phash(matched)),
            })
        else:
            hashes.append((h, p))
            kept.append(p)

    print(f"\n  感知重复：{len(perceptual_dupes)} 张  →  剩余 {len(kept)} 张")

    # ── 输出 unique_images.txt ─────────────────────────────────────────────
    unique_txt = OUTPUT_DIR / "unique_images.txt"
    with open(unique_txt, "w") as f:
        for p in kept:
            f.write(str(p) + "\n")
    print(f"\n保留列表 → {unique_txt}  ({len(kept)} 张)")

    # ── 输出 JSON 报告 ─────────────────────────────────────────────────────
    report = {
        "summary": {
            "total_scanned": total,
            "exact_duplicates_removed": len(exact_dupes),
            "perceptual_duplicates_removed": len(perceptual_dupes),
            "kept": len(kept),
        },
        "exact_duplicates": exact_dupes,
        "perceptual_duplicates": perceptual_dupes,
        "kept_images": [str(p) for p in kept],
    }
    json_path = OUTPUT_DIR / "dedup_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── 输出人类可读摘要 ───────────────────────────────────────────────────
    txt_path = OUTPUT_DIR / "dedup_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        s = report["summary"]
        f.write("=" * 60 + "\n")
        f.write("图片去重报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"扫描总数：{s['total_scanned']}\n")
        f.write(f"精确重复（MD5）删除：{s['exact_duplicates_removed']}\n")
        f.write(f"感知重复（pHash）删除：{s['perceptual_duplicates_removed']}\n")
        f.write(f"最终保留：{s['kept']}\n\n")

        if exact_dupes:
            f.write("── 精确重复 ──\n")
            for d in exact_dupes:
                f.write(f"  删除: {d['removed']}\n")
                f.write(f"    ↳ 重复自: {d['duplicate_of']}\n")
            f.write("\n")

        if perceptual_dupes:
            f.write(f"── 感知重复（Hamming ≤ {PHASH_THRESHOLD}）──\n")
            for d in perceptual_dupes:
                f.write(f"  删除: {d['removed']}\n")
                f.write(f"    ↳ 相似于: {d['duplicate_of']}\n")
            f.write("\n")

        f.write("── 保留图片 ──\n")
        for p in kept:
            f.write(f"  {p}\n")

    print(f"JSON 报告  → {json_path}")
    print(f"文本报告  → {txt_path}")
    print(f"\n完成：{total} → 保留 {len(kept)}，删除 {len(exact_dupes) + len(perceptual_dupes)}")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent.parent)  # 切到项目根目录
    main()
