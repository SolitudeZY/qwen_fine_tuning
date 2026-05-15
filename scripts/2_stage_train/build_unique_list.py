"""
按子目录组合构建 unique_images.txt。

与 dedup_dataset.py 的差异：
- dedup_dataset.py：扫整个 dataset/，做完整 MD5 + pHash 去重（早期一次性流程）
- 本脚本：按指定子目录拼，跨子目录做 pHash 去重（增量流程）

适用场景：
  - 已有标注好的批次，要追加新批次（如 video_frames）
  - 多视频抽帧后跨视频去重
  - 不动既有 dataset/ 全扫产物的可复现性

用法：
  # 默认：tiled_1024 + video_frames
  python scripts/2_stage_train/build_unique_list.py

  # 指定子目录
  python scripts/2_stage_train/build_unique_list.py \\
      --include dataset/tiled_1024 dataset/video_frames

  # 跳过 pHash 跨目录去重（仅 MD5，更快）
  python scripts/2_stage_train/build_unique_list.py --no-phash

  # 自定义输出
  python scripts/2_stage_train/build_unique_list.py --output data/unique_v2.txt
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INCLUDES = [ROOT / "dataset" / "tiled_1024", ROOT / "dataset" / "video_frames"]
DEFAULT_OUTPUT = ROOT / "data" / "unique_images.txt"
PHASH_THRESHOLD = 10
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def md5_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path: Path, hash_size: int = 8) -> int:
    img = Image.open(path).convert("L").resize(
        (hash_size * 4, hash_size * 4), Image.LANCZOS
    )
    pixels = np.array(img, dtype=np.float32)
    dct_rows = dct(pixels, axis=1, norm="ortho")
    dct_2d = dct(dct_rows, axis=0, norm="ortho")
    low = dct_2d[:hash_size, :hash_size]
    bits = (low > low.mean()).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def collect(dirs: list[Path]) -> list[Path]:
    images: list[Path] = []
    for d in dirs:
        if not d.exists():
            print(f"[警告] 目录不存在：{d}", file=sys.stderr)
            continue
        images.extend(p for p in d.rglob("*") if p.suffix in IMAGE_EXTS)
    return sorted(set(images))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--include",
        nargs="+",
        type=Path,
        default=DEFAULT_INCLUDES,
        help="要纳入的子目录列表（默认 tiled_1024 + video_frames）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出路径（默认 {DEFAULT_OUTPUT.relative_to(ROOT)}）",
    )
    parser.add_argument(
        "--no-phash",
        action="store_true",
        help="跳过 pHash 跨目录去重（仅 MD5），更快",
    )
    parser.add_argument(
        "--phash-threshold",
        type=int,
        default=PHASH_THRESHOLD,
        help=f"pHash Hamming 阈值，<=视为重复（默认 {PHASH_THRESHOLD}）",
    )
    args = parser.parse_args()

    include_dirs = [d if d.is_absolute() else ROOT / d for d in args.include]
    print("纳入目录：")
    for d in include_dirs:
        print(f"  {d}")

    images = collect(include_dirs)
    total = len(images)
    print(f"\n扫描到 {total} 张图")
    if total == 0:
        sys.exit(1)

    # MD5 精确去重
    print("\n[1/2] MD5 精确去重 ...")
    md5_seen: dict[str, Path] = {}
    after_md5: list[Path] = []
    md5_dropped = 0
    for p in images:
        h = md5_hash(p)
        if h in md5_seen:
            md5_dropped += 1
        else:
            md5_seen[h] = p
            after_md5.append(p)
    print(f"  精确重复：{md5_dropped}  →  剩 {len(after_md5)}")

    # pHash 感知去重
    if args.no_phash:
        kept = after_md5
        phash_dropped = 0
    else:
        print(f"\n[2/2] pHash 感知去重（Hamming ≤ {args.phash_threshold}）...")
        kept_hashes: list[tuple[int, Path]] = []
        kept: list[Path] = []
        phash_dropped = 0
        for i, p in enumerate(after_md5):
            if i % 100 == 0:
                print(f"  {i}/{len(after_md5)}", end="\r")
            try:
                h = phash(p)
            except Exception as e:
                print(f"\n  警告：{p} pHash 失败（{e}），保留", file=sys.stderr)
                kept.append(p)
                continue
            dup = next(
                (kp for kh, kp in kept_hashes if hamming(h, kh) <= args.phash_threshold),
                None,
            )
            if dup:
                phash_dropped += 1
            else:
                kept_hashes.append((h, p))
                kept.append(p)
        print(f"  感知重复：{phash_dropped}  →  剩 {len(kept)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for p in kept:
            f.write(str(p) + "\n")
    print(f"\n写入 {args.output}（{len(kept)} 张）")
    print(f"汇总：{total} → MD5 删 {md5_dropped} → pHash 删 {phash_dropped} → 保留 {len(kept)}")


if __name__ == "__main__":
    main()
