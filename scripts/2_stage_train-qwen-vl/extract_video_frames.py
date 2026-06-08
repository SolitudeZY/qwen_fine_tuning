"""
视频抽帧脚本：从 dataset/video/ 中的无人机视频抽取训练候选帧。

流程：
  1. 按固定时间间隔抽帧（默认每秒 1 帧）
  2. 用感知哈希（pHash）做相邻帧相似度过滤，跳过近似重复
  3. 可选：1920×1080 帧切成 2 块 1080×1080，对齐 dataset/tiled_1024/ 的训练习惯
  4. 输出到 dataset/video_frames/

用法：
  python scripts/2_stage_train/extract_video_frames.py
  python scripts/2_stage_train/extract_video_frames.py --fps 0.5 --hash-threshold 6
  python scripts/2_stage_train/extract_video_frames.py --tile        # 同时输出 1080×1080 切块
  python scripts/2_stage_train/extract_video_frames.py --videos a.mp4 b.mp4

后续接现有流程：
  - 抽帧产物路径加入 unique_images.txt 后即可走 api_annotate_stage12.py 标注
"""

import argparse
import sys
from pathlib import Path

import cv2
import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "dataset" / "video"
FRAMES_DIR = ROOT / "dataset" / "video_frames"

VIDEO_EXTS = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI", ".mkv"}


def iter_videos(explicit: list[str] | None) -> list[Path]:
    if explicit:
        out = []
        for v in explicit:
            p = Path(v)
            if not p.is_absolute():
                p = VIDEO_DIR / v
            if not p.exists():
                print(f"[警告] 找不到视频：{p}", file=sys.stderr)
                continue
            out.append(p)
        return out
    return sorted(p for p in VIDEO_DIR.iterdir() if p.suffix in VIDEO_EXTS)


def phash(bgr) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def save_frame(bgr, out_path: Path, jpeg_quality: int = 92) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])


def split_1080_tiles(bgr) -> list[tuple[str, "cv2.Mat"]]:
    """1920×1080 → 左右两块 1080×1080。中间有 240px 重叠，覆盖中间画面。"""
    h, w = bgr.shape[:2]
    if w < 1080 or h < 1080:
        return []
    side = min(h, 1080)
    left = bgr[:side, :side]
    right = bgr[:side, w - side:]
    return [("L", left), ("R", right)]


def extract(
    video: Path,
    fps: float,
    hash_threshold: int,
    out_dir: Path,
    tile: bool,
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"[错误] 无法打开：{video}", file=sys.stderr)
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(src_fps / fps)))

    print(f"\n=== {video.name} ===")
    print(f"  源 FPS: {src_fps:.2f}  总帧: {total}  抽帧步长: {step} (≈每{step/src_fps:.2f}s)")

    saved = 0
    skipped = 0
    last_hash: imagehash.ImageHash | None = None
    idx = 0
    stem = video.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue

        h = phash(frame)
        if last_hash is not None and (h - last_hash) <= hash_threshold:
            skipped += 1
            idx += 1
            continue
        last_hash = h

        seconds = int(idx / src_fps)
        base = f"{stem}_t{seconds:05d}"
        save_frame(frame, out_dir / f"{base}.jpg")
        saved += 1

        if tile:
            for tag, tile_img in split_1080_tiles(frame):
                save_frame(tile_img, out_dir / f"{base}_tile{tag}.jpg")
                saved += 1

        idx += 1

    cap.release()
    print(f"  保存 {saved} 张，跳过近似 {skipped} 张")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fps", type=float, default=1.0, help="抽帧速率（每秒抽几帧），默认 1")
    parser.add_argument(
        "--hash-threshold",
        type=int,
        default=5,
        help="pHash 汉明距离阈值，<=该值视为重复（0~64，默认 5）",
    )
    parser.add_argument("--tile", action="store_true", help="同时输出左右 1080×1080 切块")
    parser.add_argument("--videos", nargs="*", help="指定视频文件名/路径，留空则处理 dataset/video/ 全部")
    parser.add_argument("--out-dir", type=Path, default=FRAMES_DIR, help=f"输出目录，默认 {FRAMES_DIR}")
    args = parser.parse_args()

    videos = iter_videos(args.videos)
    if not videos:
        print(f"[错误] 在 {VIDEO_DIR} 未找到视频", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(extract(v, args.fps, args.hash_threshold, args.out_dir, args.tile) for v in videos)
    print(f"\n完成：共保存 {total} 张到 {args.out_dir}")


if __name__ == "__main__":
    main()
