"""
对 dataset_norm/ 中已生成的切片图做二次 pHash + 特征点去重。
重复图片移入 dataset_norm_dupes/，不直接删除。
优化：按原图分组进行极其严格的组内去重，如果原图接近切片大小导致切片只是轻微平移，
那么这些高度重合的切片会被剔除，组内保留数量“越少越好”。
"""

import shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

INPUT_DIR  = Path("/home/fs-ai/llama-qwen/yolo-drone/dataset_norm")
DUPES_DIR  = Path("/home/fs-ai/llama-qwen/yolo-drone/dataset_norm_dupes")
PHASH_THRESHOLD = 12   # 放宽哈希容忍度（越小越严格，调大可以杀掉更多相似图）
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

def get_phash(img_path: Path):
    try:
        img = Image.open(img_path).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.array(img.getdata(), dtype=float).reshape((32, 32))
        dct = cv2.dct(pixels)
        dctlowfreq = dct[:8, :8]
        med = np.median(dctlowfreq)
        diff = dctlowfreq > med
        return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])
    except Exception as e:
        print(f"  [warn] pHash 失败: {img_path.name}: {e}")
        return None

def hamming(h1, h2):
    return bin(h1 ^ h2).count("1")

def is_highly_similar(img_path1, img_path2):
    """
    使用 ORB 特征点匹配来判断两张图是否只是发生了“轻微平移”。
    如果两张图的大部分特征点能强匹配上，就认为它们是“极度相似”的（重叠度极高）。
    """
    try:
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None:
            return False

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return False

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # 筛选出距离（差异）非常小的强匹配点
        good_matches = [m for m in matches if m.distance < 40]
        
        # 如果强匹配点占总特征点很大比例，说明两张图画面主体完全一样，只是视角稍微挪了一下
        min_features = min(len(des1), len(des2))
        match_ratio = len(good_matches) / min_features
        
        return match_ratio > 0.4  # 如果有40%以上的强特征点重合，判定为极度相似
    except Exception as e:
        print(f"  [warn] 相似度对比失败: {e}")
        return False

def main():
    all_imgs = sorted(
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    )
    print(f"共 {len(all_imgs)} 张图片，开始计算 pHash...")

    with ThreadPoolExecutor(max_workers=8) as ex:
        hashes = list(ex.map(get_phash, all_imgs))

    pairs = list(zip(all_imgs, hashes))
    pairs = [(p, h) for p, h in pairs if h is not None]

    DUPES_DIR.mkdir(exist_ok=True)

    kept, dupes = [], []
    
    # 按照原图前缀分组 (例如 'img_00001')
    grouped_pairs = defaultdict(list)
    for img_path, phash in pairs:
        # 文件名格式如: img_00001_global.jpg 或 img_00001_tile_00.jpg
        prefix = "_".join(img_path.stem.split("_")[:2])
        grouped_pairs[prefix].append((img_path, phash))

    print(f"将图片分为 {len(grouped_pairs)} 组进行组内强力去重（保留最少）...")

    for prefix, group in grouped_pairs.items():
        # 优先保留 global 图（将 global 排序在前面）
        group.sort(key=lambda x: "global" not in x[0].name)
        
        kept_in_group = []
        for img_path, phash in group:
            is_dup = False
            # 组内每一张新图，都要和已经决定保留的图做严苛比对
            for kept_img, kept_phash in kept_in_group:
                # 条件 1：pHash 相似（调大阈值，杀得更狠）
                hash_similar = hamming(phash, kept_phash) <= PHASH_THRESHOLD
                
                # 条件 2：特征点强匹配（专门对付“轻微平移”的切片）
                feature_similar = is_highly_similar(img_path, kept_img)
                
                if hash_similar or feature_similar:
                    is_dup = True
                    break
                    
            if is_dup:
                dupes.append(img_path)
            else:
                kept.append(img_path)
                kept_in_group.append((img_path, phash))

    print(f"\n保留: {len(kept)}  重复: {len(dupes)}")
    print(f"移动重复图片到 {DUPES_DIR} ...")

    for p in dupes:
        shutil.move(str(p), str(DUPES_DIR / p.name))

    print("完成。")
    print(f"  dataset_norm/       剩余 {len(kept)} 张")
    print(f"  dataset_norm_dupes/ 移入 {len(dupes)} 张")

if __name__ == "__main__":
    main()
