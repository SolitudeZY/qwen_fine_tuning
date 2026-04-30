"""
数据归一化脚本，用于图片去重、大图切分 (Slicing)，以便于YOLO训练

功能：
1. 感知哈希 (pHash) 去重：移除视觉上极其相似的重复图片
2. 滑动窗口切图 (Slicing)：将无人机高分辨率大图（如4000x3000）切分为重叠的小块（如 640x640 或 1024x1024）
   - 这样既能保持围栏等小目标的像素级清晰度，又能满足 YOLO 的输入要求
3. 多尺度保留 (Multi-scale)：对于大图，除了切片外，也会生成一张缩放后的全局图 (Global View)。
   - 这可以解决长条形围栏在切片中被截断、丢失全局上下文的问题。
4. 支持多线程处理，加速大批量图片处理
"""

import os
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
import cv2

# ─── 配置参数 ─────────────────────────────────────────────────────────────────
INPUT_DIR = "/home/fs-ai/llama-qwen/dataset"      # 原始图片目录
OUTPUT_DIR = "/home/fs-ai/llama-qwen/yolo-drone/dataset_norm"  # 处理后图片输出目录
TILE_SIZE = 1024                                  # 切块大小 (正方形) - YOLOv8 支持 1024
OVERLAP_RATIO = 0.25                               # 相邻切块的重叠率 
PHASH_THRESHOLD = 5                               # pHash 差异阈值
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
# ──────────────────────────────────────────────────────────────────────────────

def get_phash(img_path):
    """计算图片的感知哈希值 (pHash)"""
    try:
        img = Image.open(img_path).convert('L').resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.array(img.getdata(), dtype=float).reshape((32, 32))
        dct = cv2.dct(pixels)
        dctlowfreq = dct[:8, :8]
        med = np.median(dctlowfreq)
        diff = dctlowfreq > med
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception as e:
        print(f"Error computing pHash for {img_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """计算两个哈希值的汉明距离"""
    return bin(hash1 ^ hash2).count('1')

def create_global_view(img, target_size):
    """生成全局视角的缩放图 (Letterbox)"""
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    new_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    y_off, x_off = (target_size - nh) // 2, (target_size - nw) // 2
    new_img[y_off:y_off+nh, x_off:x_off+nw] = resized
    return new_img

def slice_image(image_path, tile_size, overlap_ratio):
    """
    将大图进行滑动窗口切片
    返回切片后的 numpy 图像列表，列表的第一个元素始终是全局缩放图
    """
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return []
            
        h, w = img.shape[:2]
        tiles = []
        
        # 1. 首先生成一张全局的缩放图 (Global View)
        # 这保证了无论图片怎么切，模型都能看到一次完整的围栏走向
        global_view = create_global_view(img, tile_size)
        tiles.append({"type": "global", "img": global_view})
        
        if h <= tile_size and w <= tile_size:
            return tiles

        # 2. 生成局部高清切片 (Local Tiles)
        stride = int(tile_size * (1 - overlap_ratio))
        nx = max(1, int(np.ceil((w - tile_size) / stride)) + 1)
        ny = max(1, int(np.ceil((h - tile_size) / stride)) + 1)
        
        for i in range(ny):
            for j in range(nx):
                y0 = i * stride
                x0 = j * stride
                
                if y0 + tile_size > h:
                    y0 = max(0, h - tile_size)
                if x0 + tile_size > w:
                    x0 = max(0, w - tile_size)
                    
                y1 = min(h, y0 + tile_size)
                x1 = min(w, x0 + tile_size)
                
                crop = img[y0:y1, x0:x1]
                
                ch, cw = crop.shape[:2]
                if ch < tile_size or cw < tile_size:
                    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                    padded[:ch, :cw] = crop
                    tiles.append({"type": "tile", "img": padded})
                else:
                    tiles.append({"type": "tile", "img": crop})
                    
        return tiles
    except Exception as e:
        print(f"Error slicing {image_path}: {e}")
        return []

def filter_useless_tile(img_array, edge_threshold=50):
    """
    使用传统的 CV 方法初步过滤纯树林、纯土壤等无意义图片。
    如果图片中的边缘信息（高频细节如脚手架、围栏、建筑结构）过少，则认为无意义。
    返回: True(保留), False(过滤)
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # 计算 Canny 边缘
    edges = cv2.Canny(gray, 100, 200)
    
    # 计算边缘像素点的比例
    edge_ratio = np.sum(edges > 0) / (img_array.shape[0] * img_array.shape[1])
    
    # 设定一个经验阈值，通常纯树林/土壤的边缘比例比较均匀且偏低，
    # 而包含脚手架、围栏、基坑等人工结构的图，边缘比例较高或有强烈的线条特征。
    # 我们这里用一个极低的阈值先卡掉纯黑/模糊/极度平滑的块
    if edge_ratio < 0.01:
        return False
        
    return True

def process_images():
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_images = []
    for ext in SUPPORTED_EXTS:
        all_images.extend(input_path.rglob(f"*{ext}"))
        
    print(f"找到 {len(all_images)} 张图片待处理...")
    
    print("开始计算 pHash 进行去重...")
    unique_hashes = []
    unique_images = []
    
    def process_hash(img_path):
        phash = get_phash(img_path)
        return img_path, phash

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_hash, all_images))
        
    for img_path, phash in results:
        if phash is None:
            continue
            
        is_duplicate = False
        for uhash in unique_hashes:
            if hamming_distance(phash, uhash) <= PHASH_THRESHOLD:
                is_duplicate = True
                break
                
        if not is_duplicate:
            unique_hashes.append(phash)
            unique_images.append(img_path)
            
    print(f"去重完成。剩余 {len(unique_images)} 张独立原图。")
    
    print(f"开始对高分辨率图片进行滑动切片 ({TILE_SIZE}x{TILE_SIZE}, 重叠 {OVERLAP_RATIO*100}%)...")
    total_tiles_generated = 0
    filtered_count = 0
    
    for i, img_path in enumerate(unique_images):
        tiles_info = slice_image(img_path, TILE_SIZE, OVERLAP_RATIO)
        for t_idx, tile_dict in enumerate(tiles_info):
            tile_img = tile_dict["img"]
            tile_type = tile_dict["type"]
            
            # 如果是全局缩放图，或者是原图就很小，直接保留不过滤
            # 只有局部切片才需要经过无意义过滤
            if tile_type == "tile" and not filter_useless_tile(tile_img):
                filtered_count += 1
                continue
                
            # 命名区分：全局图带 global 后缀，切片带 tile 后缀
            if tile_type == "global":
                out_file = output_path / f"img_{i:05d}_global.jpg"
            else:
                out_file = output_path / f"img_{i:05d}_tile_{t_idx:02d}.jpg"
                
            cv2.imwrite(str(out_file), tile_img)
            total_tiles_generated += 1
            
        if (i + 1) % 50 == 0:
            print(f"已处理 {i+1}/{len(unique_images)} 张原图，已生成 {total_tiles_generated} 张训练图，过滤了 {filtered_count} 张无意义切块")
                
    print(f"\n处理完成！")
    print(f"原始图片: {len(all_images)} 张")
    print(f"去重后独立原图: {len(unique_images)} 张")
    print(f"切片+全局图 生成最终训练图: {total_tiles_generated} 张")
    print(f"传统CV算法过滤掉无意义切块: {filtered_count} 张")
    print(f"输出目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_images()