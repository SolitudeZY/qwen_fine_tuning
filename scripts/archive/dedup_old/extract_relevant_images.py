import json
import os
from pathlib import Path
from collections import defaultdict

# ── 配置 ───────────────────────────────────────────────────────────────────
DATASET_DIR = Path('/home/fs-ai/llama-qwen/dataset')
OUTPUT_ALL = Path('/home/fs-ai/llama-qwen/data/all_labels_images.json')
OUTPUT_RELEVANT = Path('/home/fs-ai/llama-qwen/data/relevant_images.json')
OUTPUT_RELEVANT_TXT = Path('/home/fs-ai/llama-qwen/data/relevant_images.txt')

# 与我们任务（高空防坠、临边防护、围栏、基坑等）强相关的标签关键词
RELEVANT_KEYWORDS = [
    "防护", "围挡", "基坑", "边坡", "临边", "泥浆池", 
    "平台", "护栏", "网片", "标识", "标牌", "警示", 
    "水马", "防撞", "反光", "安全锥", "孔口", "井盖", 
    "爬模", "爬梯", "塔梯", "抱箍", "限位装置"
]

def is_relevant_label(label: str) -> bool:
    """判断一个标签是否与我们的任务强相关"""
    if not label:
        return False
    for kw in RELEVANT_KEYWORDS:
        if kw in label:
            return True
    return False

def main():
    json_files = list(DATASET_DIR.rglob('*.json'))
    print(f"找到 {len(json_files)} 个 JSON 标注文件。")

    # 数据结构
    # label_to_images: { "标签名": ["图1路径", "图2路径"] }
    label_to_images = defaultdict(list)
    
    # image_to_labels: { "图路径": ["标签1", "标签2"] }
    image_to_labels = defaultdict(list)

    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 获取图片路径
                img_name = data.get("imagePath", "")
                if not img_name:
                    continue
                img_path = jf.parent / img_name
                img_str = str(img_path.resolve())

                for shape in data.get('shapes', []):
                    label = shape.get('label')
                    if label:
                        label_to_images[label].append(img_str)
                        if label not in image_to_labels[img_str]:
                            image_to_labels[img_str].append(label)
        except Exception as e:
            print(f"解析 {jf} 失败: {e}")

    # 1. 保存所有标签及其对应的图片
    # 去重图片路径
    all_output_data = {}
    for label, imgs in label_to_images.items():
        all_output_data[label] = list(set(imgs))
    
    OUTPUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ALL, 'w', encoding='utf-8') as f:
        json.dump(all_output_data, f, ensure_ascii=False, indent=2)
    print(f"\n已将所有标签及图片映射保存至: {OUTPUT_ALL}")

    # 2. 筛选强相关图片
    relevant_images_data = {}
    relevant_image_paths = set()

    for img_path, labels in image_to_labels.items():
        # 只要这张图里包含至少一个强相关标签，我们就保留这张图
        relevant_labels_in_img = [lbl for lbl in labels if is_relevant_label(lbl)]
        if relevant_labels_in_img:
            relevant_images_data[img_path] = labels
            relevant_image_paths.add(img_path)

    # 保存相关图片 JSON (包含路径和它拥有的所有标签)
    with open(OUTPUT_RELEVANT, 'w', encoding='utf-8') as f:
        json.dump(relevant_images_data, f, ensure_ascii=False, indent=2)
    
    # 保存纯文本路径列表（方便后续脚本直接读取去重和切分）
    with open(OUTPUT_RELEVANT_TXT, 'w', encoding='utf-8') as f:
        for p in sorted(relevant_image_paths):
            f.write(f"{p}\n")

    print(f"\n--- 筛选结果 ---")
    print(f"总图片数 (有标注的): {len(image_to_labels)}")
    print(f"强相关图片数: {len(relevant_image_paths)}")
    print(f"相关图片详细信息保存至: {OUTPUT_RELEVANT}")
    print(f"纯路径列表保存至: {OUTPUT_RELEVANT_TXT}")
    
    # 打印一些被保留的代表性标签
    kept_labels = set()
    for labels in relevant_images_data.values():
        kept_labels.update([lbl for lbl in labels if is_relevant_label(lbl)])
    print(f"\n命中的强相关标签示例 ({len(kept_labels)}种):")
    for lbl in sorted(list(kept_labels))[:20]:
        print(f"  - {lbl}")
    if len(kept_labels) > 20:
        print("  - ...")

if __name__ == '__main__':
    main()
