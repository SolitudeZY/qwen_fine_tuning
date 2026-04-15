import os
import json
import csv
import shutil

DATASET_DIR = "/home/fs-ai/llama-qwen/dataset"
OUTPUT_CSV = "/home/fs-ai/llama-qwen/outputs/all_fences_for_review.csv"
FENCE_DATASET_DIR = "/home/fs-ai/llama-qwen/dataset_fences_only"

def find_and_copy_all_fences():
    """
    遍历整个 dataset 目录，寻找所有包含了围栏相关标签的 LabelMe JSON 文件，
    并提取对应的图片路径和包含的围栏类型，输出为 CSV 供人工审查。
    同时，将这些图片和对应的 JSON 文件复制到一个新的目录，方便集中精细化标注。
    """
    fence_labels = {"围栏", "护栏", "临边防护", "围挡", "防护网"}
    results = []

    print(f"[*] 正在扫描目录: {DATASET_DIR} ...")
    os.makedirs(FENCE_DATASET_DIR, exist_ok=True)
    print(f"[*] 准备将围栏相关数据复制到: {FENCE_DATASET_DIR}")

    for root, _, files in os.walk(DATASET_DIR):
        for file in files:
            if file.lower().endswith(".json"):
                json_path = os.path.join(root, file)
                
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                        
                    found_fences = set()
                    for shape in data.get("shapes", []):
                        label = shape.get("label", "")
                        if label in fence_labels:
                            found_fences.add(label)
                            
                    if found_fences:
                        # 尝试找到对应的图片文件
                        img_path = data.get("imagePath", "")
                        if img_path:
                            # LabelMe 的 imagePath 可能是相对路径，也可能只是个文件名
                            full_img_path = os.path.join(root, img_path)
                            if not os.path.exists(full_img_path):
                                # 尝试直接替换扩展名为 jpg/JPG/png 等
                                base = os.path.splitext(json_path)[0]
                                for ext in [".jpg", ".JPG", ".png", ".jpeg"]:
                                    if os.path.exists(base + ext):
                                        full_img_path = base + ext
                                        break
                            
                            if os.path.exists(full_img_path):
                                results.append({
                                    "json_path": json_path,
                                    "image_path": full_img_path,
                                    "fence_types": " | ".join(found_fences)
                                })
                                
                                # 复制图片和 JSON 到专属目录
                                dest_img_path = os.path.join(FENCE_DATASET_DIR, os.path.basename(full_img_path))
                                dest_json_path = os.path.join(FENCE_DATASET_DIR, os.path.basename(json_path))
                                
                                # 为了避免同名文件被覆盖，可以加上源目录名作为前缀（这里简化处理，如果确定没有同名可以直接复制）
                                # 如果你的数据集存在同名文件，请告诉我，我可以修改重命名逻辑
                                shutil.copy2(full_img_path, dest_img_path)
                                shutil.copy2(json_path, dest_json_path)

                except Exception as e:
                    pass

    return results

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    results = find_and_copy_all_fences()
    
    print(f"[*] 扫描并复制完成！共找到 {len(results)} 张包含围栏标注的图片。")
    print(f"[*] 纯围栏数据集已保存在: {FENCE_DATASET_DIR}")
    print(f"[*] 正在将结果保存到: {OUTPUT_CSV}")
    
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Path", "JSON Path", "Fence Types", "Is Violation (Y/N/Review)", "Violation Type (e.g., 断开, 倒伏, 未合围)", "Violation Box"])
        for res in results:
            writer.writerow([res["image_path"], res["json_path"], res["fence_types"], "", "", ""])
            
    print("[*] 保存成功。你可以打开 CSV 文件进行人工标注审查。")
