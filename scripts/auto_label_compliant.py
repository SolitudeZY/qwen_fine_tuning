import os
import json
from PIL import Image

def generate_empty_labelme_json(image_dir):
    """
    为指定目录下的所有图片（jpg, png等）生成一个“干净”的 LabelMe JSON 文件。
    这些 JSON 文件没有任何违规框（shapes为空），表示这张图是完全合规的。
    """
    supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    generated_count = 0
    
    print(f"[*] 正在扫描目录: {image_dir}")
    
    for filename in os.listdir(image_dir):
        if filename.endswith(supported_formats):
            img_path = os.path.join(image_dir, filename)
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(image_dir, json_filename)
            
            # 如果 JSON 已经存在，就不覆盖了，假设用户可能有其他标注
            if os.path.exists(json_path):
                continue
                
            try:
                # 获取图片的宽高
                with Image.open(img_path) as img:
                    width, height = img.size
                    
                # 构造空的 LabelMe 格式 JSON
                empty_json = {
                    "version": "5.2.1",
                    "flags": {},
                    "shapes": [], # 没有任何违规框！这就是合规的精髓
                    "imagePath": filename,
                    "imageData": None,
                    "imageHeight": height,
                    "imageWidth": width
                }
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_json, f, ensure_ascii=False, indent=2)
                    
                generated_count += 1
                
            except Exception as e:
                print(f"[!] 处理图片 {filename} 时出错: {e}")
                
    print(f"[*] 完成！共为 {generated_count} 张图片生成了表示'合规'的空 JSON 标注文件。")

if __name__ == "__main__":
    # 处理网上搜集的图片
    websearch_dir = "/home/fs-ai/llama-qwen/dataset_compliant_fences/websearch"
    if os.path.exists(websearch_dir):
        generate_empty_labelme_json(websearch_dir)
    else:
        print(f"[!] 目录不存在: {websearch_dir}")
