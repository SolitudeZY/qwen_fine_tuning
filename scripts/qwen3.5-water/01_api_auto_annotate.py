import os
import json
import base64
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description="API 自动标注环水保图片数据")
    parser.add_argument("--xml-path", type=str, default="/home/fs-ai/llama-qwen/data/环水保_labeled/annotations.xml")
    parser.add_argument("--img-dir", type=str, default="/home/fs-ai/llama-qwen/data/环水保_labeled/images/Train")
    parser.add_argument("--output", type=str, default="/home/fs-ai/llama-qwen/data/annotate_cache_water.json")
    args = parser.parse_args()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请先设置环境变量 DASHSCOPE_API_KEY")

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    xml_path = Path(args.xml_path)
    img_dir = Path(args.img_dir)
    output_path = Path(args.output)

    if not xml_path.exists():
        raise FileNotFoundError(f"找不到 XML 文件: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取现有的 cache，支持断点续传
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {}

    images = root.findall(".//image")
    print(f"总计找到 {len(images)} 张图片标注")

    for img in tqdm(images, desc="API 标注中"):
        img_name = img.get("name")
        img_path = img_dir / img_name
        
        if not img_path.exists():
            print(f"[Warning] 图片不存在: {img_path}")
            continue
            
        # CVAT 标签解析
        tag_elem = img.find("tag")
        if tag_elem is None:
            continue
        label = tag_elem.get("label")
        
        # 映射标签
        if label == "water_normal":
            water_status = "normal"
            status_zh = "正常（无明显污染）"
        elif label == "water_polluted":
            water_status = "polluted"
            status_zh = "污染（有藻类、油污、垃圾或浑浊等）"
        else:
            continue

        # 如果已经标注过，且有描述，则跳过
        if str(img_path) in cache and cache[str(img_path)].get("description"):
            continue

        # 调用 API
        base64_image = encode_image(img_path)
        prompt = f"这是一张水域的俯拍图片。已知该水体的状态是：{status_zh}。请结合图片内容，给出一句话简短描述，解释为什么它是这个状态（例如：水面清澈无杂物，或水面有大面积绿色浮萍等）。请只输出一句话描述，不要输出多余内容。"

        try:
            response = client.chat.completions.create(
                model="qwen-vl-plus",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=100,
                temperature=0.3
            )
            description = response.choices[0].message.content.strip()
            
            # 保存到 cache，兼容 review_ui 格式
            cache[str(img_path)] = {
                "image": str(img_path),
                "water_status": water_status,
                "description": description,
                "reviewed": False,
                "confidence": "high",  # 默认给 high，审查时可以看
            }
            
            # 实时写入，防止中断
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"API 请求失败 ({img_name}): {e}")

    print(f"全部完成，缓存已保存至: {output_path}")

if __name__ == "__main__":
    main()
