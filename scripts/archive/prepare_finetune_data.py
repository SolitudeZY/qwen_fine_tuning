import os
import json
import random
import base64
from openai import OpenAI
from tqdm import tqdm

# ============ 配置区域 ============
# 阿里云 DashScope API Key (需替换为你自己的 Key)
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-5798425ff35942dda384a634eb342036")
# 基础 URL
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 数据集目录
NON_COMPLIANT_DIR = "/home/fs-ai/llama-qwen/Fences_noncomlaint"
COMPLIANT_DIR = "/home/fs-ai/llama-qwen/dataset_compliant_fences"
OUTPUT_JSONL = "/home/fs-ai/llama-qwen/data/finetune_fences_minimal_cot.jsonl"

# 客户端初始化
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 系统提示词 (用于微调时)
SYSTEM_PROMPT = (
    "你是一名严谨的边坡与基坑施工安全监测专家，负责基于施工现场图片输出可落地的监测结论。\n"
    "你的任务不是泛泛描述图片，而是围绕高坠风险和临边防护进行监测判定，并输出结构化的推理过程和JSON结果。\n"
    "【场景识别要求】：\n"
    "- 基坑：向下开挖形成的坑槽，坑边通常需要防护隔离。\n"
    "- 边坡：倾斜土坡、岩坡本身通常不是必须全封闭围挡的对象。但坡顶作业平台、临空边若存在坠落风险，仍需防护。\n"
    "- 如果只是可见大面积施工坡面，不要仅凭'是边坡'或'是基坑'或就判为违规。\n"
    "【输出要求】：\n"
    "1. 首先，你需要详细描述图片中危险区域的状态，并进行安全推理。\n"
    "2. 然后，你必须在回答的最后严格以 JSON 格式输出判定结果。\n"
    "JSON 字段必须包含：violation_detected(布尔), violation_type(字符串), severity(字符串), suggestion(字符串)。\n"
    "3. 如果检测到违规(violation_detected=true)，你必须在最后输出的 JSON 对象中，直接包含一个名为 violation_boxes 的数组，"
    "每个元素包含：label(违规简述), bbox(四个整数的数组，格式为[x_min, y_min, x_max, y_max]，"
    "坐标为相对于图片宽高的千分比坐标，范围0-1000)。"
    "绝不能将 violation_boxes 写在 JSON 外部。\n"
    "例如，整个 JSON 应该是这样的结构：\n"
    "{\n"
    "  \"violation_detected\": true,\n"
    "  \"violation_type\": \"...\",\n"
    "  \"severity\": \"...\",\n"
    "  \"suggestion\": \"...\",\n"
    "  \"violation_boxes\": [{\"label\": \"围栏断口\", \"bbox\": [130, 300, 450, 680]}]\n"
    "}\n"
    "4. 如果合规(violation_detected=false)，violation_boxes 为空数组 []。"
)

DEFAULT_QUERY = "请对这张施工现场图片执行边坡/基坑安全监测，识别场景类型，提取关键观察，判断是否存在临边高坠风险及防护缺陷，并输出安全推理与 JSON 结果。如果存在违规，请用 violation_boxes 标出违规区域的位置坐标。"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_format(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext in ['.jpg', '.jpeg']: return 'jpeg'
    if ext == '.png': return 'png'
    return 'jpeg'

def convert_bbox_to_thousandth(points, width, height):
    """将 LabelMe 坐标转换为 0-1000 的千分比坐标 [x_min, y_min, x_max, y_max]"""
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    
    # 限制在图片范围内并转换为千分比
    xmin_t = max(0, min(1000, int((xmin / width) * 1000)))
    ymin_t = max(0, min(1000, int((ymin / height) * 1000)))
    xmax_t = max(0, min(1000, int((xmax / width) * 1000)))
    ymax_t = max(0, min(1000, int((ymax / height) * 1000)))
    
    return [xmin_t, ymin_t, xmax_t, ymax_t]

def generate_reasoning_via_qwen_max(img_path, ground_truth_boxes, is_violation):
    """
    调用 Qwen-VL-Max，让其根据我们提供的“绝对真理(Ground Truth)”反推生成完美的推理过程。
    """
    base64_image = encode_image(img_path)
    img_format = get_image_format(img_path)
    
    # 构建给大模型的 Prompt
    if is_violation:
        gt_info = f"这张图片存在违规，人工标注的违规区域及坐标如下（千分比格式）：\n"
        for b in ground_truth_boxes:
            gt_info += f"- 标签：{b['label']}, 坐标：{b['bbox']}\n"
        gt_info += "请你基于以上真实坐标和标签，生成一段极简的安全检查推理过程。只用一两句话说明发现了什么违规，然后立即严格按照以下 JSON 格式输出最终结果，必须包含上述提供的 violation_boxes。"
    else:
        gt_info = "这张图片的临边防护设施（如围栏、围挡）完全合规，不存在断口、倒伏等违规情况。请你生成一段极简的安全检查推理过程。只用一句话说明防护措施到位，然后立即输出 violation_detected 为 false 的 JSON，且 violation_boxes 必须为空数组 []。"

    prompt = f"""
你现在是一个用于生成微调数据集的 Teacher Model。
你需要模仿一个高效的工业安全检查AI助手，对下方的施工现场图片进行快速分析。
【核心要求】：
{gt_info}

你的输出必须符合以下极简格式（先输出一两句话的短文本，最后输出一段纯 JSON，不要带 markdown 代码块）：

【推理】：
...（一两句话说明情况）

{{
  "violation_detected": {str(is_violation).lower()},
  "violation_type": "...",
  "severity": "...",
  "suggestion": "...",
  "violation_boxes": [{{"label": "...", "bbox": [...]}}] // 根据提供的信息填写，如果是合规的则必须是 []
}}
"""

    try:
        response = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/{img_format};base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[!] API 调用失败 ({os.path.basename(img_path)}): {e}")
        return None

def process_directory(directory, is_violation):
    """处理目录下的所有图片和 JSON，生成微调数据格式"""
    dataset = []
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    
    files = [f for f in os.listdir(directory) if f.endswith(supported_formats)]
    print(f"[*] 正在处理 {'违规(正)' if is_violation else '合规(负)'} 样本，共 {len(files)} 张图片...")
    
    for filename in tqdm(files):
        img_path = os.path.join(directory, filename)
        json_path = os.path.join(directory, os.path.splitext(filename)[0] + ".json")
        
        if not os.path.exists(json_path):
            print(f"[!] 找不到对应的 JSON 标注文件: {json_path}")
            continue
            
        with open(json_path, 'r', encoding='utf-8') as f:
            labelme_data = json.load(f)
            
        width = labelme_data.get('imageWidth')
        height = labelme_data.get('imageHeight')
        
        if not width or not height:
            with Image.open(img_path) as img:
                width, height = img.size
                
        # 提取 Ground Truth BBox
        gt_boxes = []
        for shape in labelme_data.get('shapes', []):
            label = shape.get('label', '')
            points = shape.get('points', [])
            if len(points) >= 2:
                bbox_1000 = convert_bbox_to_thousandth(points, width, height)
                gt_boxes.append({"label": label, "bbox": bbox_1000})
                
        # 调用 API 生成推理过程
        assistant_content = generate_reasoning_via_qwen_max(img_path, gt_boxes, is_violation)
        
        if assistant_content:
            # 构造微调所需的 JSONL 单行数据
            record = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": f"<image>{DEFAULT_QUERY}"
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content
                    }
                ],
                "images": [
                    img_path
                ]
            }
            dataset.append(record)
            
    return dataset

def main():
    if API_KEY == "your-dashscope-api-key":
        print("[!] 请先在脚本中配置你的 DASHSCOPE_API_KEY！")
        return
        
    print(f"[*] 开始生成微调数据集...")
    
    # 1. 处理违规数据
    non_compliant_data = process_directory(NON_COMPLIANT_DIR, is_violation=True)
    
    # 2. 处理合规数据
    compliant_data = process_directory(COMPLIANT_DIR, is_violation=False)
    
    # 3. 合并打乱
    final_dataset = non_compliant_data + compliant_data
    random.shuffle(final_dataset)
    
    # 4. 保存为 JSONL
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for record in final_dataset:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
    print("\n" + "="*50)
    print(f"[*] 微调数据集生成成功！")
    print(f"[*] 违规样本数: {len(non_compliant_data)}")
    print(f"[*] 合规样本数: {len(compliant_data)}")
    print(f"[*] 总计样本数: {len(final_dataset)}")
    print(f"[*] 数据集路径: {OUTPUT_JSONL}")
    print("="*50)

if __name__ == "__main__":
    main()
