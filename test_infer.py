import sys
import os
sys.path.append(os.getcwd())
from scripts.model_utils import load_vlm, infer_vlm

model_path = "/home/fs-ai/llama-qwen/models/Qwen/Qwen3-VL-2B-Instruct"
image_path = "/home/fs-ai/llama-qwen/dataset/标注111-220/DJI_0025.JPG"

print("Loading model...")
model, processor, family = load_vlm(model_path)

messages = [
    {"role": "system", "content": "你是一名严谨的边坡与基坑施工安全监测专家... 必须在回答的最后严格以 JSON 格式输出判定结果。"},
    {"role": "user", "content": [{"type": "image", "image": f"file://{os.path.abspath(image_path)}"}, {"type": "text", "text": "请判断是否存在临边高坠风险及防护缺陷，并严格输出 JSON。"}]}
]

print("Inferring...")
response = infer_vlm(model, processor, family, messages, max_new_tokens=768)
print("Response length:", len(response))
print("Contains {: ", "{" in response)
print("Contains }: ", "}" in response)
print(response)
