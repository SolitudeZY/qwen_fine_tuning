import os
import json
import time
from tqdm import tqdm
from model_utils import load_vlm, infer_vlm
from chat import SYSTEM_PROMPT, DEFAULT_QUERY, _extract_json_object
from datetime import datetime

import csv

# 配置
CSV_DATA_PATH = "/home/fs-ai/llama-qwen/outputs/all_fences_for_review.csv"
NON_COMPLIANT_DIR = "/home/fs-ai/llama-qwen/Fences_noncomlaint"
MODEL_PATH = "/home/fs-ai/llama-qwen/models/Qwen/Qwen3-VL-2B-Instruct"
LORA_PATH = "/home/fs-ai/llama-qwen/outputs/qwen3vl_2b_fences_minimal_cot_lora/v0-20260409-191023/checkpoint-330"
# 备用路径-输出长但准确"/home/fs-ai/llama-qwen/outputs/qwen3vl_2b_fences_lora/v4-20260409-161021/checkpoint-420"
TEST_COUNT = 50

def get_fence_violation_images():
    """从违规数据集中提取确实存在违规框的图片路径"""
    images = []
    if not os.path.exists(NON_COMPLIANT_DIR):
        print(f"[!] 找不到违规数据集目录: {NON_COMPLIANT_DIR}")
        return images
        
    supported_formats = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    for filename in os.listdir(NON_COMPLIANT_DIR):
        if filename.endswith(supported_formats):
            img_path = os.path.join(NON_COMPLIANT_DIR, filename)
            json_path = os.path.splitext(img_path)[0] + ".json"
            
            # 只有当对应的 JSON 文件存在，并且里面确实画了框（shapes不为空）时，才认为是真正的违规图片
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        label_data = json.load(f)
                        if len(label_data.get("shapes", [])) > 0:
                            images.append({
                                "path": img_path,
                                "expected_type": "已知违规数据",
                                "expected_suggestion": "违规数据集"
                            })
                except Exception:
                    pass
    return images

def test_model(use_lora=True):
    print(f"[*] 正在从 {NON_COMPLIANT_DIR} 提取违规测试图片...")
    all_violation_images = get_fence_violation_images()
    test_images = all_violation_images[:TEST_COUNT]

    lora = LORA_PATH if use_lora else None
    model_type_str = "微调模型 (带 LoRA)" if use_lora else "原始基座模型 (无 LoRA)"
    print(f"\n Lora路径为 {LORA_PATH}")

    print(f"[*] 成功提取 {len(test_images)} 张违规图片，准备加载 {model_type_str}...")
    model, processor, model_family = load_vlm(MODEL_PATH, lora_path=lora)
    
    print(f"\n[*] 开始批量测试 (共 {len(test_images)} 张)...")
    
    correct_count = 0
    total_count = len(test_images)
    results = []
    
    start_time = time.time()
    
    for i, item in enumerate(tqdm(test_images)):
        img_path = item["path"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(img_path)}"},
                    {"type": "text", "text": DEFAULT_QUERY},
                ],
            },
        ]
        
        try:
            response = infer_vlm(model, processor, model_family, messages, max_new_tokens=1536)
            parsed_json, _, _ = _extract_json_object(response)
            
            if parsed_json:
                is_violation = parsed_json.get("violation_detected", False)
                # 兼容模型输出字符串 "true" / "false" 或大写
                if isinstance(is_violation, str):
                    is_violation = (is_violation.lower() == "true")
                
                # 新增逻辑：如果 violation_boxes 不为空，则强制认为是违规
                if not is_violation and parsed_json.get("violation_boxes") and len(parsed_json.get("violation_boxes")) > 0:
                    print(f"  [!] 发现逻辑矛盾: {os.path.basename(img_path)} 输出 false 但带有违规框，强制判定为违规。")
                    is_violation = True
            else:
                # 如果没解析出来 JSON，尝试正则粗暴匹配
                is_violation = "violation_detected: true" in response.lower() or '"violation_detected": true' in response.lower()
                
                # 正则匹配是否输出了违规框
                if not is_violation and '"bbox"' in response.lower():
                    is_violation = True
                    
            if is_violation:
                correct_count += 1
                status = "✅ 成功 (违规)"
            else:
                status = "❌ 失败 (合规)"
                
            results.append({
                "image_name": os.path.basename(img_path),
                "image_path": img_path,
                "status": status,
                "is_violation_detected": is_violation,
                "parsed_type": parsed_json.get("violation_type") if parsed_json else "Parse Error",
                "boxes_detected": len(parsed_json.get("violation_boxes", [])) if parsed_json else 0,
                "raw_response": response
            })
            
        except Exception as e:
            results.append({
                "image_name": os.path.basename(img_path),
                "image_path": img_path,
                "status": f"⚠️ 报错 ({str(e)})",
                "is_violation_detected": False,
                "parsed_type": "Error",
                "boxes_detected": 0,
                "raw_response": ""
            })

    end_time = time.time()

    timestamp = int(time.time())
    date_str = datetime.now().strftime('%Y%m%d')
    # 将结果保存到文件
    prefix = "lora_" if use_lora else "base_"
    output_file = f"/home/fs-ai/llama-qwen/outputs/{prefix}fence_test_results_{date_str}_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    recall_rate = f"{correct_count / total_count * 100:.2f}%" if total_count > 0 else "0.00%"
    avg_time = f"{(end_time - start_time) / total_count:.2f}s/张" if total_count > 0 else "0.00s/张"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": model_type_str,
            "summary": {
                "total_tested": total_count,
                "correct_violations": correct_count,
                "missed_violations": total_count - correct_count,
                "recall_rate": recall_rate,
                "time_elapsed": f"{end_time - start_time:.2f}s"
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print(f" {model_type_str} 测试报告")
    print("="*60)
    print(f"总测试数量: {total_count}")
    print(f"成功检出违规: {correct_count}")
    print(f"漏检(误判为合规): {total_count - correct_count}")
    print(f"准确率 (Recall): {recall_rate}")
    print(f"总耗时: {end_time - start_time:.2f}s (平均 {avg_time})")
    print(f"详细测试结果已保存至: {output_file}")
    print("="*60)
    
    print("\n前10条详细结果:")
    for res in results[:10]:
        print(f" - {res['image_name']}: {res['status']} | 检出类型: {res.get('parsed_type', 'N/A')}")

if __name__ == "__main__":
    import sys
    use_lora = "--base-only" not in sys.argv
    test_model(use_lora=use_lora)