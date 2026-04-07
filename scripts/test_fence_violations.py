import os
import json
import time
from tqdm import tqdm
from model_utils import load_vlm, infer_vlm
from chat import SYSTEM_PROMPT, DEFAULT_QUERY, _extract_json_object

# 配置
DATA_PATH = "/home/fs-ai/llama-qwen/data/train_balanced_clean.jsonl"
MODEL_PATH = "/home/fs-ai/llama-qwen/models/Qwen/Qwen3-VL-2B-Instruct"
LORA_PATH = "/home/fs-ai/llama-qwen/outputs/qwen3vl_2b_lora/v0-20260407-094017/checkpoint-170"
TEST_COUNT = 50

def get_fence_violation_images():
    """从数据集中提取包含真实 LabelMe 围栏标签且违规的图片路径"""
    images = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            try:
                ans = json.loads(item["messages"][-1]["content"])
                # 只有当图片本身是违规的，我们才继续检查
                if ans.get("violation_detected") is True:
                    img_path = item["images"][0]
                    if not os.path.exists(img_path):
                        continue
                        
                    # 尝试找到对应的 LabelMe JSON 文件
                    # 假设它和图片在同一个目录，后缀为 .json
                    json_path = os.path.splitext(img_path)[0] + ".json"
                    
                    if os.path.exists(json_path):
                        with open(json_path, "r", encoding="utf-8") as jf:
                            labelme_data = json.load(jf)
                            
                        # 检查 LabelMe 数据中是否真的画了围栏相关的框
                        has_fence_label = False
                        for shape in labelme_data.get("shapes", []):
                            label = shape.get("label", "")
                            if label in ["围栏", "护栏", "临边防护", "围挡", "防护网"]:
                                has_fence_label = True
                                break
                                
                        if has_fence_label:
                            images.append({
                                "path": img_path,
                                "expected_type": ans.get("violation_type", ""),
                                "expected_suggestion": ans.get("suggestion", "")
                            })
            except Exception as e:
                pass
    return images

def test_model():
    print(f"[*] 正在从 {DATA_PATH} 提取围栏违规测试图片...")
    all_violation_images = get_fence_violation_images()
    test_images = all_violation_images[:TEST_COUNT]
    
    print(f"[*] 成功提取 {len(test_images)} 张违规图片，准备加载模型...")
    model, processor, model_family = load_vlm(MODEL_PATH, lora_path=LORA_PATH)
    
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
            response = infer_vlm(model, processor, model_family, messages, max_new_tokens=768)
            parsed_json, _, _ = _extract_json_object(response)
            
            if parsed_json:
                is_violation = parsed_json.get("violation_detected", False)
                # 兼容模型输出字符串 "true" / "false"
                if isinstance(is_violation, str):
                    is_violation = (is_violation.lower() == "true")
            else:
                # 如果没解析出来 JSON，尝试正则粗暴匹配
                is_violation = "violation_detected: true" in response.lower() or '"violation_detected": true' in response.lower()
                
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
    
    # 将结果保存到文件
    output_file = f"/home/fs-ai/llama-qwen/outputs/fence_test_results_{int(time.time())}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_tested": total_count,
                "correct_violations": correct_count,
                "missed_violations": total_count - correct_count,
                "recall_rate": f"{correct_count / total_count * 100:.2f}%",
                "time_elapsed": f"{end_time - start_time:.2f}s"
            },
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print("\n" + "="*60)
    print(" 测试报告")
    print("="*60)
    print(f"总测试数量: {total_count}")
    print(f"成功检出违规: {correct_count}")
    print(f"漏检(误判为合规): {total_count - correct_count}")
    print(f"准确率 (Recall): {correct_count / total_count * 100:.2f}%")
    print(f"总耗时: {end_time - start_time:.2f}s (平均 { (end_time - start_time)/total_count:.2f}s/张)")
    print(f"详细测试结果已保存至: {output_file}")
    print("="*60)
    
    print("\n前10条详细结果:")
    for res in results[:10]:
        print(f" - {res['image_name']}: {res['status']} | 检出类型: {res.get('parsed_type', 'N/A')}")

if __name__ == "__main__":
    test_model()