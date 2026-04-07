import json

def get_fence_violations():
    data_path = "/home/fs-ai/llama-qwen/data/train_balanced_clean.jsonl"
    fence_violation_images = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            assistant_response = item["messages"][-1]["content"]
            try:
                response_json = json.loads(assistant_response)
                # 我们寻找违规(violation_detected=true)的图片，并且最好是跟防护/围栏相关的
                # 比如 violation_type 包含 'edge_protection' 或者 'multiple_issues' 等
                # 或者 suggestion 中提到了围栏、围挡等
                if response_json.get("violation_detected") is True:
                    v_type = response_json.get("violation_type", "")
                    suggestion = response_json.get("suggestion", "")
                    content = response_json.get("monitoring_content", "")
                    result_desc = response_json.get("monitoring_result", "")
                    
                    if "protection" in v_type or "围栏" in suggestion or "围栏" in result_desc or "围挡" in suggestion or "护栏" in suggestion:
                        fence_violation_images.append({
                            "image": item["images"][0],
                            "expected_violation": True,
                            "violation_type": v_type
                        })
            except json.JSONDecodeError:
                pass
                
    return fence_violation_images

if __name__ == "__main__":
    images = get_fence_violations()
    print(f"Found {len(images)} images with fence/edge protection violations.")
    for img in images[:]:
        print(img["image"])
