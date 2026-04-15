import os
import json
import random
import shutil

DATA_PATH = "/home/fs-ai/llama-qwen/data/train_balanced_clean.jsonl"
OUTPUT_DIR = "/home/fs-ai/llama-qwen/dataset_compliant_fences"

def extract_compliant_images():
    """
    从数据集中提取判定为合规(violation_detected=false)的图片，
    并筛选出其中确实包含围栏相关描述的，
    复制到独立目录中供标注使用。
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compliant_candidates = []
    
    print(f"[*] 正在从 {DATA_PATH} 筛选合规图片...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            try:
                ans = json.loads(item["messages"][-1]["content"])
                if ans.get("violation_detected") is False:
                    # 检查内容中是否提到了防护设施
                    obs = " ".join(ans.get("key_observations", []))
                    res = ans.get("monitoring_result", "")
                    content = ans.get("monitoring_content", "")
                    text = obs + res + content
                    
                    if "围栏" in text or "护栏" in text or "围挡" in text or "防护" in text:
                        img_path = item["images"][0]
                        if os.path.exists(img_path):
                            # 确保也有对应的 JSON 文件
                            json_path = os.path.splitext(img_path)[0] + ".json"
                            if os.path.exists(json_path):
                                compliant_candidates.append((img_path, json_path))
            except Exception as e:
                pass
                
    print(f"[*] 找到 {len(compliant_candidates)} 张可能包含合规围栏的图片。")
    
    # 我们抽取大约与违规图片相等数量的图片（比如 150 张）
    target_count = min(150, len(compliant_candidates))
    selected = random.sample(compliant_candidates, target_count)
    
    print(f"[*] 随机抽取了 {target_count} 张，正在复制到 {OUTPUT_DIR} ...")
    
    for img_path, json_path in selected:
        img_dest = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
        json_dest = os.path.join(OUTPUT_DIR, os.path.basename(json_path))
        
        # 复制文件
        shutil.copy2(img_path, img_dest)
        shutil.copy2(json_path, json_dest)
        
    print(f"[*] 提取完成！你可以使用 LabelMe 打开 {OUTPUT_DIR} 进行合规数据的标注。")

if __name__ == "__main__":
    extract_compliant_images()
