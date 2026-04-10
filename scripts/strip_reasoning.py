import json
import re

INPUT_FILE = "/home/fs-ai/llama-qwen/data/finetune_fences.jsonl"
OUTPUT_FILE = "/home/fs-ai/llama-qwen/data/finetune_fences_json_only.jsonl"

def extract_json_only(response_text):
    """
    使用与 chat.py 中相同的逻辑提取 JSON。
    """
    json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()
        
    end = response_text.rfind("}")
    if end != -1:
        start = response_text.rfind("{", 0, end)
        while start != -1:
            candidate = response_text[start:end+1]
            try:
                candidate_fixed = candidate.replace("'", '"')
                json.loads(candidate_fixed) # 验证一下是不是有效的 JSON
                return candidate_fixed.strip()
            except json.JSONDecodeError:
                start = response_text.rfind("{", 0, start)
                
    return None

def main():
    cleaned_records = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            # 找到 assistant 的回答
            for msg in record.get("messages", []):
                if msg["role"] == "assistant":
                    original_content = msg["content"]
                    
                    # 剥离推理过程，只保留 JSON
                    json_str = extract_json_only(original_content)
                    if json_str:
                        msg["content"] = json_str
                    else:
                        print(f"[!] 无法从记录中提取 JSON: {record['images'][0]}")
                        
                # 同时也需要修改 system prompt，告诉它不需要输出推理过程了
                elif msg["role"] == "system":
                    original_system = msg["content"]
                    # 简单粗暴地替换掉要求
                    new_system = original_system.replace("1. 首先，你需要详细描述图片中危险区域的状态，并进行安全推理。\n2. 然后，你必须在回答的最后严格以 JSON 格式输出判定结果。", "请直接严格以 JSON 格式输出判定结果，不要输出任何其他的推理过程或解释文本。")
                    new_system = new_system.replace("输出安全推理与 JSON 结果", "输出 JSON 结果")
                    msg["content"] = new_system
                    
                # 也要修改 user prompt
                elif msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, str):
                        msg["content"] = content.replace("并输出安全推理与 JSON 结果", "并直接输出 JSON 结果")
                    
            cleaned_records.append(record)
            
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in cleaned_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print(f"[*] 成功生成只包含 JSON 输出的训练集: {OUTPUT_FILE}")
    print(f"[*] 共处理了 {len(cleaned_records)} 条数据。")

if __name__ == "__main__":
    main()
