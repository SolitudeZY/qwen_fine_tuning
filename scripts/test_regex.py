import json
import re

def test_extract(response: str):
    # 2. 尝试从后往前寻找能够成功 parse 的完整 JSON 对象
    end = response.rfind("}")
    if end != -1:
        start = response.rfind("{", 0, end)
        while start != -1:
            candidate = response[start:end+1]
            try:
                candidate = candidate.replace("'", '"')
                return "JSON Method", json.loads(candidate)
            except json.JSONDecodeError:
                start = response.rfind("{", 0, start)
                
    fallback_json = {}
    if "violation_detected: true" in response.lower() or '"violation_detected": true' in response.lower():
        fallback_json["violation_detected"] = True
        
    if "violation_detected" in fallback_json:
        if "violation_boxes" not in fallback_json:
            box_objects = re.findall(r'\{\s*["\']label["\']\s*:\s*["\'](.*?)["\']\s*,\s*["\']bbox["\']\s*:\s*(\[[^\]]+\])\s*\}', response)
            if box_objects:
                parsed_boxes = []
                for label, bbox_str in box_objects:
                    try:
                        bbox = json.loads(bbox_str)
                        parsed_boxes.append({"label": label, "bbox": bbox})
                    except:
                        pass
                if parsed_boxes:
                    fallback_json["violation_boxes"] = parsed_boxes
        return "Regex Method", fallback_json
    return "Failed", None

sample = """
violation_boxes:  
[{"label": "围栏断口", "bbox": [130, 300, 450, 680]}]  
[{"label": "围栏未闭合", "bbox": [120, 450, 750, 750]}] 

JSON 结果： 
------------------------------------------------------------ 
  状态: !! 违规 !! 
  类型: 临边防护不连续 
  严重程度: 中等 
  建议: 立即修复围栏断口和未闭合区域，确保所有临边均设置连续、牢固的防护栏杆，并加装踢脚板和警示标志，防止高坠事故。
"""

method, result = test_extract(sample)
print(f"Method: {method}")
print(f"Result: {json.dumps(result, ensure_ascii=False, indent=2)}")
