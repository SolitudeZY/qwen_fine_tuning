import json

response = """- 图像显示的是一个在山坡上进行的基坑或边坡施工场景，有大型施工机械（起重机）和施工材料（模板、钢筋等）。
- 从图中可以看到，施工区域的边坡有明显的土方开挖，且有施工平台和临时设施，存在临边作业的可能。
- 有工人在施工平台上作业，且未设置有效的临边防护，存在高坠风险。
- 施工现场的边坡未设置有效的防护设施，如安全网、防护栏杆等，存在安全隐患。
- 有施工人员在临边作业，未佩戴安全帽，未采取必要的安全措施，存在高坠风险。
- 施工现场的边坡未设置有效的防护设施，存在安全隐患。
"""

json_str = response
json_start = response.rfind("{")
json_end = response.rfind("}")
if json_start != -1 and json_end != -1 and json_end > json_start:
    json_str = response[json_start:json_end+1]

try:
    result = json.loads(json_str)
    print("Success")
except json.JSONDecodeError:
    print("Decode Error")
    print(f"JSON string to parse was: {repr(json_str)}")
