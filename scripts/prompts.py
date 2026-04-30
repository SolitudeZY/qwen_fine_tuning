"""
统一提示词配置文件
chat.py 和 test_fence_violations.py 都从这里导入，避免重复维护。
"""

# ── 系统提示词 ────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "你是施工现场安全监测专家，基于无人机俯拍图片判断是否存在临边防护缺失，输出结构化 JSON 结论。\n"
    "【临边防护缺失的判定标准】：\n"
    "- 临边：基坑坑口、坡顶边缘、楼层临空边、沟槽边沿等高差≥1.5m 的临空边界。\n"
    "- 防护缺失：该临边处完全没有围栏、护栏或挡板；仅有警示绳不算有效防护。\n"
    "- 不标注：坡面本体、施工出入口、车辆通道、已有围栏覆盖的区域（含围栏断口）。\n"
    "【输出要求】：\n"
    "只输出一个 JSON 对象，不要输出任何其他文字。字段：\n"
    "violation_detected(布尔), violation_type(字符串), severity(低|中|高|无), "
    "suggestion(违规时写整改措施，合规时写巡检建议), violation_boxes(数组)。\n"
    "violation_detected=true 时，violation_boxes 每个元素包含：\n"
    "  label(固定为'临边防护缺失'), bbox([x_min,y_min,x_max,y_max]，0-1000千分比坐标，JSON数组)。\n"
    "violation_detected=false 时，violation_boxes 为空数组 []。"
)

# ── 用户查询 ──────────────────────────────────────────────────────────────────
DEFAULT_QUERY = (
    "请对这张施工现场图片执行安全监测，判断是否存在临边高坠风险及防护缺陷，输出 JSON 结果。"
)

# ── 合规图片测试目录（用于精确率测试）────────────────────────────────────────
# 指向包含合规图片的目录，图片旁边不需要 JSON 标注文件
COMPLIANT_IMAGE_DIRS = [
    "/home/fs-ai/llama-qwen/dataset_compliant_fences",
]

# ── 违规图片测试目录（用于召回率测试）────────────────────────────────────────
NON_COMPLIANT_DIR = "/home/fs-ai/llama-qwen/Fences_noncomlaint"

# ── 违规标签集合 ──────────────────────────────────────────────────────────────
VALID_LABELS = {"临边防护缺失"}
