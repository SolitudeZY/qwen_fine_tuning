"""
调用阿里云百炼 (DashScope) Qwen-VL-Max API 进行批量自动打标

由于本地 7B 模型能力不足，经常混淆基坑与边坡，此脚本转为调用线上视觉模型进行监测标注。

功能:
1. 读取 dataset 目录下所有的图片 (.jpg, .png)
2. 直接将图片和 Prompt 发送给 Qwen-VL-Max
3. 解析 API 返回的结构化 JSON 监测结果
4. 保存为中间态 JSON 文件 (可供 label_ui.py 修改，也可直接转为 ms-swift)
5. 支持断点续传
"""

import os
import pathlib
import sys
import pathlib

# 解决 dashscope 已安装但当前 Python 解释器找不到的问题
# 将 dashscope 所在真实 site-packages 目录临时注入 sys.path
dashscope_site = pathlib.Path("/home/fs-ai/miniconda3/envs/qwen-ft/lib/python3.11/site-packages")
if str(dashscope_site) not in sys.path:
    sys.path.insert(0, str(dashscope_site))

# 再次尝试导入；若仍失败则给出友好提示
try:
    from dashscope import MultiModalConversation
except ModuleNotFoundError as e:
    print("当前 Python 环境仍无法找到 dashscope，请确认：")
    print("1. 您运行脚本时使用的 Python 解释器与安装 dashscope 的环境一致；")
    print("2. 或者直接在安装 dashscope 的环境下运行脚本。")
    print("原始报错：", e)
    sys.exit(1)

import json
import glob
import time
from tqdm import tqdm
from dashscope import MultiModalConversation

# ============ 核心配置 ============
DATASET_DIR = "/home/fs-ai/llama-qwen/dataset"
OUTPUT_JSON = "/home/fs-ai/llama-qwen/outputs/api_annotations_cot.json"
MODEL_NAME = "qwen-vl-max"

# 必须在环境变量中设置 DASHSCOPE_API_KEY
API_KEY = os.environ.get("DASHSCOPE_API_KEY")

SYS_PROMPT = (
    "你是一名严谨的边坡与基坑施工安全监测专家，负责基于施工现场图片输出可落地的监测结论。\n"
    "你的任务不是泛泛描述图片，而是围绕高坠风险和临边防护进行监测判定，并输出结构化结果。\n"
    "如果图片中危险边缘、人员可达性或防护设施状态看不清，请优先输出 uncertain，不要为了给出结论而过度推断。\n"
    "\n"
    "【场景识别要求】\n"
    "1. 先判断监测对象更接近：foundation_pit(基坑)、slope(边坡)、platform_or_edge(高处平台/临边)、mixed(混合场景)、unknown(无法判断)。\n"
    "2. 必须区分“边坡坡面”与“需防护的危险临边”：\n"
    "- 基坑：向下开挖形成的坑槽、坑洞、沟槽，坑边、通道边、临空边缘通常需要防护隔离。\n"
    "- 边坡：倾斜土坡、岩坡、开挖坡面本体通常不是必须全封闭围挡的对象，但坡顶作业平台、临空边、通行边、临边作业面若存在坠落风险，仍需防护。\n"
    "- 如果只是可见大面积施工坡面，且看不到明确临边作业平台或坠落边缘，不要仅凭“是边坡”就判为违规。\n"
    "\n"
    "【重点监测内容】\n"
    "请重点关注以下可见内容：\n"
    "- 是否存在明确的高坠危险边缘、坑边、孔洞边、通道边、平台临边。\n"
    "- 是否设置了围栏、护栏、围挡、警戒线、踢脚板、盖板或其他物理隔离。\n"
    "- 防护设施是否完整、连续、明显破损、倒伏、缺段、变形或形同虚设。\n"
    "- 是否存在人员可接近危险边缘但无有效隔离的情况。\n"
    "- 是否因画面遮挡、距离过远、夜间、模糊而无法可靠判断。\n"
    "\n"
    "【判定原则】\n"
    "1. 只基于图中可见证据判断，不要臆测画面外信息。\n"
    "2. 不能确认时输出 uncertain，不要强行给出违规结论。\n"
    "3. 如果同时存在多个问题，给出 multiple_issues。\n"
    "4. 如画面主要体现施工监测对象状态而非明显违规，也应输出监测内容摘要和监测结果摘要。\n"
    "\n"
    "【输出要求】\n"
    "只输出一个 JSON 对象，不要输出 markdown，不要输出代码块，不要额外解释。\n"
    "JSON 必须包含以下字段：\n"
    "- scene_type: foundation_pit | slope | platform_or_edge | mixed | unknown\n"
    "- monitoring_content: 字符串，概括本图正在监测的对象和内容\n"
    "- monitoring_result: 字符串，给出一句简洁监测结论\n"
    "- key_observations: 字符串数组，列出 2-5 条基于图像的关键观察\n"
    "- risk_points: 字符串数组，列出风险点；无明显风险时返回空数组\n"
    "- compliance_status: compliant | non_compliant | uncertain\n"
    "- violation_detected: 布尔值；无法判断时为 null\n"
    "- violation_type: normal | no_edge_protection | fence_damaged | guardrail_deformed | warning_missing | unsafe_access | multiple_issues | uncertain\n"
    "- severity: low | medium | high | critical | unknown\n"
    "- confidence: low | medium | high\n"
    "- suggestion: 字符串，给出具体整改或复核建议\n"
    "\n"
    "【一致性要求】\n"
    "- compliance_status=compliant 时，violation_detected=false, violation_type=normal。\n"
    "- compliance_status=non_compliant 时，violation_detected=true。\n"
    "- compliance_status=uncertain 时，violation_detected=null, violation_type=uncertain, severity=unknown。\n"
)

USER_PROMPT = (
    "请对这张施工现场图片执行边坡/基坑安全监测，识别场景类型，提取关键观察，"
    "判断是否存在临边高坠风险及防护缺陷，并严格只输出 JSON 对象。"
)

REQUIRED_JSON_KEYS = {
    "scene_type": "unknown",
    "monitoring_content": "",
    "monitoring_result": "",
    "key_observations": [],
    "risk_points": [],
    "compliance_status": "uncertain",
    "violation_detected": None,
    "violation_type": "uncertain",
    "severity": "unknown",
    "confidence": "low",
    "suggestion": "请结合现场复核图像中危险边缘、防护设施连续性和人员可达性。"
}


def normalize_vlm_json(data: dict) -> dict:
    """补全字段并尽量维持监测输出的一致性。"""
    normalized = dict(REQUIRED_JSON_KEYS)
    if isinstance(data, dict):
        normalized.update(data)

    if not isinstance(normalized.get("key_observations"), list):
        normalized["key_observations"] = [str(normalized["key_observations"])]
    if not isinstance(normalized.get("risk_points"), list):
        normalized["risk_points"] = [str(normalized["risk_points"])]

    status = normalized.get("compliance_status")
    if status == "compliant":
        normalized["violation_detected"] = False
        normalized["violation_type"] = "normal"
        if normalized.get("severity") == "unknown":
            normalized["severity"] = "low"
    elif status == "non_compliant":
        normalized["violation_detected"] = True
        if normalized.get("violation_type") in ("normal", "uncertain", "", None):
            normalized["violation_type"] = "multiple_issues" if len(normalized["risk_points"]) > 1 else "no_edge_protection"
        if normalized.get("severity") == "unknown":
            normalized["severity"] = "medium"
    else:
        normalized["compliance_status"] = "uncertain"
        normalized["violation_detected"] = None
        normalized["violation_type"] = "uncertain"
        normalized["severity"] = "unknown"

    return normalized


def extract_response_text(response) -> str:
    """兼容 DashScope 返回的不同 content 结构。"""
    try:
        content = response.output.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(part for part in text_parts if part).strip()
    except Exception:
        pass
    return ""


def call_qwen_vl_api(image_path: str, max_retries: int = 3) -> dict:
    """调用阿里云 Qwen-VL-Max 接口，包含错误重试机制"""
    messages = [
        {
            "role": "system",
            "content": [{"text": SYS_PROMPT}]
        },
        {
            "role": "user",
            "content": [
                {"image": f"file://{os.path.abspath(image_path)}"},
                {"text": USER_PROMPT}
            ]
        }
    ]

    for attempt in range(max_retries):
        try:
            response = MultiModalConversation.call(
                model=MODEL_NAME,
                messages=messages,
                api_key=API_KEY
            )

            if response.status_code == 200:
                response_text = extract_response_text(response)
                if not response_text:
                    return {"success": False, "error": "API返回成功但未提取到文本内容"}
                return {"success": True, "text": response_text}
            else:
                error_msg = f"API Error: {response.code} - {response.message}"
                if attempt < max_retries - 1:
                    print(f"\n[Attempt {attempt+1}/{max_retries}] API返回错误，3秒后重试: {error_msg}")
                    time.sleep(3)
                    continue
                return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                print(f"\n[Attempt {attempt+1}/{max_retries}] 发生网络或SSL异常，3秒后重试: {error_msg}")
                time.sleep(3)
                continue
            return {"success": False, "error": error_msg}

    return {"success": False, "error": "达到最大重试次数"}


def parse_vlm_output(output_text: str) -> dict:
    """尽量从模型输出中提取 JSON，并补全缺失字段。"""
    cleaned_text = output_text.strip()
    json_result = None

    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`").strip()
        if cleaned_text.startswith("json"):
            cleaned_text = cleaned_text[4:].strip()

    try:
        json_result = json.loads(cleaned_text)
    except Exception:
        json_start = cleaned_text.find("{")
        json_end = cleaned_text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                json_result = json.loads(cleaned_text[json_start:json_end + 1])
            except Exception:
                json_result = None

    if json_result is None:
        json_result = normalize_vlm_json({
            "monitoring_content": "模型未返回可解析的结构化监测结果",
            "monitoring_result": "本次结果解析失败，需人工复核原始输出",
            "key_observations": [],
            "risk_points": [],
            "raw": output_text,
        })
    else:
        json_result = normalize_vlm_json(json_result)

    return {"cot_reasoning": "", "json_output": json_result, "raw_response": output_text}


def main(max_samples=None):
    if not API_KEY:
        print("错误: 未找到 DASHSCOPE_API_KEY 环境变量！")
        print("请运行: export DASHSCOPE_API_KEY='your-key-here'")
        return

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # 1. 收集所有图片 (不管有没有 json)
    print("正在搜集图片...")
    img_files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"):
        img_files.extend(glob.glob(os.path.join(DATASET_DIR, "**", ext), recursive=True))

    img_files = sorted(list(set(img_files)))
    print(f"共找到 {len(img_files)} 张图片。")

    # 2. 读取已处理的结果（断点续传）
    results = []
    processed_paths = set()
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
            processed_paths = {r["image_path"] for r in results}
        print(f"已加载 {len(processed_paths)} 条历史处理记录。")

    # 3. 过滤出未处理的图片
    to_process = [p for p in img_files if p not in processed_paths]
    print(f"剩余 {len(to_process)} 张图片待处理。")

    if max_samples:
        to_process = to_process[:max_samples]
        print(f"\n[测试模式] 本次仅处理前 {max_samples} 张图片。")

    if not to_process:
        print("所有图片均已处理完毕！")
        return

    # 4. 开始调用 API
    print("\nStarting Qwen-VL-Max API Annotation...")
    for i, img_path in enumerate(tqdm(to_process)):
        api_res = call_qwen_vl_api(img_path)

        if not api_res["success"]:
            print(f"\n[Error] {img_path}: {api_res['error']}")
            # API 报错（限流等），休息一会重试
            time.sleep(3)
            continue

        response_text = api_res["text"]
        vlm_parsed = parse_vlm_output(response_text)

        # 构造并保存记录
        record = {
            "image_path": img_path,
            "original_labels": [],  # 纯API不再依赖原标注
            "vlm_reasoning": vlm_parsed["cot_reasoning"],
            "vlm_json": vlm_parsed["json_output"],
            "raw_response": vlm_parsed["raw_response"]
        }
        results.append(record)

        # 每次成功都保存（防中断）
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 商业 API 并发限流（免费额度/低级权限每秒 1-2 次，稍微 sleep 一下防 429）
        time.sleep(1.5)

    print(f"\n打标完成！总结果数: {len(results)}")
    print(f"结果已保存至: {OUTPUT_JSON}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=10, help="运行测试的图片数量")
    parser.add_argument("--all", action="store_true", help="跑全量数据")
    args = parser.parse_args()

    max_samples = None if args.all else args.test
    main(max_samples=max_samples)
