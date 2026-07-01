"""
环水保: Qwen3.5-2B 测试和推理脚本 (支持 MQTT)
用法:
  python scripts/qwen3.5-water/04_mqtt_inference.py \
      --model-path /home/fs-ai/llama-qwen/models/Qwen/Qwen3.5-2B \
      --lora-path /home/fs-ai/llama-qwen/outputs/qwen3.5-water/train/... \
      --mqtt-serve

2026-06-16 best model:  /home/fs-ai/llama-qwen/outputs/qwen3.5-water/train/v1-20260616-085800/checkpoint-600
"""

import argparse
import base64
import binascii
import json
import os
import queue
import tempfile
import threading
import time
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore", message=".*pad_token_id.*")

DEFAULT_MQTT_BROKER = "218.3.126.49"
DEFAULT_MQTT_PORT = 1883
DEFAULT_MQTT_REQUEST_TOPIC = "water/detect/request"
DEFAULT_MQTT_RESPONSE_TOPIC = "water/detect/result"
DEFAULT_MQTT_OSD_TOPIC = "thing/product/+/osd"

# ==========================================
# 日志配置
# ==========================================
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "mqtt_water.log"

logger = logging.getLogger("MQTT_Water")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 输出到控制台
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 输出到文件 (按大小滚动，最大10MB，保留5个备份)
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# ==========================================

# 缓存各个无人机最新的 location 数据，key: stream_id, value: location dict
latest_locations = {}

SYSTEM_PROMPT = (
    "你是水环境保护专家，基于无人机俯拍图片判断水体状态是否正常，输出结构化 JSON 结论。\n"
    "【水体状态判定标准】：\n"
    "- water_status=\"normal\"：水体清澈或轻度浑浊，无大面积藻类、漂浮物、泡沫或异常颜色。\n"
    "- water_status=\"polluted\"：水面存在大面积藻类/浮萍、明显泡沫/油污、异常颜色（如发黑/发红/发绿）、大量漂浮垃圾、或水体严重浑浊。\n"
    "【输出要求】：\n"
    "只输出一个 JSON 对象，包含 water_status（\"normal\" 或 \"polluted\"） 和 description（一句话描述原因）。"
)
DEFAULT_QUERY = "<image>请分析这张水域图片，判断水体状态是否正常，以JSON格式输出结果。"

def load_qwen35(model_path: str, lora_path: str | None = None, device_map: str = "auto"):
    logger.info(f"加载基础模型: {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa"  # 避免 sm_120 的 flash_attn bug
    )
    # 彻底关闭 generation 的 pad_token_id 警告
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    if lora_path and os.path.exists(lora_path):
        logger.info(f"加载 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def _decode_base64_image_to_temp_file(image_base64: str, suffix: str = ".jpg") -> str:
    if image_base64.startswith("data:") and "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_base64.strip(), validate=True)
    fd, temp_path = tempfile.mkstemp(prefix="water_mqtt_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(image_bytes)
    return temp_path

def infer_qwen35(model, processor, image_path: str):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": DEFAULT_QUERY},
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=512)
    elapsed = time.time() - start_time
    
    generated_ids_trimmed = [
        out_ids[in_ids_len:] for in_ids_len, out_ids in zip([len(x) for x in inputs.input_ids], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    
    # 提取 JSON
    parsed = {}
    try:
        # 去除可能存在的 markdown block
        json_str = output_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON 解析失败: {e}\nRaw Output: {output_text}")
    
    return parsed, output_text, elapsed

def process_mqtt_payload(payload_data: dict, model, processor) -> dict:
    image_base64 = payload_data.get("img") or payload_data.get("image_base64")
    if not image_base64:
        raise ValueError("缺少图片字段 (img/image_base64)")
    
    stream_id = payload_data.get("stream_id", "")
    
    # 尝试从 OSD 缓存中获取最新的 location 信息，若没有则用 payload 自带的，或默认无效
    location = payload_data.get("location")
    if stream_id in latest_locations:
        location = latest_locations[stream_id]
    if not location:
        location = {"valid": False}
        
    temp_path = None
    try:
        temp_path = _decode_base64_image_to_temp_file(image_base64)
        parsed, raw_output, elapsed = infer_qwen35(model, processor, temp_path)
        
        water_status = parsed.get("water_status", "unknown")
        
        return {
            "success": True,
            "processing_time_sec": round(elapsed, 3),
            "water_status": water_status,
            "description": parsed.get("description", ""),
            "stream_id": stream_id,
            "time": payload_data.get("time", ""),
            "location": location,
            "img": image_base64 if water_status == "polluted" else "",
            "raw_output": raw_output
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def run_mqtt_service(args, model, processor):
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise ImportError("缺少 paho-mqtt，请先安装: pip install paho-mqtt") from exc

    client = mqtt.Client()
    
    def on_connect(c, userdata, flags, rc):
        logger.info(f"已连接 broker={args.mqtt_broker}:{args.mqtt_port}, 订阅请求={args.topic_req}, 订阅OSD={DEFAULT_MQTT_OSD_TOPIC}")
        c.subscribe(args.topic_req)
        c.subscribe(DEFAULT_MQTT_OSD_TOPIC)

    def on_message(c, userdata, msg):
        try:
            payload_text = msg.payload.decode("utf-8")
            payload_data = json.loads(payload_text)
            
            # 如果收到的是 OSD 信息，则更新位置缓存并返回
            if msg.topic.endswith("/osd"):
                data = payload_data.get("data", {})
                sn = data.get("sn")
                host = data.get("host", {})
                if sn and "latitude" in host and "longitude" in host and "height" in host:
                    latest_locations[sn] = {
                        "valid": True,
                        "gps": {
                            "height": host.get("height"),
                            "latitude": host.get("latitude"),
                            "longitude": host.get("longitude")
                        }
                    }
                return

            logger.info(f"收到请求 topic={msg.topic}")
            result = process_mqtt_payload(payload_data, model, processor)
            
            if result.get("water_status") == "polluted":
                pub_topic = args.topic_res
                c.publish(pub_topic, json.dumps(result, ensure_ascii=False))
                logger.warning(f"已回复报警 topic={pub_topic}, status={result['water_status']}, time={result['processing_time_sec']}s")
            else:
                logger.info(f"检测结果为合规 (status={result.get('water_status')})，忽略回传, time={result.get('processing_time_sec')}s")
                
        except Exception as e:
            logger.error(f"消息处理失败: {e}")
            error_resp = {"success": False, "error": str(e)}
            c.publish(args.topic_res, json.dumps(error_resp, ensure_ascii=False))

    client.on_connect = on_connect
    client.on_message = on_message

    logger.info(f"启动服务中...")
    client.connect(args.mqtt_broker, args.mqtt_port, 60)
    client.loop_forever()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="基础模型路径")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA 路径")
    parser.add_argument("--image-path", type=str, default=None, help="单图推理测试")
    
    # MQTT Config
    parser.add_argument("--mqtt-serve", action="store_true", help="启动 MQTT 服务")
    parser.add_argument("--mqtt-broker", type=str, default=DEFAULT_MQTT_BROKER)
    parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT)
    parser.add_argument("--topic-req", type=str, default=DEFAULT_MQTT_REQUEST_TOPIC)
    parser.add_argument("--topic-res", type=str, default=DEFAULT_MQTT_RESPONSE_TOPIC)
    
    args = parser.parse_args()

    model, processor = load_qwen35(args.model_path, args.lora_path)

    if args.mqtt_serve:
        run_mqtt_service(args, model, processor)
    elif args.image_path:
        parsed, raw, elapsed = infer_qwen35(model, processor, args.image_path)
        logger.info(f"\n[测试结果] 耗时: {elapsed:.2f}s")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
    else:
        logger.error("请指定 --mqtt-serve 或 --image-path")

if __name__ == "__main__":
    main()
