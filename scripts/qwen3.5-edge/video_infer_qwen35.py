"""
Qwen3.5 视频抽帧与违规推理脚本

特点:
  1. 动态帧差法抽帧：画面变化不大时复用上一帧推理结果，极大节省推理时间
  2. 连续多帧滑窗判定：消除单帧运动模糊造成的误报
  3. 复用 chat_qwen35.py 的模型加载与推理逻辑
  4. 自动生成 JSON 违规报告并画框保存

使用方法：
bash -lc 'source /home/fs-ai/miniconda3/etc/profile.d/conda.sh && \
conda activate qwen35-ft-rollback && \
python /home/fs-ai/llama-qwen/scripts/qwen3.5-train/video_infer_qwen35.py \
  --mqtt-serve \
  --mqtt-publish-mode immediate \
  --model-path /home/fs-ai/llama-qwen/outputs/stage1_merged/qwen3.5/merged-20260609-164452 \
  --lora-path /home/fs-ai/llama-qwen/outputs/stage2_json/qwen3.5/v2-20260609-164542/checkpoint-500'
"""

import argparse
import base64
import binascii
import cv2
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import queue
import shutil
import sys
import threading
import time
import tempfile
import datetime
import warnings
warnings.filterwarnings("ignore", message=".*pad_token_id.*")
from collections import deque
from pathlib import Path
import numpy as np

# ==========================================
# 日志配置
# ==========================================
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / "mqtt_edge.log"

logger = logging.getLogger("MQTT_Edge")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 输出到控制台
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 输出到文件
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
# ==========================================

# 将当前目录和上一级加入 sys.path 以便复用现有模块
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from chat_qwen35 import (
    load_qwen35,
    infer_qwen35,
    draw_violation_boxes,
    DEFAULT_MODEL_PATH,
    DEFAULT_QUERY,
    SYSTEM_PROMPT
)

DEFAULT_VIDEO_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "test-video")
DEFAULT_MQTT_BROKER = "218.3.126.49"
DEFAULT_MQTT_PORT = 1883
DEFAULT_MQTT_REQUEST_TOPIC = "jiahe/frame/+"
DEFAULT_MQTT_RESPONSE_TOPIC = "jiahe/alertsNonRealtime/{stream_id}"
DEFAULT_MQTT_OSD_TOPIC = "thing/product/+/osd"
DEFAULT_MQTT_PUBLISH_MODE = "immediate"
DEFAULT_MQTT_FLUSH_INTERVAL = 5.0
DEFAULT_MQTT_QUEUE_SIZE = 100

# 缓存各个无人机最新的 location 数据，key: stream_id (sn), value: location dict
latest_locations = {}


def _strip_data_uri_prefix(image_base64: str) -> str:
    """兼容 data:image/jpeg;base64,... 这类前缀。"""
    if image_base64.startswith("data:") and "," in image_base64:
        return image_base64.split(",", 1)[1]
    return image_base64


def _decode_base64_image_to_temp_file(image_base64: str, suffix: str = ".jpg") -> str:
    """将 base64 图片写入临时文件，返回文件路径。"""
    normalized = _strip_data_uri_prefix(image_base64.strip())
    try:
        image_bytes = base64.b64decode(normalized, validate=True)
    except binascii.Error as exc:
        raise ValueError("image_base64 不是合法的 base64 图片数据") from exc

    fd, temp_path = tempfile.mkstemp(prefix="mqtt_image_", suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(image_bytes)
    return temp_path


def _encode_image_file_to_base64(image_path: str) -> str:
    """读取图片文件并转为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _load_image_size(image_path: str) -> tuple[int, int]:
    """读取图片尺寸，返回 width, height。"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片尺寸: {image_path}")
    height, width = image.shape[:2]
    return width, height


def _relative_bbox_to_pixel_bbox(bbox: list, width: int, height: int) -> list[int]:
    """将 0-1000 千分比坐标转为像素坐标。"""
    if len(bbox) != 4:
        return []
    x0 = max(0, min(width, int(round(float(bbox[0]) / 1000 * width))))
    y0 = max(0, min(height, int(round(float(bbox[1]) / 1000 * height))))
    x1 = max(0, min(width, int(round(float(bbox[2]) / 1000 * width))))
    y1 = max(0, min(height, int(round(float(bbox[3]) / 1000 * height))))
    return [x0, y0, x1, y1]


def _normalize_confidence(box: dict) -> float:
    """兼容 confidence / score 字段，不存在时给默认值。"""
    value = box.get("confidence", box.get("score", 1.0))
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 1.0
    return round(max(0.0, min(1.0, confidence)), 3)


def process_image_base64(
    image_base64: str,
    model,
    processor,
    query: str,
    system_prompt: str,
    max_new_tokens: int,
    include_visualized_image: bool = True,
) -> dict:
    """
    处理单张 base64 图片，返回可直接用于 MQTT 回传的 JSON 结果。

    返回字段:
      - report: 结构化检测报告
      - annotated_image_base64: 画框后的图片 base64；无框时返回原图
      - raw_model_output: 原始模型输出，便于排障
    """
    input_image_path = None
    annotated_image_path = None
    overall_start = time.time()

    try:
        input_image_path = _decode_base64_image_to_temp_file(image_base64)
        image_width, image_height = _load_image_size(input_image_path)
        parsed, json_str, raw, elapsed = infer_qwen35(
            model=model,
            processor=processor,
            image_path=input_image_path,
            query=query,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

        success = parsed is not None
        boxes = parsed.get("violation_boxes", []) if parsed else []
        detections = []
        for box in boxes:
            pixel_bbox = _relative_bbox_to_pixel_bbox(box.get("bbox", []), image_width, image_height)
            if not pixel_bbox:
                continue
            class_name = box.get("label") or parsed.get("violation_type", "异常") if parsed else "异常"
            detections.append(
                {
                    "bbox": pixel_bbox,
                    "class_id": 0,
                    "class_name": class_name,
                    "confidence": _normalize_confidence(box),
                }
            )

        annotated_image_base64 = None
        annotated = False

        if include_visualized_image:
            if boxes:
                with tempfile.NamedTemporaryFile(prefix="mqtt_annotated_", suffix=".jpg", delete=False) as tmp_file:
                    annotated_image_path = tmp_file.name
                draw_violation_boxes(input_image_path, boxes, annotated_image_path)
                annotated_image_base64 = _encode_image_file_to_base64(annotated_image_path)
                annotated = True
            else:
                annotated_image_base64 = _encode_image_file_to_base64(input_image_path)

        report = {
            "success": success,
            "processing_time_sec": round(elapsed, 3),
            "violation_detected": parsed.get("violation_detected", False) if parsed else False,
            "violation_type": parsed.get("violation_type", "") if parsed else "",
            "severity": parsed.get("severity", "") if parsed else "",
            "suggestion": parsed.get("suggestion", "") if parsed else "",
            "violation_boxes": boxes,
            "detections": detections,
            "desc": parsed.get("suggestion", "") if parsed else "",
            "raw_json": json_str,
        }

        return {
            "success": success,
            "message": "ok" if success else "模型输出未能解析为 JSON",
            "image_annotated": annotated,
            "annotated_image_base64": annotated_image_base64,
            "report": report,
            "detections": detections,
            "desc": report["desc"],
            "raw_model_output": raw,
            "total_processing_time_sec": round(time.time() - overall_start, 3),
        }
    except Exception as exc:
        return {
            "success": False,
            "message": str(exc),
            "image_annotated": False,
            "annotated_image_base64": None,
            "report": {
                "success": False,
                "processing_time_sec": 0.0,
                "violation_detected": False,
                "violation_type": "",
                "severity": "",
                "suggestion": "",
                "violation_boxes": [],
                "detections": [],
                "desc": "处理失败",
                "raw_json": "",
            },
            "detections": [],
            "desc": "处理失败",
            "raw_model_output": "",
            "total_processing_time_sec": round(time.time() - overall_start, 3),
        }
    finally:
        if input_image_path and os.path.exists(input_image_path):
            os.remove(input_image_path)
        if annotated_image_path and os.path.exists(annotated_image_path):
            os.remove(annotated_image_path)


def process_mqtt_payload(
    payload,
    model,
    processor,
    query: str,
    system_prompt: str,
    max_new_tokens: int,
    include_visualized_image: bool = True,
) -> dict:
    """
    处理 MQTT 消息负载，返回与平台实时接口一致的结构。

    示例:
      {"img":"...","stream_id":"xxx","time":"2026-06-11T14:19:06.636","location":{"valid":false}}
    """
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("MQTT payload 必须是 JSON 对象或其字符串")

    image_base64 = payload.get("img") or payload.get("image_base64") or payload.get("image")
    if not image_base64:
        raise ValueError("MQTT payload 缺少 img/image_base64 字段")

    result = process_image_base64(
        image_base64=image_base64,
        model=model,
        processor=processor,
        query=query,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        include_visualized_image=include_visualized_image,
    )
    
    stream_id = payload.get("stream_id", "")
    
    # 尝试从 OSD 缓存中获取最新的 location 信息，若没有则用 payload 自带的，或默认无效
    location = payload.get("location")
    if stream_id in latest_locations:
        location = latest_locations[stream_id]
    if not location:
        location = {"valid": False}
        
    return {
        "desc": result.get("desc", ""),
        "detections": result.get("detections", []),
        "img": result.get("annotated_image_base64") or image_base64,
        "location": location,
        "stream_id": stream_id,
        "time": payload.get("time", ""),
    }


def build_mqtt_error_response(payload, error_message: str) -> dict:
    """构造 MQTT 协议错误返回，尽量保留原始上下文字段。"""
    payload_dict = {}
    if isinstance(payload, bytes):
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError:
            payload = ""
    if isinstance(payload, str):
        try:
            payload_dict = json.loads(payload)
        except Exception:
            payload_dict = {}
    elif isinstance(payload, dict):
        payload_dict = payload

    return {
        "desc": f"处理失败: {error_message}",
        "detections": [],
        "img": payload_dict.get("img", ""),
        "location": payload_dict.get("location", {"valid": False}),
        "stream_id": payload_dict.get("stream_id", ""),
        "time": payload_dict.get("time", ""),
    }


def extract_stream_id_from_topic(topic: str) -> str:
    """从请求 topic 提取末尾的 stream_id。"""
    if not topic:
        return ""
    return topic.rstrip("/").split("/")[-1]


def resolve_response_topic(response_topic_template: str, payload: dict, request_topic: str) -> str:
    """根据模板、payload 和请求 topic 解析最终回包 topic。"""
    stream_id = payload.get("stream_id") or extract_stream_id_from_topic(request_topic)
    if "{stream_id}" in response_topic_template:
        return response_topic_template.format(stream_id=stream_id)
    return response_topic_template


def prepare_mqtt_response(
    request_topic: str,
    raw_payload,
    response_topic_template: str,
    model,
    processor,
    query: str,
    system_prompt: str,
    max_new_tokens: int,
    include_visualized_image: bool,
) -> tuple[str, dict, str]:
    """处理单条 MQTT 消息并生成回包主题、回包内容和状态文本。"""
    publish_topic = response_topic_template
    try:
        if isinstance(raw_payload, bytes):
            payload_text = raw_payload.decode("utf-8")
        else:
            payload_text = str(raw_payload)
        payload_data = json.loads(payload_text)
        if isinstance(payload_data, dict) and not payload_data.get("stream_id"):
            payload_data["stream_id"] = extract_stream_id_from_topic(request_topic)

        if isinstance(payload_data, dict) and payload_data.get("response_topic"):
            publish_topic = str(payload_data["response_topic"])
        elif isinstance(payload_data, dict):
            publish_topic = resolve_response_topic(response_topic_template, payload_data, request_topic)

        response_payload = process_mqtt_payload(
            payload=payload_data,
            model=model,
            processor=processor,
            query=query,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            include_visualized_image=include_visualized_image,
        )
        return publish_topic, response_payload, "ok"
    except Exception as exc:
        error_payload = build_mqtt_error_response(raw_payload, str(exc))
        if not error_payload.get("stream_id"):
            error_payload["stream_id"] = extract_stream_id_from_topic(request_topic)
        publish_topic = resolve_response_topic(response_topic_template, error_payload, request_topic)
        return publish_topic, error_payload, f"error: {exc}"


def should_publish_alert_response(item: dict) -> bool:
    """仅对违规告警结果发布消息；合规结果不回传。"""
    if not item["status_text"].startswith("ok"):
        return True
    detections = item["response_payload"].get("detections", [])
    return bool(detections)


def run_mqtt_service(
    broker_host: str,
    broker_port: int,
    request_topic: str,
    response_topic: str,
    model,
    processor,
    query: str,
    system_prompt: str,
    max_new_tokens: int,
    include_visualized_image: bool,
    client_id: str | None = None,
    keepalive: int = 60,
    qos: int = 0,
    publish_mode: str = DEFAULT_MQTT_PUBLISH_MODE,
    flush_interval_sec: float = DEFAULT_MQTT_FLUSH_INTERVAL,
    queue_size: int = DEFAULT_MQTT_QUEUE_SIZE,
) -> None:
    """启动 MQTT 常驻服务，收到图片消息后同步推理并回传结果。"""
    try:
        import paho.mqtt.client as mqtt
    except ImportError as exc:
        raise ImportError("缺少 paho-mqtt，请先安装: pip install paho-mqtt") from exc

    inbound_queue: queue.Queue[dict] = queue.Queue(maxsize=max(queue_size, 1))
    outbound_queue: queue.Queue[dict] = queue.Queue()
    stop_event = threading.Event()

    def publish_response(client, item: dict) -> None:
        if not should_publish_alert_response(item):
            elapsed = time.time() - item["started_at"]
            logger.info(
                f"[MQTT] topic={item['request_topic']} 检测结果为合规，已忽略回传, "
                f"elapsed={elapsed:.2f}s"
            )
            return

        result = client.publish(
            item["publish_topic"],
            json.dumps(item["response_payload"], ensure_ascii=False),
            qos=qos,
        )
        elapsed = time.time() - item["started_at"]
        logger.warning(
            f"[MQTT] topic={item['request_topic']} -> {item['publish_topic']}, "
            f"status={item['status_text']}, publish_rc={result.rc}, elapsed={elapsed:.2f}s"
        )

    def worker_loop(_client) -> None:
        while not stop_event.is_set():
            try:
                item = inbound_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            publish_topic, response_payload, status_text = prepare_mqtt_response(
                request_topic=item["request_topic"],
                raw_payload=item["raw_payload"],
                response_topic_template=response_topic,
                model=model,
                processor=processor,
                query=query,
                system_prompt=system_prompt,
                max_new_tokens=max_new_tokens,
                include_visualized_image=include_visualized_image,
            )
            outbound_queue.put(
                {
                    "request_topic": item["request_topic"],
                    "publish_topic": publish_topic,
                    "response_payload": response_payload,
                    "status_text": status_text,
                    "started_at": item["received_at"],
                }
            )
            inbound_queue.task_done()

    def publisher_loop(client) -> None:
        pending_items = []
        last_flush_at = time.time()
        while not stop_event.is_set():
            timeout = 0.5
            if publish_mode == "interval":
                timeout = max(0.1, flush_interval_sec - (time.time() - last_flush_at))
            try:
                item = outbound_queue.get(timeout=timeout)
                if publish_mode == "immediate":
                    publish_response(client, item)
                else:
                    if not pending_items:
                        last_flush_at = time.time()
                    pending_items.append(item)
                outbound_queue.task_done()
            except queue.Empty:
                pass

            if publish_mode == "interval" and pending_items and (time.time() - last_flush_at >= flush_interval_sec):
                for pending in pending_items:
                    publish_response(client, pending)
                pending_items.clear()
                last_flush_at = time.time()

        for pending in pending_items:
            publish_response(client, pending)

    def on_connect(client, _userdata, _flags, reason_code, _properties=None):
        logger.info(
            f"[MQTT] 已连接 broker={broker_host}:{broker_port}, "
            f"reason_code={reason_code}, 订阅请求主题={request_topic}, 订阅OSD主题={DEFAULT_MQTT_OSD_TOPIC}"
        )
        client.subscribe(request_topic, qos=qos)
        client.subscribe(DEFAULT_MQTT_OSD_TOPIC, qos=qos)

    def on_message(client, _userdata, msg):
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

            inbound_queue.put_nowait(
                {
                    "request_topic": msg.topic,
                    "raw_payload": msg.payload,
                    "received_at": time.time(),
                }
            )
            logger.info(f"[MQTT] 已入队 topic={msg.topic}, queue_size={inbound_queue.qsize()}")
        except queue.Full:
            response_payload = build_mqtt_error_response(msg.payload, "服务忙，消息队列已满")
            response_payload["stream_id"] = response_payload.get("stream_id") or extract_stream_id_from_topic(msg.topic)
            publish_topic = resolve_response_topic(response_topic, response_payload, msg.topic)
            status_text = "error: queue full"
            response_payload["desc"] = "处理失败: 服务忙，消息队列已满"
            result = client.publish(
                publish_topic,
                json.dumps(response_payload, ensure_ascii=False),
                qos=qos,
            )
            logger.warning(
                f"[MQTT] topic={msg.topic} -> {publish_topic}, "
                f"status={status_text}, publish_rc={result.rc}, queue_full=True"
            )

    def on_disconnect(_client, _userdata, *args):
        reason_code = args[1] if len(args) >= 2 else (args[0] if args else "unknown")
        logger.warning(f"[MQTT] 连接断开，reason_code={reason_code}")

    client = mqtt.Client(client_id=client_id or "")
    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    worker_thread = threading.Thread(target=worker_loop, args=(client,), daemon=True, name="mqtt-infer-worker")
    publisher_thread = threading.Thread(target=publisher_loop, args=(client,), daemon=True, name="mqtt-publish-worker")
    worker_thread.start()
    publisher_thread.start()

    logger.info(
        f"[MQTT] 启动服务 broker={broker_host}:{broker_port}, "
        f"request_topic={request_topic}, response_topic={response_topic}, "
        f"publish_mode={publish_mode}, flush_interval_sec={flush_interval_sec}, queue_size={queue_size}"
    )
    try:
        client.connect(broker_host, broker_port, keepalive=keepalive)
        client.loop_forever()
    finally:
        stop_event.set()


def calculate_frame_diff(frame1, frame2):
    """计算两帧图像的绝对差异均值"""
    if frame1 is None or frame2 is None:
        return float('inf')
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    return np.mean(diff)


def process_video(
    video_path: str,
    model,
    processor,
    diff_threshold: float,
    window_size: int,
    fps_sample_rate: float,
    output_dir: str,
    visualize: bool,
    query: str,
    system_prompt: str,
    max_new_tokens: int
):
    start_process_time = time.time()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频文件: {video_path}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(video_fps / fps_sample_rate))
    
    print(f"\n[视频信息] 路径: {video_path}")
    print(f"[视频信息] 原始 FPS: {video_fps:.2f}, 总帧数: {total_frames}")
    print(f"[抽帧策略] 目标采样率: {fps_sample_rate} FPS (每隔 {frame_interval} 帧抽取1次)")
    print(f"[判决逻辑] 帧差阈值: {diff_threshold}, 滑窗大小: {window_size} 帧\n")

    # 构建输出目录
    video_name = Path(video_path).stem
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(output_dir, f"{video_name}-{current_time}")
    alert_frames_dir = os.path.join(run_output_dir, "alert_frames")
    os.makedirs(alert_frames_dir, exist_ok=True)

    # 临时目录，用于存放需要送入大模型推理的帧图片
    temp_dir = tempfile.mkdtemp(prefix="qwen_video_frames_")
    
    prev_frame = None
    prev_parsed_result = None
    
    window = deque(maxlen=window_size)
    violation_events = []
    current_event = None

    frame_idx = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            timestamp_sec = frame_idx / video_fps
            extracted_count += 1
            
            # 1. 动态帧差判断
            diff = calculate_frame_diff(prev_frame, frame)
            
            parsed = None
            if diff < diff_threshold and prev_parsed_result is not None:
                # 画面静止，跳过推理，复用上一次结果
                print(f"[时间 {timestamp_sec:05.1f}s] 帧差 {diff:04.1f} < {diff_threshold}，跳过推理。")
                parsed = prev_parsed_result
            else:
                # 画面有变化，写入临时图片并调用大模型
                tmp_img_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(tmp_img_path, frame)
                
                print(f"[时间 {timestamp_sec:05.1f}s] 帧差 {diff:04.1f} >= {diff_threshold}，触发大模型推理...", end="", flush=True)
                parsed, _, _, elapsed = infer_qwen35(
                    model, processor, tmp_img_path, query, system_prompt, max_new_tokens
                )
                print(f" 耗时: {elapsed:.1f}s, 结论: {'违规' if parsed and parsed.get('violation_detected') else '合规'}")
                
                prev_frame = frame.copy()
                prev_parsed_result = parsed
            
            # 2. 滑窗逻辑判定
            is_violation = parsed and parsed.get("violation_detected", False)
            window.append(bool(is_violation))
            
            # 判定规则：滑窗内超过半数帧为违规，即确认这段时间违规
            threshold = window_size // 2 + 1 if window_size > 1 else 1
            confirmed_violation = sum(window) >= threshold
            
            if confirmed_violation:
                if current_event is None:
                    current_event = {
                        "start_time": round(timestamp_sec, 1),
                        "end_time": round(timestamp_sec, 1),
                        "violation_type": parsed.get("violation_type", "未知") if parsed else "未知",
                        "severity": parsed.get("severity", "未知") if parsed else "未知",
                        "suggestion": parsed.get("suggestion", "") if parsed else "",
                        "boxes": parsed.get("violation_boxes", []) if parsed else [],
                        "frames": []
                    }
                else:
                    current_event["end_time"] = round(timestamp_sec, 1)
                    
                # 3. 可视化保存
                if visualize and is_violation and parsed and parsed.get("violation_boxes"):
                    tmp_img_path = os.path.join(temp_dir, f"frame_{frame_idx}.jpg")
                    if not os.path.exists(tmp_img_path):
                        cv2.imwrite(tmp_img_path, frame)
                    
                    out_name = f"violation_{timestamp_sec:.1f}s.jpg"
                    out_path = os.path.join(alert_frames_dir, out_name)
                    # 避免重复保存同一秒的帧
                    if out_path not in current_event["frames"]:
                        draw_violation_boxes(tmp_img_path, parsed["violation_boxes"], out_path)
                        current_event["frames"].append(out_path)
            else:
                if current_event is not None:
                    violation_events.append(current_event)
                    current_event = None
                    
        frame_idx += 1
        
    if current_event is not None:
        violation_events.append(current_event)
        
    cap.release()
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    end_process_time = time.time()
    total_process_time = end_process_time - start_process_time
    
    print(f"\n[处理完成] 视频提取 {extracted_count} 帧，最终判定 {len(violation_events)} 个违规时间段。")
    print(f"[处理完成] 总耗时: {total_process_time:.1f} 秒")
    
    report_data = {
        "video_path": os.path.abspath(video_path),
        "total_processing_time_sec": round(total_process_time, 1),
        "violation_events": violation_events
    }
    
    report_path = os.path.join(run_output_dir, "violation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"[报告已保存] {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3.5 视频/MQTT 图片推理")
    parser.add_argument("--video", type=str, default=None, help="视频文件路径")
    parser.add_argument("--image-base64", type=str, default=None, help="单张图片的 base64 字符串")
    parser.add_argument("--mqtt-payload-json", type=str, default=None, help="MQTT 消息 JSON 字符串")
    parser.add_argument("--mqtt-payload-file", type=str, default=None, help="保存 MQTT 消息 JSON 的文件路径")
    parser.add_argument("--mqtt-serve", action="store_true", help="以 MQTT 常驻服务模式运行")
    parser.add_argument("--mqtt-broker", type=str, default=DEFAULT_MQTT_BROKER, help="MQTT broker 地址")
    parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT, help="MQTT broker 端口")
    parser.add_argument("--mqtt-request-topic", type=str, default=DEFAULT_MQTT_REQUEST_TOPIC, help="订阅请求主题")
    parser.add_argument("--mqtt-response-topic", type=str, default=DEFAULT_MQTT_RESPONSE_TOPIC, help="发布响应主题")
    parser.add_argument("--mqtt-client-id", type=str, default=None, help="MQTT client_id")
    parser.add_argument("--mqtt-keepalive", type=int, default=60, help="MQTT keepalive 秒数")
    parser.add_argument("--mqtt-qos", type=int, default=0, help="MQTT QoS，默认 0")
    parser.add_argument(
        "--mqtt-publish-mode",
        type=str,
        choices=["immediate", "interval"],
        default=DEFAULT_MQTT_PUBLISH_MODE,
        help="MQTT 回传模式：immediate=每处理完一张立即返回，interval=按间隔批量返回",
    )
    parser.add_argument(
        "--mqtt-flush-interval",
        type=float,
        default=DEFAULT_MQTT_FLUSH_INTERVAL,
        help="MQTT interval 模式下的回传间隔秒数，默认 5 秒",
    )
    parser.add_argument(
        "--mqtt-queue-size",
        type=int,
        default=DEFAULT_MQTT_QUEUE_SIZE,
        help="MQTT 请求队列大小，默认 100",
    )
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA路径")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_VIDEO_OUTPUT_DIR, help="输出目录")
    
    # 视频处理相关参数
    parser.add_argument("--fps", type=float, default=1.0, help="抽帧目标帧率 (默认 1 FPS)")
    parser.add_argument("--diff-threshold", type=float, default=5.0, help="触发推理的帧差均值阈值 (默认 5.0)")
    parser.add_argument("--window-size", type=int, default=3, help="滑窗大小 (默认 3 帧)")
    
    parser.add_argument("--visualize", action="store_true", help="是否绘制违规检测框并保存")
    parser.add_argument("--skip-visualized-image", action="store_true", help="base64/MQTT 模式下不返回标注图 base64")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="推理提示词")
    parser.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT, help="系统提示词")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成长度")
    parser.add_argument("--device-map", type=str, default="auto", help="transformers device_map")
    parser.add_argument("--load-in-4bit", action="store_true", help="显式启用 4bit 量化")
    
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.video, args.image_base64, args.mqtt_payload_json, args.mqtt_payload_file, args.mqtt_serve]):
        parser.error("请至少提供 --video、--image-base64、--mqtt-payload-json 或 --mqtt-payload-file 之一")

    logger.info(f"[video_infer_qwen35] 加载模型: {args.model_path}")
    model, processor = load_qwen35(
        model_path=args.model_path,
        lora_path=args.lora_path,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )
    
    # 彻底关闭 generation 的 pad_token_id 警告
    if hasattr(model, "generation_config") and hasattr(model.generation_config, "eos_token_id"):
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    if args.lora_path:
        logger.info(f"[video_infer_qwen35] 加载 LoRA: {args.lora_path}")

    if args.mqtt_serve:
        run_mqtt_service(
            broker_host=args.mqtt_broker,
            broker_port=args.mqtt_port,
            request_topic=args.mqtt_request_topic,
            response_topic=args.mqtt_response_topic,
            model=model,
            processor=processor,
            query=args.query,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            include_visualized_image=not args.skip_visualized_image,
            client_id=args.mqtt_client_id,
            keepalive=args.mqtt_keepalive,
            qos=args.mqtt_qos,
            publish_mode=args.mqtt_publish_mode,
            flush_interval_sec=args.mqtt_flush_interval,
            queue_size=args.mqtt_queue_size,
        )
        return

    if args.video:
        process_video(
            video_path=args.video,
            model=model,
            processor=processor,
            diff_threshold=args.diff_threshold,
            window_size=args.window_size,
            fps_sample_rate=args.fps,
            output_dir=args.output_dir,
            visualize=args.visualize,
            query=args.query,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens
        )
        return

    if args.image_base64:
        result = process_image_base64(
            image_base64=args.image_base64,
            model=model,
            processor=processor,
            query=args.query,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            include_visualized_image=not args.skip_visualized_image,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.mqtt_payload_json or args.mqtt_payload_file:
        payload = args.mqtt_payload_json
        if args.mqtt_payload_file:
            with open(args.mqtt_payload_file, "r", encoding="utf-8") as f:
                payload = f.read()
        result = process_mqtt_payload(
            payload=payload,
            model=model,
            processor=processor,
            query=args.query,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            include_visualized_image=not args.skip_visualized_image,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
