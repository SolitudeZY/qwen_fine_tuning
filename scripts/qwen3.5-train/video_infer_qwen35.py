"""
Qwen3.5 视频抽帧与违规推理脚本

特点:
  1. 动态帧差法抽帧：画面变化不大时复用上一帧推理结果，极大节省推理时间
  2. 连续多帧滑窗判定：消除单帧运动模糊造成的误报
  3. 复用 chat_qwen35.py 的模型加载与推理逻辑
  4. 自动生成 JSON 违规报告并画框保存
"""

import argparse
import cv2
import json
import os
import shutil
import sys
import time
import tempfile
import datetime
from collections import deque
from pathlib import Path
import numpy as np

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
                parsed, json_str, raw, elapsed = infer_qwen35(
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
    parser = argparse.ArgumentParser(description="Qwen3.5 视频抽帧与违规推理")
    parser.add_argument("--video", type=str, required=True, help="视频文件路径")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="模型路径")
    parser.add_argument("--lora-path", type=str, default=None, help="LoRA路径")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_VIDEO_OUTPUT_DIR, help="输出目录")
    
    # 视频处理相关参数
    parser.add_argument("--fps", type=float, default=1.0, help="抽帧目标帧率 (默认 1 FPS)")
    parser.add_argument("--diff-threshold", type=float, default=5.0, help="触发推理的帧差均值阈值 (默认 5.0)")
    parser.add_argument("--window-size", type=int, default=3, help="滑窗大小 (默认 3 帧)")
    
    parser.add_argument("--visualize", action="store_true", help="是否绘制违规检测框并保存")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="推理提示词")
    parser.add_argument("--system-prompt", type=str, default=SYSTEM_PROMPT, help="系统提示词")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成长度")
    parser.add_argument("--device-map", type=str, default="auto", help="transformers device_map")
    parser.add_argument("--load-in-4bit", action="store_true", help="显式启用 4bit 量化")
    
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model, processor = load_qwen35(
        model_path=args.model_path,
        lora_path=args.lora_path,
        load_in_4bit=args.load_in_4bit,
        device_map=args.device_map,
    )

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


if __name__ == "__main__":
    main()
