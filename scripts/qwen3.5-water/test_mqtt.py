"""
MQTT 链路测试脚本

模拟大疆无人机平台：
1. 先发送一条带有 GPS 坐标的 OSD 消息
2. 等待 2 秒
3. 发送一张测试图片到检测请求 Topic

用法：
    python test_mqtt.py --mode water  # 测试环水保
    python test_mqtt.py --mode edge   # 测试临边防护
"""

import argparse
import json
import time
import base64
import paho.mqtt.client as mqtt

BROKER = "218.3.126.49"
PORT = 1883
TEST_DRONE_SN = "TEST_DRONE_123"

def on_connect(client, userdata, flags, rc):
    print(f"已连接到 MQTT Broker, code={rc}")
    
    mode = userdata["mode"]
    
    # 根据模式选择 Topic 和测试图片
    if mode == "water":
        req_topic = "water/detect/request"
        res_topic = "water/detect/result"
        test_img_path = "/home/fs-ai/llama-qwen/data/环水保_labeled/images/Train/DJI_20260511091246_0773_D_r01_c00.jpg"
    elif mode == "edge":
        req_topic = f"jiahe/frame/{TEST_DRONE_SN}"
        res_topic = f"jiahe/alertsNonRealtime/{TEST_DRONE_SN}"
        test_img_path = "/home/fs-ai/llama-qwen/test/img_00100_tile_23.jpg"
        # 合规图片测试：/home/fs-ai/llama-qwen/test/img_00100_tile_23.jpg
        # 违规图片测试：/home/fs-ai/llama-qwen/test/img_00110_tile_11.jpg
    else:
        print(f"未知模式: {mode}")
        return

    # 订阅返回结果
    client.subscribe(res_topic)
    print(f"已订阅结果 Topic: {res_topic}")

    # ==========================================
    # 步骤 1: 模拟发送 OSD 信息
    # ==========================================
    osd_topic = f"thing/product/{TEST_DRONE_SN}/osd"
    osd_payload = {
        "biz_code": "device_osd",
        "data": {
            "sn": TEST_DRONE_SN,
            "host": {
                "latitude": 39.9042,
                "longitude": 116.4074,
                "height": 120.5
            }
        }
    }
    client.publish(osd_topic, json.dumps(osd_payload))
    print(f"\n[1] 已发送模拟 OSD 数据 -> {osd_topic}")
    
    # 等待 OSD 被处理
    time.sleep(2)
    
    # ==========================================
    # 步骤 2: 模拟发送图片检测请求
    # ==========================================
    try:
        with open(test_img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"读取图片失败: {test_img_path}, 错误: {e}")
        client.disconnect()
        return
        
    req_payload = {
        "stream_id": TEST_DRONE_SN,
        "time": "2026-06-16T12:00:00",
        "img": img_b64
    }
    client.publish(req_topic, json.dumps(req_payload))
    print(f"[2] 已发送测试图片请求 -> {req_topic}")
    print("等待模型返回结果 (如果检测未违规，则不会收到回传)...\n")

def on_message(client, userdata, msg):
    print(f"\n>>> 收到模型回传结果 <<<")
    print(f"Topic: {msg.topic}")
    
    data = json.loads(msg.payload.decode("utf-8"))
    
    # 验证位置信息是否正确附加
    location = data.get("location", {})
    if location.get("valid"):
        gps = location.get("gps", {})
        print(f"[OK] 成功获取缓存的 GPS: 纬度={gps.get('latitude')}, 经度={gps.get('longitude')}, 高度={gps.get('height')}")
    else:
        print("[FAIL] 警告：没有获取到 GPS 信息！")
        
    mode = userdata["mode"]
    if mode == "water":
        print(f"水体状态: {data.get('water_status')}")
        print(f"检测描述: {data.get('description')}")
        print(f"是否返回原图: {'是' if data.get('img') else '否'}")
    elif mode == "edge":
        print(f"违规类型: {data.get('violation_type', data.get('detections', [{}])[0].get('class_name', ''))}")
        print(f"严重程度: {data.get('severity', '')}")
        print(f"整改建议: {data.get('desc')}")
        print(f"检测框数量: {len(data.get('detections', []))}")
        print(f"是否返回画框图: {'是' if data.get('img') else '否'}")
    
    # 测试结束，断开连接
    client.disconnect()

def main():
    parser = argparse.ArgumentParser(description="MQTT 链路测试脚本")
    parser.add_argument("--mode", type=str, choices=["water", "edge"], default="water", help="测试模式：water(环水保) 或 edge(临边防护)")
    args = parser.parse_args()

    client = mqtt.Client(client_id=f"test_client_001_{args.mode}", userdata={"mode": args.mode})
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"准备连接 {BROKER}:{PORT} ... (模式: {args.mode})")
    client.connect(BROKER, PORT, 60)
    client.loop_forever()

if __name__ == "__main__":
    main()