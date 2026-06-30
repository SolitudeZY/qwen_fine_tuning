# 环水保 MQTT 测试接口文档

## 1. 适用范围

本文档用于前端或联调同事测试“环水保（水体污染检测）”MQTT 接口。

当前服务端脚本：

- `/home/fs-ai/llama-qwen/scripts/qwen3.5-water/04_mqtt_inference.py`

当前测试脚本参考：

- `/home/fs-ai/llama-qwen/scripts/qwen3.5-water/test_mqtt.py`

## 2. 接口概览

服务端通过 MQTT 接收无人机图片请求，并输出水体检测结果。

当前有 3 类 Topic：

- 请求 Topic：`water/detect/request`
- 结果 Topic：`water/detect/result`
- OSD Topic：`thing/product/{stream_id}/osd`

其中：

- `request` 用于发送待检测图片
- `result` 用于接收检测结果
- `osd` 用于提前上报无人机 GPS 信息，服务端会做缓存，并附加到检测结果里

## 3. 关键业务规则

### 3.1 仅报警回传

当前环水保任务遵循“仅报警”规则：

- 当模型判断 `water_status = "polluted"` 时，服务端会发布结果到 `water/detect/result`
- 当模型判断 `water_status = "normal"` 时，服务端不会回传结果消息

因此前端测试时需要注意：

- 发送正常水体图片后，收不到结果是符合预期的，不代表服务异常

### 3.2 图片字段

请求中图片字段必须使用：

- `img`

兼容字段：

- `image_base64`

推荐统一发送 `img`，内容为图片的 Base64 字符串。

### 3.3 GPS 位置信息来源

服务端优先使用 OSD Topic 中缓存的位置信息。

匹配规则：

- 请求消息中的 `stream_id`
- 必须和 OSD 消息中的 `data.sn` 一致

如果没有匹配到 OSD 缓存：

- 返回中会带 `location.valid=false`

### 3.4 返回原图字段

当前结果中的 `img` 字段返回原图 Base64。

规则：

- `polluted` 时返回原图 Base64
- `normal` 时不会回传消息，因此也不存在返回图

## 4. Broker 信息

当前默认配置如下：

- Broker：`218.3.126.49`
- Port：`1883`

如果后续环境切换，以服务端实际启动参数为准：

- `--mqtt-broker`
- `--mqtt-port`
- `--topic-req`
- `--topic-res`

## 5. Topic 定义

### 5.1 请求 Topic

`water/detect/request`

用途：

- 前端向服务端发送检测请求

### 5.2 结果 Topic

`water/detect/result`

用途：

- 服务端向前端回传污染检测结果

### 5.3 OSD Topic

`thing/product/{stream_id}/osd`

用途：

- 上报无人机位置信息

示例：

- `thing/product/TEST_DRONE_123/osd`

## 6. 请求报文

### 6.1 请求字段说明

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `stream_id` | string | 是 | 设备或视频流唯一标识，需与 OSD 中的 `sn` 对应 |
| `time` | string | 否 | 请求时间，建议使用 ISO8601 格式 |
| `img` | string | 是 | 图片 Base64 |
| `location` | object | 否 | 可选透传位置；若已发送 OSD，服务端优先使用 OSD 缓存 |

### 6.2 请求示例

```json
{
  "stream_id": "TEST_DRONE_123",
  "time": "2026-06-30T15:30:00",
  "img": "/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

### 6.3 带 location 的请求示例

```json
{
  "stream_id": "TEST_DRONE_123",
  "time": "2026-06-30T15:30:00",
  "img": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "location": {
    "valid": true,
    "gps": {
      "height": 120.5,
      "latitude": 39.9042,
      "longitude": 116.4074
    }
  }
}
```

## 7. OSD 报文

### 7.1 OSD 字段说明

OSD 用于上报设备最新 GPS 信息，服务端仅缓存，不会直接返回响应。

### 7.2 OSD 示例

```json
{
  "biz_code": "device_osd",
  "data": {
    "sn": "TEST_DRONE_123",
    "host": {
      "latitude": 39.9042,
      "longitude": 116.4074,
      "height": 120.5
    }
  }
}
```

## 8. 结果报文

### 8.1 仅污染时返回

只有当模型判断为污染水体时，才会收到结果消息。

### 8.2 结果字段说明

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `success` | boolean | 是 | 是否处理成功 |
| `processing_time_sec` | number | 是 | 推理耗时，单位秒 |
| `water_status` | string | 是 | `polluted` 或 `normal`，但当前仅 `polluted` 才会回传 |
| `description` | string | 是 | 对污染原因的一句话描述 |
| `stream_id` | string | 是 | 原样透传 |
| `time` | string | 否 | 原样透传 |
| `location` | object | 是 | GPS 信息，优先使用 OSD 缓存 |
| `img` | string | 是 | 原图 Base64，污染时返回 |
| `raw_output` | string | 是 | 模型原始输出，便于排查 |

### 8.3 结果示例

```json
{
  "success": true,
  "processing_time_sec": 3.241,
  "water_status": "polluted",
  "description": "水面存在大面积漂浮物和明显异常颜色，疑似污染。",
  "stream_id": "TEST_DRONE_123",
  "time": "2026-06-30T15:30:00",
  "location": {
    "valid": true,
    "gps": {
      "height": 120.5,
      "latitude": 39.9042,
      "longitude": 116.4074
    }
  },
  "img": "/9j/4AAQSkZJRgABAQAAAQABAAD...",
  "raw_output": "{\"water_status\":\"polluted\",\"description\":\"水面存在大面积漂浮物和明显异常颜色，疑似污染。\"}"
}
```

## 9. 异常报文

当消息处理失败时，服务端会向结果 Topic 回传错误消息。

### 9.1 异常字段说明

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `success` | boolean | 是 | 固定为 `false` |
| `error` | string | 是 | 错误原因 |

### 9.2 异常示例

```json
{
  "success": false,
  "error": "缺少图片字段 (img/image_base64)"
}
```

## 10. 前端测试建议

### 10.1 推荐测试流程

1. 先订阅 `water/detect/result`
2. 先发送一条 OSD 消息到 `thing/product/{stream_id}/osd`
3. 等待 1 到 2 秒
4. 再发送图片请求到 `water/detect/request`
5. 根据图片是否污染，判断是否应收到结果

### 10.2 判定标准

- 发送污染图：应收到 `success=true` 的结果消息
- 发送正常图：通常不会收到结果消息
- 缺字段或错误数据：应收到 `success=false` 的错误消息

## 11. Python 测试脚本

现有链路测试脚本：

- `/home/fs-ai/llama-qwen/scripts/qwen3.5-water/test_mqtt.py`

执行方式：

```bash
python /home/fs-ai/llama-qwen/scripts/qwen3.5-water/test_mqtt.py --mode water
```

## 12. 注意事项

- 请求与 OSD 的 `stream_id/sn` 必须一致，否则无法命中缓存位置
- `img` 必须是合法 Base64，不能直接传图片路径
- 当前结果 Topic 是固定单 Topic：`water/detect/result`
- 当前 `normal` 不回传结果，这一点前端联调时必须特别注意
