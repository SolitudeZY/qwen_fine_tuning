# 环水保 MQTT 测试用例

## 1. 测试目标

验证环水保 MQTT 服务在以下场景下是否符合预期：

- 正常收发
- OSD 位置信息缓存
- 污染图回传
- 正常图静默
- 异常请求报错

服务脚本：

- `/home/fs-ai/llama-qwen/scripts/qwen3.5-water/04_mqtt_inference.py`

测试脚本：

- `/home/fs-ai/llama-qwen/scripts/qwen3.5-water/test_mqtt.py`

## 2. 测试环境

- Broker：`218.3.126.49`
- Port：`1883`
- 请求 Topic：`water/detect/request`
- 结果 Topic：`water/detect/result`
- OSD Topic 模板：`thing/product/{stream_id}/osd`

## 3. 测试前准备

### 3.1 启动服务端

```bash
python /home/fs-ai/llama-qwen/scripts/qwen3.5-water/04_mqtt_inference.py \
  --model-path /home/fs-ai/llama-qwen/models/Qwen/Qwen3.5-2B \
  --lora-path /home/fs-ai/llama-qwen/outputs/qwen3.5-water/train/v1-20260616-085800/checkpoint-600 \
  --mqtt-serve
```

### 3.2 启动结果订阅

可使用 MQTTX、MQTT Explorer 或任意 MQTT 客户端订阅：

- `water/detect/result`

## 4. 测试用例

### 用例 1：污染图片正常回传

**目的**

- 验证污染图片会正常收到结果

**前置条件**

- 服务端已启动
- 已订阅 `water/detect/result`

**步骤**

1. 发送 OSD 消息到 `thing/product/TEST_DRONE_123/osd`
2. 等待 2 秒
3. 发送一张污染水体图片到 `water/detect/request`

**请求示例**

```json
{
  "stream_id": "TEST_DRONE_123",
  "time": "2026-06-30T15:30:00",
  "img": "污染图Base64"
}
```

**预期结果**

- 收到 `water/detect/result` 消息
- `success=true`
- `water_status="polluted"`
- `description` 非空
- `img` 非空
- `location.valid=true`

### 用例 2：正常图片不回传

**目的**

- 验证正常水体图片按“仅报警”规则不回传

**步骤**

1. 发送 OSD 消息
2. 等待 2 秒
3. 发送一张正常水体图片到 `water/detect/request`
4. 观察 `water/detect/result`

**预期结果**

- 不收到结果消息
- 这是符合预期的

### 用例 3：未发送 OSD 时仍可检测

**目的**

- 验证没有 OSD 时，检测功能仍可工作

**步骤**

1. 不发送 OSD
2. 直接发送污染图请求

**预期结果**

- 收到结果消息
- `success=true`
- `water_status="polluted"`
- `location.valid=false`

### 用例 4：请求自带 location 且无 OSD

**目的**

- 验证未命中 OSD 缓存时，请求自带 location 能透传

**步骤**

1. 不发送 OSD
2. 发送带 `location` 的污染图请求

**请求示例**

```json
{
  "stream_id": "TEST_DRONE_123",
  "time": "2026-06-30T15:30:00",
  "img": "污染图Base64",
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

**预期结果**

- 收到结果消息
- `location.valid=true`
- `location.gps` 与请求一致

### 用例 5：OSD 缓存覆盖请求自带 location

**目的**

- 验证服务端优先使用 OSD 缓存位置

**步骤**

1. 发送 OSD，坐标设为 A
2. 发送污染图请求，并在请求内附带另一组坐标 B

**预期结果**

- 收到结果消息
- 返回 `location` 为 OSD 中的坐标 A
- 不是请求中的坐标 B

### 用例 6：缺少 img 字段

**目的**

- 验证缺字段时会回错误消息

**步骤**

1. 发送如下错误请求

```json
{
  "stream_id": "TEST_DRONE_123",
  "time": "2026-06-30T15:30:00"
}
```

**预期结果**

- 收到结果消息
- `success=false`
- `error` 包含 `缺少图片字段`

### 用例 7：img 不是合法 Base64

**目的**

- 验证非法图片数据处理

**步骤**

1. 发送如下请求

```json
{
  "stream_id": "TEST_DRONE_123",
  "time": "2026-06-30T15:30:00",
  "img": "not_base64"
}
```

**预期结果**

- 收到结果消息
- `success=false`
- `error` 为 Base64 解析相关错误

### 用例 8：stream_id 与 OSD 的 sn 不一致

**目的**

- 验证无法命中 GPS 缓存的情况

**步骤**

1. 发送 OSD 到 `thing/product/TEST_DRONE_ABC/osd`
2. 请求里发送：

```json
{
  "stream_id": "TEST_DRONE_XYZ",
  "time": "2026-06-30T15:30:00",
  "img": "污染图Base64"
}
```

**预期结果**

- 仍可收到污染结果
- `location.valid=false` 或使用请求自带 `location`

### 用例 9：连续多次发送污染图

**目的**

- 验证服务可连续处理多条请求

**步骤**

1. 连续发送 5 张污染图

**预期结果**

- 每张污染图都收到对应结果
- 服务不崩溃，不断连

### 用例 10：连续多次发送正常图

**目的**

- 验证正常图静默处理

**步骤**

1. 连续发送 5 张正常图

**预期结果**

- 一般不收到任何结果消息
- 服务端日志显示“合规，忽略回传”

## 5. 建议测试数据

### 污染图

- 使用一张明显含有藻类、漂浮物、油污、异常颜色或严重浑浊的图片

### 正常图

- 使用一张水面较清澈、无明显污染特征的图片

### 官方测试脚本内置图片

可直接参考：

- `/home/fs-ai/llama-qwen/scripts/qwen3.5-water/test_mqtt.py`

其中 `water` 模式的测试图路径为：

- `/home/fs-ai/llama-qwen/data/环水保_labeled/images/Train/DJI_20260511091246_0773_D_r01_c00.jpg`

## 6. 联调判定建议

前端联调时建议重点确认以下 4 点：

- 请求格式是否正确
- OSD 和请求中的 `stream_id/sn` 是否匹配
- 正常图不回传是否已被前端视为“符合协议”
- 污染图结果中的 `img/location/description` 是否可直接用于页面展示

## 7. 备注

- 当前 `result` Topic 是统一单 Topic，不按 `stream_id` 分流
- 当前异常请求也会向 `water/detect/result` 回错误消息
- 当前 `raw_output` 是模型原始文本输出，建议前端仅用于调试，不作为正式展示字段
