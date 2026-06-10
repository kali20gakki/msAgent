# 评测错误处理参考

## 常见错误与处理

### 1. 推理服务启动失败

**错误信息**：
```
Failed to start inference engine: port 8000 already in use
```

**原因**：端口被占用

**解决**：
- 更换 `evaluation.port` 和 `inference_engine.port`
- 或等待占用端口的进程结束

---

### 2. HCCL 初始化失败

**错误信息**：
```
HCCL init failed: device 0,1 not available
```

**原因**：
- NPU 设备不可用
- `device_indices` 与物理设备不匹配

**解决**：
- 检查 `npu-smi info` 确认设备状态
- 确认 `device_indices` 在可用范围内

---

### 3. 模型未找到

**错误信息**：
```
Model not found at /path/to/quantized_model
```

**原因**：量化产物路径错误

**解决**：
- 确认 `quantize` 步骤成功完成
- 检查 `save_path` 与 Evaluation YAML 中的路径对齐

---

### 4. 评测超时

**错误信息**：
```
Evaluation timeout after 10800s
```

**原因**：
- 数据集太大
- 模型推理速度慢
- `aisbench.timeout` 设置过短

**解决**：
- 增加 `aisbench.timeout`
- 减少 `max_out_len`
- 增加 `batch_size`（如显存允许）

---

### 5. 数据集配置错误

**错误信息**：
```
Dataset config not found: xxx_gen_xxx
```

**原因**：`config_name` 不是有效的 AISBench 注册名

**解决**：
- 检查 [数据集映射表](dataset_mapping.md)
- 使用正确的 `config_name`

---

### 6. 健康检查失败

**错误信息**：
```
Health check failed: /v1/models returned 503
```

**原因**：
- vLLM 服务启动但未就绪
- 模型加载失败

**解决**：
- 增加 `startup_timeout`
- 检查模型路径和格式

---

### 7. 其他未知错误

**解决**：
- 寻求上层Agent解决

---

## 错误上报格式

发生错误时，向 orchestrator 返回：

```json
{
  "ok": false,
  "error": "简短错误描述",
  "error_code": "ERROR_TYPE",
  "partial_results": {},
  "suggestion": "建议的解决方式"
}
```

| error_code | 含义 | 建议 |
|------------|------|------|
| `INFERENCE_START_ERROR` | 推理服务启动失败 | 检查端口和设备 |
| `HCCL_ERROR` | NPU 通信错误 | 检查设备状态 |
| `MODEL_NOT_FOUND` | 模型未找到 | 检查量化产物路径 |
| `TIMEOUT` | 评测超时 | 增加超时时间 |
| `DATASET_ERROR` | 数据集错误 | 检查 config_name |
| `HEALTH_CHECK_ERROR` | 健康检查失败 | 检查服务状态 |
| `UNKNOWN_ERROR` | 未知错误 | 寻求上层Agent解决 |
