# 量化错误处理参考

## 常见错误与处理

### 1. 路径格式错误

**错误信息**：
```
Unexpected token '/' at position ...
```

**原因**：路径未加引号，JSON 解析失败

**解决**：
```json
// ❌ 错误
{"config_path": /path/to/practice.yaml}

// ✅ 正确
{"config_path": "/path/to/practice.yaml"}
```

---

### 2. Practice YAML 校验失败

**错误信息**：
```
PracticeConfig validation failed: ...
```

**原因**：YAML 格式不符合规范

**解决**：
- 检查是否通过 `yaml_validation_validate`
- 确认 `metadata.label` 是 dict 而非字符串
- 确认 `apiversion: modelslim_v1`

---

### 3. 模型加载失败

**错误信息**：
```
Model load failed: ...
```

**原因**：
- `model_path` 不存在
- 缺少 `config.json` 或 `pytorch_model.bin`
- `trust_remote_code` 未设置

**解决**：
- 确认模型路径正确
- 确认模型文件完整
- 如需远程代码，设置 `trust_remote_code: true`

---

### 4. OOM（内存不足）

**错误信息**：
```
RuntimeError: CUDA out of memory
或
Ascend OOM
```

**原因**：模型太大，设备内存不足

**解决**：
- 减少 batch size（在 Practice YAML 中配置）
- 使用更小的校准数据集
- 换用更大显存设备

---

### 5. 校准数据缺失

**错误信息**：
```
Dataset not found: mix_calib.jsonl
```

**原因**：`spec.dataset` 指定的数据集不存在

**解决**：
- 确认 `lab_calib` 中有该数据集
- 使用正确的短名（非路径）

---

### 6. 其他未知错误

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
  "suggestion": "建议的解决方式"
}
```

| error_code | 含义 | 建议 |
|------------|------|------|
| `VALIDATION_ERROR` | YAML 校验失败 | 检查 Practice YAML 格式 |
| `MODEL_LOAD_ERROR` | 模型加载失败 | 检查模型路径和完整性 |
| `OOM_ERROR` | 内存不足 | 减小 batch size 或换设备 |
| `DATASET_ERROR` | 数据集错误 | 检查校准数据集 |
| `MCP_ERROR` | MCP 内部错误 | 重试或检查环境 |
| `UNKNOWN_ERROR` | 未知错误 | 寻求上层Agent解决 |
