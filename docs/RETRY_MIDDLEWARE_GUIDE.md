# Retry Middleware 使用指南（Deepagents/LangChain）

`msagent` 当前只使用官方中间件：

- `init_chat_model(max_retries, timeout)`
- `ToolRetryMiddleware(max_retries, tools, on_failure, backoff_factor, initial_delay, max_delay, jitter)`

## 对应的 YAML 配置

```yaml
retry:
  enabled: true
  model:
    enabled: true
    max_retries: 3
    timeout: 120.0
  tool:
    enabled: true
    max_retries: 2
    tools: null
    retry_on: null
    on_failure: continue
    backoff_factor: 2.0
    initial_delay: 1.0
    max_delay: 60.0
    jitter: true
```

## 常见配置建议

### 网络不稳定

```yaml
retry:
  enabled: true
  model:
    enabled: true
    max_retries: 8
    timeout: 180.0
  tool:
    enabled: true
    max_retries: 4
    retry_on:
      - TimeoutError
      - ConnectionError
    backoff_factor: 2.0
    initial_delay: 2.0
    max_delay: 60.0
    jitter: true
```

### 只保留 LLM 重试

```yaml
retry:
  enabled: true
  model:
    enabled: true
    max_retries: 5
    timeout: 120.0
  tool:
    enabled: false
    max_retries: 0
```

### 完全关闭重试覆盖

```yaml
retry:
  enabled: false
```

## 注意事项

- `retry.model.timeout` 为 `null` 时，使用 LLM 自身配置超时。
- `retry.enabled=false` 时，不挂载 `ToolRetryMiddleware`，且模型重试数置为 `0`。
- `tool.tools` 为 `null` 时，表示对所有工具生效。
- `tool.retry_on` 使用异常类名字符串；当前支持解析内置异常名与 `module.ClassName` 形式。
