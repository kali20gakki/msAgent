# Retry Middleware User Guide (Deep Agents/LangChain)

msAgent currently uses only official middleware:

- `init_chat_model(max_retries, timeout)`
- `ToolRetryMiddleware(max_retries, tools, on_failure, backoff_factor, initial_delay, max_delay, jitter)`

## 1. Corresponding YAML Configuration

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

## 2. Common Configuration Recommendations

### 2.1 Unstable Network

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

### 2.2 Keeping Only LLM Retries

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

### 2.3 Completely Disabling Retry Override

```yaml
retry:
  enabled: false
```

## 3. Notes

- When `retry.model.timeout` is `null`, msAgent uses the timeout configured by the LLM itself.
- When `retry.enabled=false`, msAgent does not mount `ToolRetryMiddleware` and sets the model retry count to 0.
- When `tool.tools` is `null`, the setting applies to all tools.
- `tool.retry_on` uses exception class name strings. Currently, it supports parsing built-in exception names and the `module.ClassName` form.
