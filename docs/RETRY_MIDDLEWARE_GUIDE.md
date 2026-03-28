# Retry Middleware Guide

`msagent` 的 retry middleware 负责两件事：

- LLM 请求失败后的重试
- 可选的 Tool 调用失败后重试

在 agent YAML 默认配置路径里，tool retry 已不再启用；tool 超时也不再由它控制。

## 基本用法

```python
from msagent.middlewares import RetryMiddleware, RetryConfig, ToolRetryConfig

llm_retry_config = RetryConfig(
    max_retries=5,
    base_delay=2.0,
    max_delay=60.0,
)

tool_retry_config = ToolRetryConfig(
    max_retries=3,
    exclude_tools=["delete_file"],
)

retry_middleware = RetryMiddleware(
    llm_config=llm_retry_config,
    tool_config=tool_retry_config,
    enable_llm_retry=True,
    enable_tool_retry=True,
)
```

## 配置项

### `RetryConfig`

| Field | Type | Default |
| --- | --- | --- |
| `max_retries` | `int` | `3` |
| `base_delay` | `float` | `1.0` |
| `max_delay` | `float` | `60.0` |
| `exponential_base` | `float` | `2.0` |
| `jitter` | `bool` | `True` |
| `retryable_exceptions` | `tuple[type[Exception], ...]` | timeout / connection related exceptions |
| `on_retry` | `Callable | None` | `None` |

### `ToolRetryConfig`

| Field | Type | Default |
| --- | --- | --- |
| `max_retries` | `int` | `2` |
| `retryable_tools` | `list[str]` | `[]` |
| `exclude_tools` | `list[str]` | `[]` |

## 工厂函数

```python
from msagent.middlewares import create_retry_middleware

retry_middleware = create_retry_middleware(
    max_llm_retries=3,
    max_tool_retries=2,
    llm_timeout=None,
    enable_circuit_breaker=True,
)
```

## Tool 超时配置

如果你需要限制 MCP tool 的执行时间，请在 MCP 配置里设置：

```json
{
  "mcpServers": {
    "server-name": {
      "invoke_timeout": 60.0,
      "repair_timeout": 30
    }
  }
}
```

不要再在 `ToolRetryConfig` 或 agent `retry` 配置里设置 `tool_timeout`。
