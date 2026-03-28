# Agent Retry 实现说明

本文概述 agent `retry` 配置到运行时 middleware 的生效链路。

## 当前行为

- `RetryPolicyConfig` 负责解析 agent YAML 中的 `retry` 配置
- `deep_agent.py` 会把配置转换成 `RetryMiddleware` / `CircuitBreakerRetryMiddleware`
- agent 配置路径下不再启用 `msagent` 侧 tool retry
- tool 超时不再由 `msagent` 本地注入

## 为什么移除 `tool_timeout`

之前 `msagent` 会在 tool 调用外层再包一层 `asyncio.wait_for(...)`。
此前 agent 配置里还会控制 tool 重试次数。

这会和 MCP 自带的超时机制重复：

- MCP `invoke_timeout`
- MCP `repair_timeout`

现在统一改为：

- agent `retry` 只控制 LLM retry 和熔断相关能力
- tool 超时只由 MCP 配置控制
- tool 是否重试也不再由 agent YAML 配置

## 关键链路

1. `.msagent/agents/*.yml` 或 `.msagent/subagents/*.yml`
2. `src/msagent/configs/agent.py`
3. `src/msagent/agents/deep_agent.py`
4. `src/msagent/middlewares/retry.py`
5. `src/msagent/mcp/client.py`
6. `src/msagent/mcp/tool.py`

## 配置示例

```yaml
retry:
  enabled: true
  llm_max_retries: 3
  llm_base_delay: 1.0
  llm_max_delay: 60.0
  enable_circuit_breaker: false
  circuit_breaker_threshold: 5
  circuit_breaker_recovery: 60.0
```

MCP 超时示例：

```json
{
  "mcpServers": {
    "msprof-mcp": {
      "invoke_timeout": 60.0,
      "repair_timeout": 30
    }
  }
}
```

## 验证点

- `RetryMiddleware.awrap_tool_call()` 不再使用本地 `asyncio.wait_for`
- `ToolRetryConfig` 不再包含 `timeout`
- 默认 agent/subagent YAML 不再暴露 `tool_timeout`
