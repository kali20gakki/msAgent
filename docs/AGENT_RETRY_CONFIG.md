# Agent Retry 配置说明

本文说明 agent YAML 中 `retry` 配置的当前行为。

## 配置结构

```yaml
version: "2.3.0"
name: my_agent
llm: default

retry:
  enabled: true
  llm_max_retries: 3
  llm_base_delay: 1.0
  llm_max_delay: 60.0
  enable_circuit_breaker: false
  circuit_breaker_threshold: 5
  circuit_breaker_recovery: 60.0
```

## 字段说明

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enabled` | `bool` | `true` | 是否启用重试机制 |
| `llm_max_retries` | `int` | `3` | LLM 请求最大重试次数 |
| `llm_base_delay` | `float` | `1.0` | LLM 重试初始退避时间，单位秒 |
| `llm_max_delay` | `float` | `60.0` | LLM 重试最大退避时间，单位秒 |
| `enable_circuit_breaker` | `bool` | `false` | 是否启用熔断器 |
| `circuit_breaker_threshold` | `int` | `5` | 连续失败多少次后打开熔断器 |
| `circuit_breaker_recovery` | `float` | `60.0` | 熔断恢复等待时间，单位秒 |

## 重要变更

`retry` 中不再支持 `tool_timeout`。

原因：

- `msagent` 不再接管 tool 调用层的重试和超时，避免和 MCP 原生机制叠加。
- MCP 已经提供 `invoke_timeout` 和 `repair_timeout`，应作为 tool 超时的唯一来源。

## Tool 超时应该配在哪里

请在 MCP 配置中设置：

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

## 示例

### 默认推荐

```yaml
retry:
  enabled: true
```

### 网络不稳定时增加重试

```yaml
retry:
  enabled: true
  llm_max_retries: 5
  llm_base_delay: 0.5
  llm_max_delay: 30.0
```

### 生产环境

```yaml
retry:
  enabled: true
  llm_max_retries: 3
  llm_base_delay: 1.0
  llm_max_delay: 60.0
  enable_circuit_breaker: true
  circuit_breaker_threshold: 5
  circuit_breaker_recovery: 60.0
```

## 迁移说明

如果旧配置里仍然保留了 `tool_timeout`：

- 新代码不会再使用它
- 建议直接从 agent/subagent YAML 中删除
- tool 超时请迁移到 MCP 的 `invoke_timeout`
