# Agent YAML 配置中的 Retry 设置

本文档介绍如何在 agent YAML 配置文件中设置 retry 参数。

---

## 配置结构

在 agent YAML 配置文件中添加 `retry` 部分：

```yaml
version: "2.3.0"
name: my_agent
description: Agent with custom retry settings
prompt:
- prompts/agents/my_agent.md
llm: default
default: true

# Retry configuration
retry:
  enabled: true                    # 是否启用重试 (默认: true)
  llm_max_retries: 3               # LLM 请求最大重试次数 (默认: 3)
  llm_base_delay: 1.0              # LLM 重试初始延迟，秒 (默认: 1.0)
  llm_max_delay: 60.0              # LLM 重试最大延迟，秒 (默认: 60.0)
  tool_max_retries: 2              # Tool 调用最大重试次数 (默认: 2)
  tool_timeout: 30.0               # Tool 调用超时时间，秒 (默认: null)
  enable_circuit_breaker: false    # 是否启用熔断器 (默认: false)
  circuit_breaker_threshold: 5     # 熔断器触发阈值 (默认: 5)
  circuit_breaker_recovery: 60.0   # 熔断器恢复时间，秒 (默认: 60.0)
```

---

## 配置示例

### 示例 1: 默认配置（推荐）

```yaml
version: "2.3.0"
name: msagent
description: Default agent with standard retry
prompt:
- prompts/agents/msagent.md
llm: default
default: true

# 使用默认重试配置
retry:
  enabled: true
```

### 示例 2: 高频重试（不稳定网络环境）

```yaml
version: "2.3.0"
name: unstable_network_agent
description: Agent with aggressive retry for unstable networks
prompt:
- prompts/agents/msagent.md
llm: default

retry:
  enabled: true
  llm_max_retries: 5               # 更多重试
  llm_base_delay: 0.5              # 更快重试
  llm_max_delay: 30.0
  tool_max_retries: 3
  tool_timeout: 45.0               # Tool 更长超时
```

### 示例 3: 保守重试（生产环境）

```yaml
version: "2.3.0"
name: production_agent
description: Agent with conservative retry settings
prompt:
- prompts/agents/msagent.md
llm: default

retry:
  enabled: true
  llm_max_retries: 2               # 较少重试
  llm_base_delay: 2.0              # 较慢重试
  tool_max_retries: 1
  tool_timeout: 20.0               # Tool 更短超时
  enable_circuit_breaker: true     # 启用熔断保护
  circuit_breaker_threshold: 3     # 快速熔断
  circuit_breaker_recovery: 120.0  # 较长恢复时间
```

### 示例 4: 禁用重试

```yaml
version: "2.3.0"
name: no_retry_agent
description: Agent without retry
prompt:
- prompts/agents/msagent.md
llm: default

retry:
  enabled: false
```

### 示例 5: SubAgent 重试配置

```yaml
version: "2.3.0"
name: research_subagent
description: SubAgent with custom retry
prompt:
- prompts/agents/research.md
llm: default

# SubAgent 可以有独立的 retry 配置
retry:
  enabled: true
  llm_max_retries: 3
  tool_max_retries: 2
  tool_timeout: 60.0               # 研究任务可能需要更长时间
```

---

## 配置参数详解

### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 总开关，设为 `false` 禁用所有重试 |
| `llm_max_retries` | int | `3` | LLM API 调用失败后的最大重试次数 |
| `tool_max_retries` | int | `2` | Tool 调用失败后的最大重试次数 |

### 延迟参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `llm_base_delay` | float | `1.0` | 首次重试等待时间（秒） |
| `llm_max_delay` | float | `60.0` | 最大重试等待时间（秒） |

延迟计算公式（指数退避）：
```
delay = min(base_delay * 2^attempt, max_delay)
```

示例：
- 第 1 次重试：1.0 秒
- 第 2 次重试：2.0 秒
- 第 3 次重试：4.0 秒
- 第 4 次重试：8.0 秒（未超过 max_delay）

### Tool 超时参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tool_timeout` | float | `null` | 每次 tool 调用超时时间（秒） |

- 设为 `null` 或省略：不设置超时
- 建议值：30-60 秒

### 熔断器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_circuit_breaker` | bool | `false` | 是否启用熔断器模式 |
| `circuit_breaker_threshold` | int | `5` | 连续失败多少次后开启熔断 |
| `circuit_breaker_recovery` | float | `60.0` | 熔断后等待多少秒尝试恢复 |

熔断器状态流转：
```
┌─────────┐    失败 < threshold    ┌─────────┐
│  Closed │ ◄───────────────────── │  Half   │
│ (正常)  │                        │  Open   │
└────┬────┘                        │(测试中) │
     │ 失败 >= threshold           └────┬────┘
     ▼                                  │
┌─────────┐    等待 recovery 时间     │
│  Open   │ ─────────────────────────► │
│ (熔断)  │                            │
└─────────┘◄───────────────────────────┘
           测试成功
```

---

## 版本迁移

### 从 2.2.x 迁移到 2.3.0

新版本自动添加默认 retry 配置，无需手动修改现有配置文件。系统会自动迁移：

```python
# 旧版本配置 (2.2.1)
version: "2.2.1"
name: my_agent

# 迁移后自动添加 retry 配置：
retry:
  enabled: true
  llm_max_retries: 3
  llm_base_delay: 1.0
  llm_max_delay: 60.0
  tool_max_retries: 2
  tool_timeout: null
  enable_circuit_breaker: false
  circuit_breaker_threshold: 5
  circuit_breaker_recovery: 60.0
```

如需自定义，手动添加 retry 部分即可。

---

## 配置覆盖规则

### SubAgent 继承

SubAgent 可以使用独立的 retry 配置：

```yaml
# Main agent
version: "2.3.0"
name: main_agent
retry:
  llm_max_retries: 3

# SubAgent (agents/subagents/my_sub.yml)
version: "2.3.0"
name: my_sub
retry:
  llm_max_retries: 5  # SubAgent 使用自己的配置
```

### 环境变量覆盖（未来支持）

计划支持通过环境变量覆盖配置：

```bash
export MSAGENT_RETRY_LLM_MAX_RETRIES=5
export MSAGENT_RETRY_ENABLED=false
```

---

## 调试与监控

### 启用调试日志

```python
import logging
logging.getLogger("msagent.middlewares.retry").setLevel(logging.DEBUG)
```

### 查看重试统计

日志输出示例：
```
INFO: LLM request attempt 1 failed: TimeoutError. Retrying in 2.00s...
INFO: LLM request attempt 2 failed: ConnectionError. Retrying in 4.50s...
WARNING: LLM request failed after 3 attempts: APIError
```

### 监控指标（自定义回调）

```python
from msagent.middlewares import RetryConfig, RetryMiddleware

def on_retry(attempt, exception, delay):
    # 发送到监控系统
    metrics.increment("agent.retry", tags={
        "agent": "my_agent",
        "attempt": attempt,
        "error": type(exception).__name__
    })

retry_config = RetryConfig(
    max_retries=3,
    on_retry=on_retry,
)

middleware = RetryMiddleware(llm_config=retry_config)
```

---

## 最佳实践

### 1. 开发环境

```yaml
retry:
  enabled: true
  llm_max_retries: 1               # 快速失败，便于发现问题
  tool_max_retries: 0              # Tool 不重试
  enable_circuit_breaker: false    # 禁用熔断
```

### 2. 测试环境

```yaml
retry:
  enabled: true
  llm_max_retries: 2
  llm_base_delay: 0.5              # 快速重试
  tool_max_retries: 1
  enable_circuit_breaker: true     # 测试熔断逻辑
  circuit_breaker_threshold: 10
```

### 3. 生产环境

```yaml
retry:
  enabled: true
  llm_max_retries: 3
  llm_base_delay: 1.0
  llm_max_delay: 60.0
  tool_max_retries: 2
  tool_timeout: 30.0
  enable_circuit_breaker: true     # 必须启用熔断
  circuit_breaker_threshold: 5
  circuit_breaker_recovery: 60.0
```

### 4. 长时间任务 Agent

```yaml
retry:
  enabled: true
  llm_max_retries: 5               # 更多重试机会
  tool_max_retries: 3
  tool_timeout: 120.0              # 2 分钟超时
  llm_max_delay: 120.0             # 更长最大延迟
```

---

## 故障排查

### 重试不生效

1. 检查 `enabled` 是否为 `true`
2. 检查版本是否为 `2.3.0` 或更高
3. 查看日志确认中间件已加载

### 重试太频繁

```yaml
retry:
  llm_max_retries: 1               # 减少重试次数
  llm_base_delay: 5.0              # 增加延迟
```

### 熔断器频繁触发

```yaml
retry:
  circuit_breaker_threshold: 10    # 增加阈值
  circuit_breaker_recovery: 30.0   # 减少恢复时间
```

### Tool 超时

```yaml
retry:
  tool_timeout: 60.0               # 增加超时
  tool_max_retries: 1              # 减少重试
```

---

## 参考

- [Retry Middleware 指南](./RETRY_MIDDLEWARE_GUIDE.md)
- [配置版本迁移](./CONFIG_MIGRATION.md)
