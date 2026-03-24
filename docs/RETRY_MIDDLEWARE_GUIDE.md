# LLM 与 Tool Call 重试机制使用指南

本文档介绍如何在 msagent 中使用 LangChain Middleware 实现 LLM 请求和 Tool Call 的超时重试机制。

## 目录

- [概述](#概述)
- [快速开始](#快速开始)
- [配置选项](#配置选项)
- [使用示例](#使用示例)
- [高级功能](#高级功能)
- [故障排查](#故障排查)

---

## 概述

msagent 现在内置了基于 LangChain Middleware 的重试机制，可以自动处理：

1. **LLM 请求重试**: 当 LLM API 调用失败时（超时、连接错误等）自动重试
2. **Tool Call 重试**: 当工具调用失败时自动重试，支持自定义超时
3. **熔断器模式**: 防止级联故障，在连续失败后暂时禁用重试

---

## 快速开始

### 1. 基本使用（默认配置）

`create_react_agent` 现在默认启用重试功能：

```python
from msagent.agents.react_agent import create_react_agent

# 默认启用重试：LLM 最多重试 3 次，Tool 不重试
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=prompt,
)
```

### 2. 禁用重试

```python
# 显式设置为 None 可禁用重试
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=prompt,
    retry_middleware=None,  # 禁用重试
)
```

### 3. 自定义配置

```python
from msagent.middlewares import RetryMiddleware, RetryConfig, ToolRetryConfig

# 自定义 LLM 重试配置
llm_retry_config = RetryConfig(
    max_retries=5,           # 最多重试 5 次
    base_delay=2.0,          # 首次延迟 2 秒
    max_delay=60.0,          # 最大延迟 60 秒
    exponential_base=2.0,    # 指数退避基数
    jitter=True,             # 添加随机抖动
)

# 自定义 Tool 重试配置
tool_retry_config = ToolRetryConfig(
    max_retries=3,           # 工具调用最多重试 3 次
    timeout=30.0,            # 每次工具调用 30 秒超时
    exclude_tools=["delete_file"],  # 不_retry 这些工具
)

# 创建重试中间件
retry_middleware = RetryMiddleware(
    llm_config=llm_retry_config,
    tool_config=tool_retry_config,
    enable_llm_retry=True,    # 启用 LLM 重试
    enable_tool_retry=True,   # 启用 Tool 重试
)

# 使用自定义重试配置创建 agent
agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=prompt,
    retry_middleware=retry_middleware,
)
```

---

## 配置选项

### RetryConfig (LLM 重试配置)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_retries` | int | 3 | 最大重试次数 |
| `base_delay` | float | 1.0 | 初始延迟（秒） |
| `max_delay` | float | 60.0 | 最大延迟（秒） |
| `exponential_base` | float | 2.0 | 指数退避基数 |
| `jitter` | bool | True | 是否添加随机抖动 |
| `retryable_exceptions` | tuple | (TimeoutError, ConnectionError, asyncio.TimeoutError) | 可重试的异常类型 |
| `on_retry` | Callable | None | 重试时的回调函数 |

### ToolRetryConfig (Tool 重试配置)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_retries` | int | 2 | 最大重试次数 |
| `timeout` | float | None | 每次调用超时时间（秒） |
| `retryable_tools` | list[str] | [] | 允许重试的工具列表（空=全部） |
| `exclude_tools` | list[str] | [] | 禁止重试的工具列表 |

---

## 使用示例

### 示例 1: 添加重试监控回调

```python
def on_retry(attempt: int, exception: Exception, delay: float) -> None:
    """每次重试时调用"""
    print(f"⚠️  第 {attempt} 次重试，{delay:.1f}秒后执行...")
    print(f"   错误: {exception}")
    
    # 可以在这里发送监控指标
    # metrics.increment("llm_retry", tags={"attempt": attempt})

retry_config = RetryConfig(
    max_retries=3,
    on_retry=on_retry,
)

retry_middleware = RetryMiddleware(llm_config=retry_config)
```

### 示例 2: 仅对特定工具启用重试

```python
tool_retry_config = ToolRetryConfig(
    max_retries=3,
    timeout=30.0,
    retryable_tools=["web_search", "file_read"],  # 只重试这些工具
)

retry_middleware = RetryMiddleware(
    enable_llm_retry=True,
    enable_tool_retry=True,
    tool_config=tool_retry_config,
)
```

### 示例 3: 熔断器模式

当系统出现大量连续失败时，熔断器会暂时阻止重试，防止级联故障：

```python
from msagent.middlewares import CircuitBreakerRetryMiddleware

cb_middleware = CircuitBreakerRetryMiddleware(
    llm_config=RetryConfig(max_retries=3),
    failure_threshold=5,      # 连续 5 次失败后开启熔断
    recovery_timeout=60.0,    # 60 秒后尝试恢复
)

agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=prompt,
    retry_middleware=cb_middleware,
)
```

熔断器状态：
- **关闭**: 正常重试
- **开启**: 连续失败超过阈值，直接抛出异常
- **半开**: 恢复超时后，允许一次测试请求

### 示例 4: 工厂函数快速配置

```python
from msagent.middlewares import create_retry_middleware

# 使用工厂函数快速创建
retry_middleware = create_retry_middleware(
    max_llm_retries=3,
    max_tool_retries=2,
    llm_timeout=None,         # LLM 使用 provider 默认超时
    tool_timeout=30.0,        # Tool 30 秒超时
    enable_circuit_breaker=True,
)
```

---

## 高级功能

### 自定义可重试异常

```python
import httpx

retry_config = RetryConfig(
    max_retries=5,
    retryable_exceptions=(
        TimeoutError,
        ConnectionError,
        asyncio.TimeoutError,
        httpx.TimeoutException,      # HTTP 客户端超时
        httpx.ConnectError,          # 连接错误
        httpx.NetworkError,          # 网络错误
    ),
)
```

### 区分可重试与不可重试错误

```python
from msagent.middlewares.retry import RetryableError, NonRetryableError

# 在自定义工具中抛出这些异常
async def my_tool():
    if temporary_error:
        raise RetryableError("临时错误，应该重试")
    if permanent_error:
        raise NonRetryableError("永久错误，不要重试")
```

### 指数退避算法

重试延迟计算公式：

```
delay = min(base_delay * (exponential_base ^ attempt), max_delay)

if jitter:
    delay *= random(0.5, 1.5)
```

示例（base_delay=1, exponential_base=2）：
- 第 1 次重试: 1s * 2^0 = 1s (+/- jitter)
- 第 2 次重试: 1s * 2^1 = 2s (+/- jitter)
- 第 3 次重试: 1s * 2^2 = 4s (+/- jitter)

---

## 故障排查

### 日志查看

重试中间件会输出以下日志：

```
INFO: LLM request attempt 1 failed: TimeoutError. Retrying in 2.00s...
INFO: LLM request attempt 2 failed: ConnectionError. Retrying in 4.50s...
WARNING: LLM request failed after 3 attempts: APIError
```

### 常见问题

**Q: 重试没有生效？**

检查以下几点：
1. 确认 `retry_middleware` 已正确传递给 `create_react_agent`
2. 检查异常类型是否在 `retryable_exceptions` 列表中
3. 查看日志确认中间件已加载

**Q: 如何调试重试逻辑？**

```python
import logging
logging.getLogger("msagent.middlewares.retry").setLevel(logging.DEBUG)
```

**Q: 重试次数太多导致响应慢？**

调整配置减少重试：

```python
RetryConfig(
    max_retries=1,      # 只重试 1 次
    base_delay=0.5,     # 快速重试
)
```

**Q: 如何对某些错误不重试？**

```python
def on_retry(attempt, exception, delay):
    if "invalid_api_key" in str(exception).lower():
        raise NonRetryableError("API key 无效，停止重试")

RetryConfig(on_retry=on_retry)
```

---

## 架构说明

### Middleware 执行顺序

```
RetryMiddleware (awrap_model_call)
├── DynamicPromptMiddleware
├── TokenCostMiddleware
├── ApprovalMiddleware
├── SandboxMiddleware
└── ReturnDirectMiddleware
    └── 实际 LLM 调用 / Tool 调用
```

`RetryMiddleware` 作为最外层包装器，可以捕获所有内部层的异常。

### 代码位置

- 中间件实现: `src/msagent/middlewares/retry.py`
- 集成代码: `src/msagent/agents/react_agent.py`
- 导出: `src/msagent/middlewares/__init__.py`

---

## 迁移指南

### 从旧版本升级

如果你之前没有使用重试机制，升级后默认启用重试，无需修改代码。

如果希望保持原有行为（无重试）：

```python
agent = create_react_agent(
    ...,
    retry_middleware=None,
)
```

---

## 参考

- [LangChain Middleware 文档](https://python.langchain.com/docs/concepts/#middleware)
- [指数退避算法](https://aws.amazon.com/cn/blogs/architecture/exponential-backoff-and-jitter/)
- [熔断器模式](https://martinfowler.com/bliki/CircuitBreaker.html)
