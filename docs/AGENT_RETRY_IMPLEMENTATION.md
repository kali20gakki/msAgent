# Agent Config Retry 功能实现总结

## 概述

成功在 agent YAML 配置中添加了 retry 配置项，实现了从配置到 middleware 的完整链路。

---

## 新增/修改的文件

### 1. 核心配置 (`src/msagent/configs/`)

| 文件 | 修改内容 |
|------|----------|
| `agent.py` | 新增 `RetryPolicyConfig` 类，为 `AgentConfig` 和 `SubAgentConfig` 添加 `retry` 字段，添加版本迁移逻辑 |
| `__init__.py` | 导出 `RetryPolicyConfig` |
| `constants.py` | 更新 `AGENT_CONFIG_VERSION` 到 `"2.3.0"` |

### 2. Agent 创建 (`src/msagent/agents/`)

| 文件 | 修改内容 |
|------|----------|
| `deep_agent.py` | 添加 `_create_retry_middleware()` 函数，`create_deep_agent()` 添加 `retry_config` 参数 |
| `factory.py` | `AgentFactory.create()` 传递 `config.retry` 给 `create_deep_agent()` |

### 3. 配置文件 (`resources/configs/default/agents/`)

| 文件 | 修改内容 |
|------|----------|
| `msagent.yml` | 添加完整 retry 配置示例 |

### 4. 文档 (`docs/` 和 `examples/`)

| 文件 | 说明 |
|------|------|
| `AGENT_RETRY_CONFIG.md` | YAML 配置完整指南 |
| `agent_retry_examples.py` | 各种场景的配置示例 |

---

## 配置结构

```yaml
version: "2.3.0"  # 注意版本升级到 2.3.0
name: my_agent
# ... 其他配置 ...

retry:
  enabled: true                    # 总开关
  llm_max_retries: 3               # LLM 重试次数
  llm_base_delay: 1.0              # 初始延迟（秒）
  llm_max_delay: 60.0              # 最大延迟（秒）
  tool_max_retries: 2              # Tool 重试次数
  tool_timeout: 30.0               # Tool 超时（秒）
  enable_circuit_breaker: false    # 熔断器开关
  circuit_breaker_threshold: 5     # 熔断阈值
  circuit_breaker_recovery: 60.0   # 恢复时间（秒）
```

---

## 数据流

```
┌─────────────────┐
│  YAML Config    │  .msagent/agents/*.yml
│  (retry section)│
└────────┬────────┘
         │ parse
         ▼
┌─────────────────┐
│ RetryPolicyConfig│  pydantic model
│   (validation)   │
└────────┬────────┘
         │ create
         ▼
┌─────────────────┐
│  _create_retry_ │  deep_agent.py
│   middleware()   │
└────────┬────────┘
         │ build
         ▼
┌─────────────────┐
│ RetryMiddleware │  middlewares/retry.py
│ or CircuitBreaker│
└────────┬────────┘
         │ inject
         ▼
┌─────────────────┐
│ create_react_   │  react_agent.py
│     agent()      │
└────────┬────────┘
         │ wrap
         ▼
┌─────────────────┐
│ CompiledState   │  Final agent
│     Graph        │
└─────────────────┘
```

---

## 使用方式

### 1. YAML 配置（推荐）

```yaml
# .msagent/agents/my_agent.yml
version: "2.3.0"
name: my_agent
llm: default
retry:
  enabled: true
  llm_max_retries: 5
  enable_circuit_breaker: true
```

### 2. 自动加载

```python
from msagent.configs import BatchAgentConfig

# retry 配置自动从 YAML 加载
config = await BatchAgentConfig.from_yaml(
    Path(".msagent/config.agents.yml"),
    batch_llm_config=llm_config,
)
agent_config = config.get_default_agent()
print(agent_config.retry.llm_max_retries)  # 5
```

### 3. Agent 创建

```python
from msagent.agents.factory import AgentFactory

# retry middleware 自动创建并注入
agent = await agent_factory.create(
    config=agent_config,  # 包含 retry 配置
    ...
)
```

---

## 版本迁移

旧版本配置（2.2.1 及以下）会自动迁移：

```python
# 在 BaseAgentConfig.migrate() 中
if from_ver < pkg_version.parse("2.3.0"):
    if "retry" not in data:
        data["retry"] = {
            "enabled": True,
            "llm_max_retries": 3,
            ...  # 其他默认值
        }
```

---

## 配置继承

### Main Agent
```yaml
# agents/main.yml
name: main
retry:
  llm_max_retries: 3
```

### SubAgent
```yaml
# subagents/sub.yml  
name: sub
retry:
  llm_max_retries: 5  # 独立配置
```

SubAgent 可以使用与 Main Agent 不同的 retry 配置。

---

## 测试验证

```bash
# 运行 retry 中间件测试
uv run pytest tests/test_retry_middleware.py -v

# 验证配置加载
uv run python -c "
from msagent.configs import RetryPolicyConfig
c = RetryPolicyConfig(llm_max_retries=5)
print(c.model_dump())
"

# 验证 middleware 创建
uv run python -c "
from msagent.agents.deep_agent import _create_retry_middleware
from msagent.configs import RetryPolicyConfig

config = RetryPolicyConfig(enabled=True, llm_max_retries=3)
middleware = _create_retry_middleware(config)
print(type(middleware).__name__)
"
```

---

## 最佳实践

### 开发环境
```yaml
retry:
  enabled: false  # 快速失败，便于调试
```

### 生产环境
```yaml
retry:
  enabled: true
  llm_max_retries: 3
  tool_max_retries: 2
  tool_timeout: 30.0
  enable_circuit_breaker: true
```

### 不稳定网络
```yaml
retry:
  llm_max_retries: 5
  llm_base_delay: 0.5
  enable_circuit_breaker: true
```

---

## 后续扩展

1. **环境变量覆盖**: `MSAGENT_RETRY_LLM_MAX_RETRIES=5`
2. **动态配置**: 运行时通过 API 修改 retry 配置
3. **细粒度控制**: 按工具类型设置不同 retry 策略
4. **指标集成**: 自动上报重试指标到监控系统
