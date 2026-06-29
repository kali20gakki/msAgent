# 添加自定义 Agent

msAgent 中的 Agent 采用声明式配置定义，无需编写 Python 代码。添加新 Agent 只需在 `resources/configs/default/` 中创建 YAML 配置文件和对应的 Prompt 文件即可。

## 1. 概述

一个完整的 Agent 由两部分组成：

- **YAML 配置文件**：定义 Agent 的名称、模型、工具、Skill、子 Agent 等
- **Prompt 文件**：定义 Agent 的系统提示词，决定其行为和能力

编辑 `resources/configs/default/` 下的文件后，首次运行时会自动拷贝到用户的 `.msagent/` 目录。

## 2. 目录结构

```text
resources/configs/default/
├── agents/                          # Agent 配置（一个文件一个 Agent）
│   ├── Profiler.yml
│   └── MyAgent.yml                  # 新增的 Agent
├── subagents/                       # 子 Agent 配置（可选）
│   └── my-subagent.yml
├── prompts/
│   └── agents/                      # Agent 的 Prompt 文件
│       └── MyAgent.md
└── ...
```

## 3. 创建 Agent 配置文件

在 `resources/configs/default/agents/` 下创建 `MyAgent.yml`（文件名必须与 `name` 一致）：

```yaml
version: __APP_VERSION__
name: MyAgent
description: 我的自定义 Agent，用于 XXX 场景
prompt:
  - prompts/agents/MyAgent.md
  - prompts/suffixes/environments.md
llm: default
checkpointer: sqlite
default: false
subagents:
  - explorer
recursion_limit: 1000
tools:
  patterns:
    - impl:deepagents:*
  use_catalog: false
  output_max_tokens: 10000
skills:
  patterns:
    - default:ascend-computation-analysis
  use_catalog: false
compression:
  auto_compress_enabled: true
  auto_compress_threshold: 0.8
  llm: default
  prompt:
    - prompts/shared/general_compression.md
    - prompts/suffixes/environments.md
retry:
  enabled: true
  model:
    enabled: true
    max_retries: 5
    timeout: 120.0
  tool:
    enabled: true
    max_retries: 5
```

### 3.1 关键字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `version` | 是 | 配置版本，使用 `__APP_VERSION__` 占位符，运行时会自动替换 |
| `name` | 是 | Agent 唯一标识，**必须与文件名一致**（如 `MyAgent.yml` 对应 `name: MyAgent`） |
| `description` | 是 | Agent 功能简述，展示在 `/agents` 列表中 |
| `prompt` | 是 | Prompt 文件路径列表（相对于 `.msagent/`），按顺序拼接 |
| `llm` | 是 | 使用的 LLM 别名，`default` 对应默认模型，也可指定 `sonnet`、`opus` 等 |
| `checkpointer` | 是 | 会话持久化后端，通常为 `sqlite` |
| `default` | 是 | 是否为默认 Agent，**整个系统中必须恰好有一个 `default: true`** |
| `subagents` | 否 | 引用的子 Agent 名称列表，需在 `.msagent/subagents/` 中定义 |
| `tools.patterns` | 是 | 工具过滤规则，格式 `类别:模块:名称`，支持 `*` 通配符，`!` 前缀表示排除 |
| `skills.patterns` | 是 | Skill 过滤规则，格式 `类别:名称`，完整列表见 [Skills](../../../skills/README.md) |

### 3.2 tools.patterns 说明

格式为 `类别:模块:名称`：

| 类别 | 说明 | 示例 |
|------|------|------|
| `impl` | 内置实现工具 | `impl:deepagents:*`（所有内置工具） |
| `mcp` | MCP 服务工具 | `mcp:msprof-mcp:*`（msprof-mcp 全部工具） |
| `internal` | 内部工具 | `internal:*:*` |

排除某工具用 `!` 前缀：`!impl:deepagents:some_tool`

## 4. 创建 Prompt 文件

在 `resources/configs/default/prompts/agents/` 下创建 `MyAgent.md`，定义 Agent 的系统提示词：

```markdown
# MyAgent - XXX 助手

你是 MyAgent，一个专注于 XXX 场景的 AI 助手。

## 硬性规则

1. **规则一**：描述行为约束
2. **规则二**：描述输出要求

## Skill 调用规则

当任务匹配以下场景时，调用 `get_skill(name="<skill-name>")`：

| Skill 名称 | 适用场景 |
|------------|----------|
| `ascend-computation-analysis` | 计算瓶颈分析 |

## 输出规范

优先使用以下结构：

问题 / 证据 / 根因 / 建议
```

Prompt 文件的写法参考现有 Agent 的 Prompt（如 [Profiler.md](../../../resources/configs/default/prompts/agents/Profiler.md)）。

## 5. 验证

### 5.1 uv run 快速验证

```bash
uv run msagent --agent MyAgent
```

### 5.2 编包安装验证

重新编包安装后，启动验证：

```bash
msagent --agent MyAgent
```

编包与安装流程详见《[编译与打包](./build-and-package.md)》。
