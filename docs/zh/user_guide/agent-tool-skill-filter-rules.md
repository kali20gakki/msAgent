# Agent YAML Tool/Skills 过滤规则说明

本文说明 `agent yml` 里 `tools` 与 `skills` 的配置方式、匹配语义、以及一份可直接实测的示例。

## 1. tools 怎么配

`tools.patterns` 使用三段式：

```text
<category>:<module>:<name_pattern>
```

- `category`:
  - `impl`：内建实现工具（当前统一归到 `deepagents` 模块）
  - `mcp`：MCP 工具
- `module`:
  - 对 `impl` 固定用 `deepagents`
  - 对 `mcp` 使用 MCP server 名（如 `msprof-mcp`）
- `name_pattern`:
  - 工具名，支持 `fnmatch` 通配符（`*`、`?`、`[abc]`）
- 负规则：前缀 `!`，例如 `!impl:deepagents:run_tool`

示例：

```yaml
tools:
  patterns:
    - impl:deepagents:get_tool
    - impl:deepagents:run_tool
    - impl:deepagents:fetch_skills
    - impl:deepagents:get_skill
    - mcp:msprof-mcp:*
    - "!mcp:msprof-mcp:*_debug"
  use_catalog: false
```

## 2. skills 怎么配

`skills.patterns` 使用两段式：

```text
<category>:<name_pattern>
```

- `category`：通常是 `default`，也可匹配其他分类
- `name_pattern`：技能名，支持 `fnmatch` 通配符
- 负规则：前缀 `!`，例如 `!default:op-mfu-calculator`

示例：

```yaml
skills:
  patterns:
    - default:mindstudio_profiler_data_check
    - default:ascend_pytorch_profiler_db_explorer
    - "!default:op-*"
  use_catalog: false
```

## 3. 过滤规则（硬生效语义）

### tools 规则

- 先收集正规则（不带 `!`）和负规则（带 `!`）。
- 仅当“命中任一正规则”且“未命中任何负规则”时，工具才可用。
- 如果没有正规则，结果是“全部禁用”。
- 格式非法（不是三段）会被忽略并打印 warning。
- 过滤在两层生效：
  - `AgentFactory.create` 构建工具列表时过滤一次。
  - model-call middleware 再过滤一次，避免 deepagents 默认注入工具绕过。

### skills 规则

- 仅当“命中任一正规则”且“未命中任何负规则”时，技能保留。
- 如果没有正规则，结果是“无技能”。
- skills 过滤后的结果进入 runtime cache，并作为 `fetch_skills/get_skill` 的优先来源。

## 4. 兼容策略（当前约束）

- 不做 runtime 兼容别名（如 `execute <-> run_command`）。
- 不做 legacy module 兼容（如 `file_system` / `grep_search` / `terminal`）。
- 只按 deepagents 最新接口命名匹配。

## 5. 一份可直接实测的完整 yml

可参考：

- `resources/configs/default/agents/msagent-filter-smoke.yml`

关键片段如下：

```yaml
tools:
  patterns:
    - impl:deepagents:get_tool
    - impl:deepagents:run_tool
    - impl:deepagents:fetch_skills
    - impl:deepagents:get_skill
    - mcp:msprof-mcp:*
  use_catalog: false

skills:
  patterns:
    - default:mindstudio_profiler_data_check
    - default:ascend_pytorch_profiler_db_explorer
  use_catalog: false
```

## 6. 实测建议

启动后询问：

```text
你有哪些mcp、tool、skills
```

预期：

- skills 仅返回 `patterns` 允许的集合
- tools 仅返回允许的 `impl:deepagents:*` + `mcp:msprof-mcp:*`
- 不应再出现被排除的 skill/tool
