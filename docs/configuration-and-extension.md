# 配置与扩展

本文档汇总 `msAgent` 当前代码实现中的项目本地配置、MCP 扩展和 Skills 扩展方式。

## 项目本地配置目录

`msAgent` 使用项目本地配置。首次在某个工作目录运行时，会自动把 `resources/configs/default/` 中的默认模板复制到：

```text
<working-dir>/.msagent/
```

如果当前工作目录是 Git 仓库，运行时还会尝试把 `.msagent/` 加入 `.git/info/exclude`，避免本地配置被误提交。

## 默认模板与运行时文件

默认模板中的主要内容如下：

| 路径 | 说明 |
|---|---|
| `.msagent/config.llms.yml` | 默认模型配置入口文件。 |
| `.msagent/llms/*.yml` | 额外的模型别名配置。 |
| `.msagent/agents/*.yml` | Agent 定义。当前默认包含 `Hermes.yml`、`Minos.yml`。 |
| `.msagent/subagents/*.yml` | SubAgent 定义。 |
| `.msagent/checkpointers/*.yml` | Checkpointer 配置。当前默认包含 `memory.yml`、`sqlite.yml`。 |
| `.msagent/sandboxes/*.yml` | Sandbox 配置模板。 |
| `.msagent/prompts/` | Agent / SubAgent 使用的 Prompt 模板。 |
| `.msagent/skills/` | 随模板分发的内置 Skills。 |
| `.msagent/config.mcp.json` | MCP 服务器配置。 |
| `.msagent/config.approval.json` | 工具审批规则配置。 |
| `.msagent/README.md` | 本地配置目录说明。 |

运行时会按需生成以下文件或目录：

| 路径 | 说明 |
|---|---|
| `.msagent/config.checkpoints.db` | 会话 checkpoint 数据库。 |
| `.msagent/.history` | 输入历史。 |
| `.msagent/memory.md` | 用户偏好和项目上下文记忆。 |
| `.msagent/cache/mcp/` | MCP 运行缓存。 |
| `.msagent/oauth/mcp/` | MCP OAuth 相关缓存。 |
| `.msagent/logs/` | 本地日志目录。 |
| `.msagent/conversation_history/` | 会话历史目录。 |

## 配置读取方式

当前实现支持“单文件配置”和“目录配置”两种方式并存：

- LLM：读取 `.msagent/config.llms.yml` 与 `.msagent/llms/*.yml`
- Agent：优先读取 `.msagent/config.agents.yml`（如果存在）以及 `.msagent/agents/*.yml`
- SubAgent：优先读取 `.msagent/config.subagents.yml`（如果存在）以及 `.msagent/subagents/*.yml`
- Checkpointer：优先读取 `.msagent/config.checkpointers.yml`（如果存在）以及 `.msagent/checkpointers/*.yml`
- Sandbox：读取 `.msagent/sandboxes/*.yml`

默认模板当前以目录式配置为主。

## MCP 配置

默认模板会启用 `msprof-mcp`。当前默认配置等价于：

```json
{
  "mcpServers": {
    "msprof-mcp": {
      "command": "msprof-mcp",
      "args": [],
      "transport": "stdio",
      "env": {},
      "include": [],
      "exclude": [],
      "enabled": true,
      "stateful": true,
      "repair_timeout": 30,
      "invoke_timeout": 3600.0
    }
  }
}
```

常用字段如下：

| 字段 | 说明 |
|---|---|
| `command` | 本地 MCP 服务启动命令。 |
| `url` | 远程 MCP 服务地址。 |
| `headers` | 远程 MCP 请求头。 |
| `args` | `command` 的参数列表。 |
| `transport` | 传输协议，支持 `stdio`、`sse`、`http`、`websocket`。 |
| `env` | 启动本地 MCP 进程时注入的环境变量。 |
| `include` / `exclude` | 允许或排除的工具列表。 |
| `enabled` | 是否启用该 MCP 服务。 |
| `stateful` | 是否保持连接，避免每次调用都重新拉起。 |
| `repair_command` | 初始化失败时可选的修复命令列表。 |
| `repair_timeout` | 修复命令超时，单位秒。 |
| `timeout` | 建立连接的超时。 |
| `sse_read_timeout` | SSE 读取超时。 |
| `invoke_timeout` | 单次工具调用超时。 |

说明：

- `streamable_http` / `streamable-http` 会在运行时规范化为 `http`
- `repair_command` 配置存在但未显式设置 `repair_timeout` 时，默认补成 `30`
- 对于 `msprof-mcp` 这类本地 `stdio` MCP，通常优先关注 `stateful` 与 `invoke_timeout`

日常使用方式：

- 用 `/mcp` 在会话中切换已有 MCP 服务的启用状态
- 直接编辑 `.msagent/config.mcp.json` 来新增、删除或调整服务定义

## Skills 扩展

当前 Skills 会按以下顺序扫描：

1. `<working-dir>/skills`
2. 内置 Skills 目录：`resources/configs/default/skills/`
3. `<working-dir>/.msagent/skills`

同名 Skill 按“先加载优先”处理，因此当前优先级是：

1. 项目根目录下的 `skills/`
2. 内置 Skills
3. `.msagent/skills/`

## Skill 目录结构

支持以下两种目录结构：

```text
skills/
  my-skill/
    SKILL.md
```

```text
skills/
  profiling/
    my-skill/
      SKILL.md
```

其中 `SKILL.md` 建议包含 frontmatter，至少提供：

```yaml
---
name: my-skill
description: 这个技能做什么
---
```

## 源码运行时的内置 Skills

源码仓库中的内置 Skills 来自 `mindstudio-skills` 子模块，对应路径：

```text
resources/configs/default/skills/
```

如果你是源码运行，建议至少执行一次：

```bash
git submodule sync --recursive
git submodule update --init --recursive resources/configs/default/skills
```

如果你希望同步上游最新提交，可改用：

```bash
git submodule sync --recursive
git submodule update --init --recursive --remote resources/configs/default/skills
```

说明：

- 不带 `--remote`：同步到主仓库当前记录的 Skills 版本
- 带 `--remote`：同步到 `mindstudio-skills` 上游默认分支的最新提交
- 执行 `--remote` 后，子模块指针会变化；如果要固定版本，记得提交对应变更
