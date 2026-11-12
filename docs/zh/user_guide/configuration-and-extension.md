# 配置与扩展

本文档汇总 `msAgent` 当前代码实现中的全局配置、项目状态、MCP 扩展和 Skills 扩展方式。

## 1. 全局配置目录

`msAgent` 将用户配置统一保存在 `MSAGENT_HOME`。未设置该环境变量时，默认目录为：

```text
~/.msagent/
```

工作目录只作为 Agent 和工具的执行根目录，不再生成项目内 `.msagent/`。安装包中的默认配置直接从 wheel 读取；全局目录只保存用户覆盖项。

现阶段不会自动导入已有的 `<working-dir>/.msagent/`，请先保留旧目录，等待后续迁移功能。

## 2. 默认模板与运行时文件

默认模板中的主要内容如下：

| 路径 | 说明 |
|---|---|
| `~/.msagent/config/config.llms.yml` | 默认模型覆盖配置。 |
| `~/.msagent/config/llms/*.yml` | 额外的模型别名覆盖配置。 |
| `~/.msagent/config/agents/*.yml` | 高级扩展使用的 Agent 最小覆盖或新增定义；正常切换不会生成。 |
| `~/.msagent/config/subagents/*.yml` | 高级扩展使用的 SubAgent 最小覆盖或新增定义。 |
| `~/.msagent/config/checkpointers/*.yml` | Checkpointer 覆盖配置。 |
| `~/.msagent/prompts/` | 用户 Prompt 文件。 |
| `~/.msagent/skills/` | 用户安装的 Skills。 |
| `~/.msagent/config/config.mcp.json` | MCP 服务器覆盖配置。 |
| `~/.msagent/config/config.approval.json` | 工具审批规则覆盖配置。 |

运行时会按需生成以下文件或目录：

| 路径 | 说明 |
|---|---|
| `~/.msagent/state/projects/<project-id>/project.json` | 工作目录标识，以及该项目当前 Agent、各 Agent 的模型偏好。 |
| `~/.msagent/state/projects/<project-id>/checkpoints.sqlite` | 当前项目的 checkpoint 数据库。 |
| `~/.msagent/state/projects/<project-id>/history` | 当前项目的输入历史。 |
| `~/.msagent/state/projects/<project-id>/memory.md` | 当前项目的长期记忆。 |
| `~/.msagent/state/projects/<project-id>/config.approval.json` | 当前项目的命令审批持久化规则。 |
| `~/.msagent/state/projects/<project-id>/conversation_history/` | 当前项目的会话历史。 |
| `~/.msagent/state/projects/<project-id>/audit_log/` | 当前项目的审计日志。 |
| `~/.msagent/cache/mcp/` | 全局 MCP 运行缓存。 |
| `~/.msagent/oauth/mcp/` | 全局 MCP OAuth 数据。 |
| `~/.msagent/logs/` | 全局日志目录。 |

## 3. 项目长期记忆

`~/.msagent/state/projects/<project-id>/memory.md` 用于保存当前项目的长期记忆。可以在交互式会话中使用以下命令维护：

| 命令 | 说明 |
|---|---|
| `/remember <content>` | 追加一条长期记忆。首次使用时会自动创建项目 memory 文件。 |
| `/showmemory` | 查看当前项目已保存的长期记忆。 |

后续会话会自动读取对应项目的 memory 文件，并作为 `<user-memory>` 上下文提供给 Agent。默认模板内容不会作为有效记忆注入。

长期记忆适合保存稳定信息，不建议写入 API Key、密码、令牌等敏感数据。

## 4. 命令审批与权限管理

审批模式、触发场景和项目授权管理见[权限审批使用指南](permission-approval-guide.md)。

## 5. 配置读取方式

当前实现支持“单文件配置”和“目录配置”两种方式并存：

- LLM：内置默认值叠加 `~/.msagent/config/config.llms.yml` 与 `config/llms/*.yml`
- Agent：内置默认值叠加 `~/.msagent/config/config.agents.yml` 与 `config/agents/*.yml`
- SubAgent：内置默认值叠加 `~/.msagent/config/config.subagents.yml` 与 `config/subagents/*.yml`
- Checkpointer：内置默认值叠加 `~/.msagent/config/config.checkpointers.yml` 与 `config/checkpointers/*.yml`

默认模板当前以目录式配置为主。

内置 Agent 的 Prompt、Tools、Skills、SubAgent 等完整定义由安装包维护，普通用户无需复制或编辑。`/agents` 切换结果保存在当前项目的 `project.json`；`/models` 的选择也按 Agent 保存在同一项目状态中。启动时的选择优先级为：显式 `--agent`/`--model` 参数、当前项目偏好、安装包默认值。

只有高级扩展或显式修改 Agent 字段时才会生成 `config/agents/*.yml`。生成文件采用字段级最小覆盖，未出现的字段继续继承安装包默认定义。

## 6. MCP 配置

内置 MCP 定义来自安装包中的 `resources/configs/default/config.mcp.json`，用户文件按 server name 覆盖内置定义。以下是一个本地 stdio 服务示例：

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

内置 MCP 服务说明：

- `msprof-mcp`：Ascend Profiling 数据分析服务（本地 stdio，默认启用）。
- `ascend-doc-mcp`：昇腾社区文档与开源仓库资料查询服务（默认启用）。通过 `msagent-ascend-doc-mcp` 启动器以 `npx` 拉起 `@opencxd/ascend-doc-mcp`，默认查询 Ascend 社区文档（产品/版本/文档目录/检索/开源仓库文件/安装下载）。需要 Node.js（≥22）与 npm/npx，首次启动会从 npm 拉取包；安装脚本已把该包预装到 `~/.msagent/ascend-doc-mcp`，预装成功时启动器直接运行本地副本，运行期无需 npm 且可离线。npm registry 的解析顺序为 `MSAGENT_NPM_REGISTRY` → 本机 npm 自身配置的 registry（`npm config get registry`，含企业下发的 `.npmrc`，内网机器据此自动使用内网源）→ `registry.npmmirror.com` → `mirrors.huaweicloud.com/repository/npm` → `registry.npmjs.org`，逐项探测后选用（内网场景见 [FAQ](../support/faq.md) 第 9 条），`MSAGENT_NPM_REGISTRY_ONLY=1` 可强制只使用显式指定的 registry。该服务不可用时不会影响会话启动，只会使 `ascend-knowledge` 子代理无法执行文档/仓库查询并向主 Agent 说明不可用；也可在会话中用 `/mcp` 关闭该服务。
- `tavily-mcp`：网络检索服务（默认关闭，配置 `TAVILY_API_KEY` 后可在 `/mcp` 中启用）。

日常使用方式：

- 用 `/mcp` 在会话中切换已有 MCP 服务的启用状态
- 直接编辑 `~/.msagent/config/config.mcp.json` 来新增、删除或调整服务定义

## 7. Skills 扩展

当前 Skills 会按以下顺序扫描：

1. `<working-dir>/skills`
2. `~/.msagent/skills`
3. 内置 Skills 目录：优先使用仓库根目录 `skills/`；已安装 wheel 时使用打包后的资源

同名 Skill 按“先加载优先”处理，因此当前优先级是：

1. 项目根目录下的 `skills/`
2. 全局用户 Skills
3. 内置 Skills

## 8. Skill 目录结构

支持以下两种目录结构：

```text
skills/
  my-skill/
    SKILL.md
```

```text
skills/
  profiler/
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

## 9. 源码运行时的内置 Skills

内置 Skills 已直接合入 `msagent` 主仓库，源码运行时默认使用仓库根目录：

```text
skills/
```

构建 wheel 时，上述目录会被打包到：

```text
resources/configs/default/skills/
```

因此源码运行和安装运行共用同一份 Skills 内容，不再需要额外执行 `git submodule` 同步。

## 10. 添加自定义 Skill

如果你希望在当前项目里扩展一个新的 Skill，推荐直接放在仓库根目录：

```text
skills/
  my-skill/
    SKILL.md
```

也支持带分类的结构：

```text
skills/
  profiler/
    my-skill/
      SKILL.md
```

这时：

- `profiler` 是分类
- `my-skill` 是 skill 名称

### 10.1 编写 `SKILL.md`

最小示例：

```md
---
name: my-skill
description: 用于处理某类固定任务的自定义 skill
---

# My Skill

当用户提出这类需求时使用这个 skill：

- 分析日志
- 生成报告

执行要求：

1. 先检查输入是否完整。
2. 优先读取项目内已有配置和样例。
3. 输出结论、依据和建议。
```

说明：

- `name` 不写时，默认取目录名
- `description` 建议填写，便于在 `/skills` 中识别
- 如果需要脚本或模板，可以继续放在 skill 目录下，例如 `scripts/`、`templates/`

### 10.2 让 Agent 能看到它

只创建文件还不够，当前 agent 还需要在配置里放开这个 skill。

例如：

```yaml
skills:
  patterns:
    - default:my-skill
  use_catalog: false
```

如果是带分类的 skill：

```yaml
skills:
  patterns:
    - profiler:my-skill
  use_catalog: false
```

规则格式是：

```text
<category>:<name_pattern>
```

这部分更完整的匹配语义，可参考 [Agent / Tool / Skill 过滤规则](agent-tool-skill-filter-rules.md)。

### 10.3 验证是否生效

启动 `msagent` 后，可以这样检查：

```text
/skills
```

或者直接指定：

```text
/skills my-skill
```

如果有重名 skill，建议写全：

```text
/skills profiler/my-skill
```

(custom-skill-faq)=

## 11. Skill 常见问题

### 11.1 `/skills` 看不到新 Skill

通常优先检查这几项：

- 路径是否正确，文件名是否是 `SKILL.md`
- 当前 agent 是否配置了对应的 `skills.patterns`
- 是否被更高优先级目录中的同名 skill 覆盖

### 11.2 Skill 存在，但 Agent 不会自动使用

这通常是以下原因之一：

- `description` 太泛，模型难以判断触发场景
- agent 的 `skills.patterns` 没有放开对应 skill
- 当前任务更匹配别的内置 skill，导致没有选中它

建议先通过 `/skills` 确认可见性，再补充更明确的 `description` 和触发说明。
