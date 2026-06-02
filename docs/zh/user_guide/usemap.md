### 进入交互式会话后，可以直接输入问题，也可以配合 `/` 命令和快捷键提升效率。

| 命令 | 说明 |
|---|---|
| `/hotkeys` | 查看键盘快捷键说明。 |
| `/agents` | 打开 Agent 选择器。 |
| `/model` | 打开模型选择器。 |
| `/threads` | 浏览并恢复历史会话线程。 |
| `/tools` | 查看当前可用工具。 |
| `/skills` | 浏览当前可用 Skills。 |
| `/mcp` | 管理 MCP 服务启用状态。 |
| `/offload` | 压缩并卸载较早的会话消息。 |
| `/tool-output` | 打开最近一次可展开的工具输出。 |
| `/clear` | 清屏并开启新线程。 |
| `/exit` | 退出当前会话。 |

### 输入区快捷键

| 快捷键 | 说明 |
|---|---|
| `Ctrl+C` | 有输入时清空输入框；连续按两次退出会话。 |
| `Ctrl+J` | 插入换行，便于多行输入。 |
| `Shift+Tab` | 循环切换审批模式。 |
| `Ctrl+B` | 切换 bash mode。 |
| `Ctrl+K` | 直接打开快捷键说明。 |
| `Ctrl+O` | 打开最近一次可展开的工具输出。 |
| `Tab` | 应用第一个补全项。 |
| `Enter` | 提交输入；如果当前选中了补全项，则先应用补全。 |

### 工具输出查看器

当某次工具调用支持展开查看时，可用 `Ctrl+O` 或 `/tool-output` 打开工具输出查看器。查看器内支持：

- 左右方向键：切换不同工具调用输出
- 上下方向键、`PageUp` / `PageDown`、`Home` / `End`：滚动内容
- 点击、`Ctrl+O` 或 `Enter`：展开或折叠完整输出
- `Esc`：关闭查看器

### 按主题深入

需要更细致地配置或排查时，可按下列主题查阅对应文档：

| 主题 | 说明 | 文档 |
|---|---|---|
| 配置与扩展 | 本地配置目录、模型 Provider、MCP 配置、Skills 扩展与加载顺序 | [配置与扩展](configuration-and-extension.md) |
| 上下文压缩 | 长会话的上下文压缩与卸载机制（`/offload` 等）配置与行为 | [上下文压缩使用指南](context-compaction-guide.md) |
| 重试与超时 | 模型调用的 Retry Middleware 重试、退避与超时配置 | [Retry Middleware 使用指南](retry-middleware-guide.md) |
| 能力边界 | Agent / Tool / Skill 的过滤与匹配规则 | [Agent / Tool / Skill 过滤规则](agent-tool-skill-filter-rules.md) |
| 文档体验审查 | 使用内置 `document-ux-review` skill 走查 README 与上手流程 | [`document-ux-review` 使用说明](document-ux-review.md) |
| 常见问题 | 安装、配置与使用过程中的高频问题排查 | [FAQ](faq.md) |
