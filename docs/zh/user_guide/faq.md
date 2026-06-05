# FAQ

## 1. msagent 第一次启动时会生成哪些本地文件？

`msagent` 使用项目本地配置。首次在某个工作目录启动时，会自动生成：

```text
.msagent/
```

这个目录里通常会包含：

- LLM 配置
- Agent / SubAgent 配置
- MCP 配置
- Prompt 模板
- Skills
- 日志、缓存、会话历史和 checkpoint 数据

更完整的目录说明见 [配置与扩展](configuration-and-extension.md)。


## 2. 如何打开或关闭 MCP 服务？

有两种常见方式：

- 在会话里通过 `/mcp` 查看和切换已配置的 MCP 服务
- 直接编辑 `.msagent/config.mcp.json`

默认模板会启用 `msprof-mcp`。如果你要接入新的本地或远程 MCP 服务，建议先参考 [配置与扩展](configuration-and-extension.md) 里的字段说明。

## 3. 如何确认 Skill 是否被识别到了？

进入会话后，可以执行：

```text
/skills
```

如果要查看某个具体 Skill：

```text
/skills my-skill
```

如果你的 Skill 带分类目录，建议写全路径：

```text
/skills profiling/my-skill
```

如果看不到，通常优先检查：

- 路径是否正确
- 文件名是否为 `SKILL.md`
- 当前 Agent 的 `skills.patterns` 是否允许该 Skill
- 是否被更高优先级目录中的同名 Skill 覆盖

加载自定义 Skill，可参考[添加自定义 Skill](configuration-and-extension.md#添加自定义-skill)

## 3. 运行日志在哪里看？

启用后，日志会写到当前工作目录下：

```text
.msagent/logs/app.log
```

如果你想看到更详细的日志，可以开启：

```bash
export MSAGENT_LOG_LEVEL=DEBUG
msagent -v
```
