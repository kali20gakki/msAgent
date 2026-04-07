# 版本与兼容性

本文档汇总当前代码仓里的版本、运行要求与兼容性信息。

## 当前版本信息

| 项目 | 说明 |
|---|---|
| 当前版本 | `0.1.2` |
| 包名 | `mindstudio-agent` |
| 命令行入口 | `msagent` |
| Python 要求 | `>=3.11` |
| 默认内置 MCP | `msprof-mcp==0.1.6` |

## 当前支持的 LLM Provider

运行时当前支持以下三种 provider：

- `openai`
- `anthropic`
- `google`

兼容性说明：

- 历史配置中的 `gemini` 会被兼容处理为 `google`
- 不再单独提供 `custom` provider；自部署服务请按协议兼容性复用上述三种 provider，并通过 `--llm-base-url` 指向自定义地址

## 当前能力概览

- 支持默认交互式会话，也支持指定 Agent 启动
- 内置 `Hermes` 与 `Minos` 两个 Agent
- 支持 Ascend 性能调优、Profiling 分析、文档走查、工具咨询等场景
- 支持 MCP 扩展，默认启用 `msprof-mcp`
- 支持 Skills 扩展，源码仓库中的内置 Skills 由 `mindstudio-skills` 子模块提供
- 支持通过 `msagent web` 启动 Web UI

## 查看本地安装版本

```bash
msagent --version
```

## 版本策略

项目遵循语义化版本（SemVer）：

- 补丁版本以兼容性修复为主
- 次版本新增功能保持向后兼容
- 主版本包含不兼容变更
