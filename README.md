# 🚀 msAgent

**msAgent** 是一个强大的命令行智能助手，专为开发者和运维人员设计。它不仅具备与大语言模型（LLM）对话的能力，还能通过 Model Context Protocol (MCP) 扩展各种本地工具，例如文件操作、代码分析、系统监控等。

<p align="center">
  <img src="docs/img/msagent.png" alt="msAgent">
</p>

## ✨ 核心特性

- **多模态交互**：支持基于 Textual 的现代化 TUI 界面，同时也提供简洁的命令行交互模式。
- **MCP 扩展支持**：原生支持 Model Context Protocol (MCP)，可以无缝集成任何符合 MCP 标准的工具（如 Fetch, Filesystem 等）。
- **多 LLM 支持**：灵活切换 OpenAI, Anthropic, Google Gemini 等多种大模型后端。
- **智能上下文管理**：自动根据任务需求调用相应的工具，无需手动介入。
- **流式响应**：实时的打字机效果，让对话更加自然流畅。

## 📦 快速开始

### 安装

使用 `uv` 进行安装（推荐）：

```bash
# Clone the repository
git clone https://github.com/weizhang/msagent.git
cd msagent

# Install dependencies and the tool
uv python install 3.12
uv sync
```

### 启动对话

#### TUI 模式（推荐）

启动现代化的终端用户界面：

```bash
uv run msagent chat --tui
```

#### 命令行模式

启动简单的命令行对话：

```bash
uv run msagent chat
```

## ⚙️ 配置指南

msAgent 需要配置 LLM 后端才能工作。首次运行会自动创建配置文件。

### 查看当前配置

```bash
uv run msagent config --show
```

### 设置 LLM 提供商

```bash
# OpenAI
uv run msagent config --llm-provider openai --llm-api-key "your-key" --llm-model "gpt-4"

# Anthropic
uv run msagent config --llm-provider anthropic --llm-api-key "your-key" --llm-model "claude-3-opus-20240229"

# Google Gemini

### MCP 服务器管理

```bash
# 列出 MCP 服务器
msprof mcp list

# 添加 MCP 服务器
msprof mcp add --name filesystem --command npx --args "-y,@modelcontextprotocol/server-filesystem,/path"

# 移除 MCP 服务器
msprof mcp remove --name filesystem
```

### 查看帮助

```bash
# 显示帮助信息
msprof --help

# 显示版本
msprof --version

# 显示详细信息
msprof info
```

## 🔌 MCP 服务器示例

### 文件系统服务器

```bash
msprof mcp add --name filesystem --command npx --args "-y,@modelcontextprotocol/server-filesystem,/home/user/documents"
```

### SQLite 服务器

```bash
msprof mcp add --name sqlite --command npx --args "-y,@modelcontextprotocol/server-sqlite,/path/to/database.db"
```

### 自定义 MCP 服务器

```bash
msprof mcp add --name myserver --command python --args "/path/to/server.py"
```

## 📁 项目结构

```
msprof-agent/
├── pyproject.toml          # 项目配置和依赖
├── README.md               # 项目文档
├── src/
│   └── msprof_agent/
│       ├── __init__.py     # 包初始化
│       ├── cli.py          # CLI 命令接口
│       ├── tui.py          # TUI 界面
│       ├── agent.py        # Agent 核心逻辑
│       ├── llm.py          # LLM 客户端
│       ├── mcp_client.py   # MCP 客户端
│       └── config.py       # 配置管理
```

## 🛠️ 开发

### 安装开发依赖

```bash
uv sync --dev
```

### 运行测试

```bash
uv run pytest
```

### 代码格式化

```bash
uv run ruff format .
uv run ruff check .
```

## 📝 命令参考

| 命令 | 描述 |
|------|------|
| `msprof chat [message]` | 启动聊天会话 |
| `msprof ask <question>` | 单轮问答 |
| `msprof config --show` | 查看配置 |
| `msprof mcp list` | 列出 MCP 服务器 |
| `msprof mcp add --name <n> --command <c>` | 添加 MCP 服务器 |
| `msprof mcp remove --name <n>` | 移除 MCP 服务器 |
| `msprof info` | 显示信息 |
| `msprof --version` | 显示版本 |

## 🔗 相关链接

- [MCP Protocol](https://modelcontextprotocol.io/)
- [OpenAI API](https://platform.openai.com/docs)
- [Anthropic API](https://docs.anthropic.com/)
- [Gemini API](https://ai.google.dev/)
- [Textual](https://textual.textualize.io/)
- [Typer](https://typer.tiangolo.com/)

## 📄 许可证

MIT License
