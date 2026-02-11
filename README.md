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
git clone https://github.com/kali20gakki/msAgent.git
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

# Gemini
uv run msagent config --llm-provider gemini --llm-api-key "your-key" --llm-model "gemini-2.0-flash"

# Custom OpenAI-compatible API (self-hosted gateway / proxy / compatible vendor)
uv run msagent config --llm-provider openai --llm-api-key "your-key" --llm-base-url "http://127.0.0.1:8045/v1" --llm-model "your-model-name"
```

### MCP 服务器管理

```bash
# 列出 MCP 服务器
msagent mcp list

# 添加 MCP 服务器
msagent mcp add --name filesystem --command npx --args "-y,@modelcontextprotocol/server-filesystem,/path"

# 移除 MCP 服务器
msagent mcp remove --name filesystem
```

### 查看帮助

```bash
# 显示帮助信息
msagent --help

# 显示版本
msagent --version

# 显示详细信息
msagent info
```

## 🔌 MCP 服务器示例

### 文件系统服务器

```bash
msagent mcp add --name filesystem --command npx --args "-y,@modelcontextprotocol/server-filesystem,/home/user/documents"
```

### SQLite 服务器

```bash
msagent mcp add --name sqlite --command npx --args "-y,@modelcontextprotocol/server-sqlite,/path/to/database.db"
```

### 自定义 MCP 服务器

```bash
msagent mcp add --name myserver --command python --args "/path/to/server.py"
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
| `msagent chat [message]` | 启动聊天会话 |
| `msagent ask <question>` | 单轮问答 |
| `msagent config --show` | 查看配置 |
| `msagent mcp list` | 列出 MCP 服务器 |
| `msagent mcp add --name <n> --command <c>` | 添加 MCP 服务器 |
| `msagent mcp remove --name <n>` | 移除 MCP 服务器 |
| `msagent info` | 显示信息 |
| `msagent --version` | 显示版本 |

## 🔗 相关链接

- [MCP Protocol](https://modelcontextprotocol.io/)
- [Textual](https://textual.textualize.io/)
- [Typer](https://typer.tiangolo.com/)

## 📄 许可证

MIT License
