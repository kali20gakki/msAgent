# 🚀 MSProf Agent

一个功能强大的 AI Agent CLI 客户端，支持多种 LLM 提供商和 MCP (Model Context Protocol) 协议。

## ✨ 特性

- 💬 **多轮对话** - 支持交互式聊天和单轮问答
- 🔌 **MCP 协议支持** - 通过 stdio 方式集成 MCP 服务器
- 🌊 **流式输出** - 实时显示 AI 响应
- 🎨 **精美 TUI** - 基于 Textual 的终端用户界面
- ⚙️ **灵活配置** - 支持配置文件和环境变量
- 🤖 **多模型支持** - OpenAI、Anthropic Claude、Google Gemini、自定义 API

## 📦 安装

### 使用 uv 安装

```bash
# 克隆项目
git clone <repository-url>
cd msprof-agent

# 使用 uv 安装依赖
uv sync

# 安装到当前环境
uv pip install -e .
```

### 使用 pip 安装

```bash
pip install -e .
```

## 🔧 配置

### 环境变量配置

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"  # 可选

# Anthropic Claude
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export ANTHROPIC_MODEL="claude-3-5-sonnet-20241022"  # 可选

# Google Gemini
export GEMINI_API_KEY="your-gemini-api-key"
export GEMINI_MODEL="gemini-pro"  # 可选

# 自定义 API
export CUSTOM_API_KEY="your-api-key"
export CUSTOM_BASE_URL="https://api.example.com/v1"
export CUSTOM_MODEL="your-model"
```

### 配置文件

配置文件位于 `~/.config/msprof-agent/config.json`：

```json
{
  "llm": {
    "provider": "openai",
    "api_key": "your-api-key",
    "base_url": "",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "mcp_servers": [
    {
      "name": "filesystem",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"],
      "env": {},
      "enabled": true
    }
  ],
  "theme": "dark"
}
```

## 🚀 使用

### 启动交互式聊天

```bash
# 命令行交互模式
msprof chat

# TUI 界面模式
msprof chat --tui

# 发送单条消息
msprof chat "Hello, how are you?"
```

### 快速提问

```bash
msprof ask "What is the weather today?"
```

### 查看配置

```bash
# 显示当前配置
msprof config --show

# 设置 LLM 提供商
msprof config --llm-provider openai --llm-base-url "http://example.com/v1" --llm-api-key "your-key" --llm-model "your-model"
```

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
