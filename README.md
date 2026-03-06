<h1 align="center">🚀 msAgent</h1>

<p align="center"><strong>面向 Ascend NPU 场景的性能问题定位助手</strong></p>

**msAgent** 聚焦“发现瓶颈 -> 定位根因 -> 给出建议”的分析闭环。  
它结合 LLM 推理能力与可扩展工具链，帮助你把复杂 Profiling 信息快速转化为可执行的优化决策。

<p align="center">
  <img src="https://raw.githubusercontent.com/kali20gakki/images/main/msagent.gif" alt="msAgent">
</p>

## 🔍 支持的分析场景与扩展能力

- ⚙️ 单卡性能问题：高耗时算子、计算热点、重叠度不足等
- 🔗 多卡性能问题：快慢卡差异、通信效率瓶颈、同步等待等
- ⏱️ 下发与调度问题：下发延迟、CPU 侧调度阻塞等
- 🧩 集群性能问题：慢节点识别与从全局到单机的逐层定位
- 🔌 MCP 扩展：基于 Model Context Protocol 接入工具（默认启用 `msprof-mcp`）
- 🧠 Skills 扩展：自动加载 `skills/` 目录技能，复用领域分析流程和知识
---

## 使用示例
- MFU计算
<p align="center">
  <img src="https://raw.githubusercontent.com/kali20gakki/images/main/mfu.jpeg" alt="mfu">
</p>


## ⚡ 快速上手

### 1) 🧰 准备环境

- Python 3.11+
- 可用的 LLM API Key（OpenAI / Anthropic / Gemini / 兼容 OpenAI 接口）

### 2) 📦 安装（推荐：PyPI）

```bash
pip install -U mindstudio-agent
```

安装完成后可用以下命令确认：

```bash
msagent --version
```

### 3) 🔐 配置 LLM（必做）

推荐先用 OpenAI：

```bash
export OPENAI_API_KEY="your-key"
msagent config --llm-provider openai --llm-api-key-env OPENAI_API_KEY --llm-model "gpt-4o-mini"
```

检查配置是否生效：

```bash
msagent config --show
```

### 4) 🖥️ 启动 TUI

```bash
msagent chat --tui
```

### 5) 📊 性能分析

把 Profiling 目录路径和你的问题一起发给 msAgent，例如：

```text
请分析 /path/to/profiler_output 的性能瓶颈，重点关注通信和高耗时算子。
```

### 6) 🧪 可选：从源码运行（开发场景）

如需调试或二次开发，再使用源码方式：

```bash
git clone --recurse-submodules https://github.com/kali20gakki/msAgent.git
cd msAgent
uv sync
uv run msagent chat --tui
```

如果你已经完成普通 `git clone`，请补充执行：

```bash
git submodule update --init --recursive
```

---

## 📚 常用命令

如果你是源码方式（`git clone` + `uv sync`）运行，请在下列命令前加 `uv run`。

| 命令 | 说明 |
|---|---|
| `msagent chat --tui` | 启动 TUI 交互 |
| `msagent chat` | 启动 CLI 交互 |
| `msagent ask "..."` | 单轮提问 |
| `msagent config --show` | 查看当前配置 |
| `msagent mcp list` | 查看 MCP 服务器 |
| `msagent info` | 查看工具信息 |

---

## 🧵 会话管理（新对话 Session）

参考 Codex / Claude Code 的交互体验，msAgent 现在支持一键切换到新会话：

- 在 TUI 输入框中输入 `/new`
- 或使用快捷键 `Ctrl+N`
- 切换后会立即清空上下文（历史消息与上下文 token），从全新 Session 开始对话

常用会话命令（TUI 输入框）：

| 命令 | 说明 |
|---|---|
| `/new` | 开启新 Session（清空上下文） |
| `/clear` | 清空当前 Session 的聊天历史 |
| `/exit` | 退出会话 |

---

## 🛠️ 参考：配置与扩展

### 🤖 LLM 配置示例

OpenAI:

```bash
export OPENAI_API_KEY="your-key"
msagent config --llm-provider openai --llm-api-key-env OPENAI_API_KEY --llm-model "gpt-4o-mini"
```

Anthropic:

```bash
export ANTHROPIC_API_KEY="your-key"
msagent config --llm-provider anthropic --llm-api-key-env ANTHROPIC_API_KEY --llm-model "claude-3-5-sonnet-20241022"
```

Gemini:

```bash
export GEMINI_API_KEY="your-key"
msagent config --llm-provider gemini --llm-api-key-env GEMINI_API_KEY --llm-model "gemini-2.0-flash"
```

说明：OpenAI 兼容接口与 OpenAI Provider 共用 `openai`（通过 `--llm-base-url` 指向兼容服务）。

`max_tokens` 默认建议使用自动模式（`0`）：
- 自动模式不会向模型显式传 `max_tokens`，由服务端按模型默认值控制（最省维护）
- 适配新模型时无需更新本地“模型参数表”
- 如需手动覆盖，可用 `--llm-max-tokens <value>`

OpenAI 兼容接口（自定义 Base URL）：

```bash
export OPENAI_API_KEY="your-key"
msagent config --llm-provider openai --llm-api-key-env OPENAI_API_KEY --llm-base-url "https://api.deepseek.com" --llm-model "deepseek-chat" --llm-max-tokens 0
```

### 🔌 MCP 服务器管理

默认配置会启用 `msprof-mcp`。你也可以手动管理 MCP：

```bash
# 列表
msagent mcp list

# 添加
msagent mcp add --name filesystem --command npx --args "-y,@modelcontextprotocol/server-filesystem,/path"

# 删除
msagent mcp remove --name filesystem
```

### 📁 配置文件位置

- 优先读取当前工作目录：`config.json`
- 若不存在，则读取：`~/.config/msagent/config.json`
- 安全策略：配置文件仅保存 `api_key_env`，不会保存明文 API Key

### 🧠 Skills

msAgent 的内置 Skills 已拆分到独立仓库 [mindstudio-skills](https://github.com/kali20gakki/mindstudio-skills)，在本仓通过 Git Submodule 挂载到根目录 `skills/`。

启动时会自动加载项目根目录 `skills/` 下的技能目录；若当前目录没有可用技能，会回退加载安装包内置技能（如 `op-mfu-calculator`）。格式如下：

```text
skills/
  <skill-name>/
    SKILL.md
```

---

## 🏗️ 编译与打包

### 打包 wheel（可直接 pip install）

```bash
bash scripts/build_whl.sh
```

构建脚本会自动执行 `git submodule update --init --recursive --depth 1 skills`，确保 `mindstudio-skills` 被打入 wheel 包。

打包完成后会在 `dist/` 目录生成 `mindstudio_agent-*.whl`，可直接安装：

```bash
pip install dist/mindstudio_agent-*.whl
```

从 TestPyPI 安装时，建议同时添加 PyPI 作为依赖源（部分依赖仅发布在 PyPI）：

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mindstudio-agent==0.1.0
```

---

## 👨‍💻 开发

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format .
```

---

## 📄 许可证

MIT License
