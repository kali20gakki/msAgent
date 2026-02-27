# msAgent

**msAgent** 是一个面向Ascend NPU性能调优的agent专家。  
它结合 LLM 分析能力与可扩展工具链，将复杂性能数据转化为清晰的瓶颈结论和可执行优化方向。

<p align="center">
  <img src="docs/img/msagent.gif" alt="msAgent">
</p>

## 支持的分析场景与扩展能力

- 单卡性能问题：高耗时算子、计算热点、重叠度不足等
- 多卡性能问题：快慢卡差异、通信效率瓶颈、同步等待等
- 下发与调度问题：下发延迟、CPU 侧调度阻塞等
- 集群性能问题：慢节点识别与从全局到单机的逐层定位
- MCP 扩展：基于 Model Context Protocol 接入工具（默认启用 `msprof-mcp`）
- Skills 扩展：自动加载 `skills/` 目录技能，复用领域分析流程和知识

---

## 快速上手

### 1) 准备环境

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- 可用的 LLM API Key（OpenAI / Anthropic / Gemini / 兼容 OpenAI 接口）

### 2) 安装依赖

```bash
git clone https://github.com/kali20gakki/msAgent.git
cd msAgent
uv sync
```

### 3) 配置 LLM（必做）

推荐先用 OpenAI：

```bash
uv run msagent config --llm-provider openai --llm-api-key "your-key" --llm-model "gpt-4o-mini"
```

检查配置是否生效：

```bash
uv run msagent config --show
```

### 4) 启动 TUI

```bash
uv run msagent chat --tui
```

### 5) 性能分析

把 Profiling 目录路径和你的问题一起发给 msAgent，例如：

```text
请分析 /path/to/profiler_output 的性能瓶颈，重点关注通信和高耗时算子。
```

---

## 常用命令

| 命令 | 说明 |
|---|---|
| `uv run msagent chat --tui` | 启动 TUI 交互 |
| `uv run msagent chat` | 启动 CLI 交互 |
| `uv run msagent ask "..."` | 单轮提问 |
| `uv run msagent config --show` | 查看当前配置 |
| `uv run msagent mcp list` | 查看 MCP 服务器 |
| `uv run msagent info` | 查看工具信息 |

---

## 参考：配置与扩展

### LLM 配置示例

Anthropic:

```bash
uv run msagent config --llm-provider anthropic --llm-api-key "your-key" --llm-model "claude-3-5-sonnet-20241022"
```

Gemini:

```bash
uv run msagent config --llm-provider gemini --llm-api-key "your-key" --llm-model "gemini-2.0-flash"
```

自定义 OpenAI 兼容接口：

```bash
uv run msagent config --llm-provider custom --llm-api-key "your-key" --llm-base-url "http://127.0.0.1:8045/v1" --llm-model "your-model-name"
```

### MCP 服务器管理

默认配置会启用 `msprof-mcp`。你也可以手动管理 MCP：

```bash
# 列表
uv run msagent mcp list

# 添加
uv run msagent mcp add --name filesystem --command npx --args "-y,@modelcontextprotocol/server-filesystem,/path"

# 删除
uv run msagent mcp remove --name filesystem
```

### 配置文件位置

- 优先读取当前工作目录：`config.json`
- 若不存在，则读取：`~/.config/msagent/config.json`

### Skills

msAgent 启动时会自动加载项目根目录 `skills/` 下的技能目录，格式如下：

```text
skills/
  <skill-name>/
    SKILL.md
```

---

## 开发

```bash
uv sync --dev
uv run pytest
uv run ruff check .
uv run ruff format .
```

---

## 许可证

MIT License
