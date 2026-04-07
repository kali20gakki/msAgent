<h1 align="center">🚀 msAgent</h1>

<p align="center"><strong>面向 Ascend NPU 场景的一站式调试调优 Agent</strong></p>

**msAgent** 聚焦“发现问题 -> 定位根因 -> 给出建议 -> 辅助验证”的调试调优闭环。  
它结合 LLM 推理能力与可扩展工具链，覆盖性能分析、文档走查、工具咨询等场景，帮助你把复杂问题快速转化为可执行的调试与优化决策。

<p align="center">
  <img src="https://github.com/luelueFLY/images/blob/main/img/msagent-hello.gif" alt="msAgent">
</p>

<p align="center">📌 文档导航：<a href="#最新消息">最新消息</a> ｜ <a href="docs/configuration-and-extension.md">配置与扩展</a> ｜ <a href="docs/build-and-package.md">编译与打包</a> ｜ <a href="docs/version-and-compatibility.md">版本与兼容性</a></p>


## 最新消息

- 2026-03-19：`mindstudio-agent` 已发布到 PyPI，推荐优先使用 `pip install -U mindstudio-agent` 安装

## 🤖 内置 Agent 与能力分工

| Agent 形象 | 名称 | 领域定位 | Agent 说明 |
|---|---|---|---|
| <img src="docs/images/Hermes.png" alt="Hermes" width="120"> | **Hermes** | 性能调优：聚焦 Ascend Profiling 分析，覆盖单卡、多卡、集群等场景，擅长快慢卡、慢节点、MFU、通信瓶颈、算子热点、下发调度等性能问题定位与优化建议。 | [查看 Hermes 说明](docs/agents/Hermes.md) |
| <img src="docs/images/Minos.png" alt="Minos" width="120"> | **Minos** | 文档体验与上手审查：聚焦 README 走查、安装流程验证、Quick Start 体验、新手 onboarding 和文档可用性评估，帮助发现文档阻塞点并输出改进建议。 | [查看 Minos 说明](docs/agents/Minos.md) |
---

## ⚡ 快速上手

### 1) 🧰 准备环境

- Python `3.11+`
- 推荐使用 `uv`
- 至少准备一个可用的 LLM API Key
- glibc >= 2.34 (msprof-mcp trace_processor binary required)

### 2) 📦 安装
推荐优先使用 **PyPI 安装**。如果你需要跟踪最新源码、参与开发，或同步最新内置 Skills，再使用 **源码运行** 方式。

说明：
- 下文中的 `msagent` 默认指已安装的命令行入口
- 如果采用源码运行，请将示例中的 `msagent` 替换为 `uv run msagent`

#### 方式一：PyPI 安装

```bash
pip install -U mindstudio-agent
```

#### 方式二：源码运行（开发 / 跟踪最新代码）

拉取源码并进入目录：

```bash
git clone --recurse-submodules https://github.com/kali20gakki/msAgent.git
cd msAgent
git submodule sync --recursive
```

如需同步 `mindstudio-skills` 上游最新版本，再执行：

```bash
git submodule update --init --recursive --remote resources/configs/default/skills
```

如果你只需要使用当前仓库锁定的 Skills 版本，可以跳过这一步。

安装依赖并启动：

```bash
uv sync
uv run msagent
```

#### 常用命令

检查版本：

```bash
msagent --version
```

开启详细日志：

```bash
msagent -v
```

启用后日志会写入当前工作目录下的 `./.msagent/logs/app.log`，同时终端会提示日志文件位置。

#### 日志级别环境变量

通过 `MSAGENT_LOG_LEVEL` 环境变量可调整日志详细程度（默认 `INFO`）：

```bash
# 调试模式，记录最详细日志
export MSAGENT_LOG_LEVEL=DEBUG
msagent -v
```

支持的级别（从低到高）：`DEBUG` < `INFO` < `WARNING` < `ERROR` < `CRITICAL`

### 3) 🔐 配置 LLM

当前 `config` 子命令直接支持的 Provider 是：`openai`、`anthropic`、`google`。

选型建议：
- OpenAI 兼容接口：使用 `openai`
- Anthropic 兼容接口：使用 `anthropic`
- Google / Gemini 接口：使用 `google`

对于用户自部署或经由网关转发的服务，不再单独区分 `custom` provider；请根据接口协议兼容性复用上述三种 provider，并通过 `--llm-base-url` 指定自定义服务地址。

下面命令使用 Linux / macOS 的环境变量写法；Windows CMD 请改成 `set KEY=value`，PowerShell 请改成 `$env:KEY = "value"`。

#### OpenAI 兼容接口

以 DeepSeek `deepseek-chat` 为例：

```bash
export OPENAI_API_KEY="your-key"
msagent config --llm-provider openai --llm-base-url "https://api.deepseek.com/v1" --llm-model "deepseek-chat"
```

#### 本地 OpenAI 兼容服务

如果是本地部署的 OpenAI 兼容服务，例如 **vLLM** 暴露的 OpenAI-compatible API，即使服务端不校验 API Key，也可以继续使用 `openai` provider：

```bash
export OPENAI_API_KEY="dummy"
msagent config --llm-provider openai --llm-base-url "http://127.0.0.1:8000/v1" --llm-model "your-model"
```

- `OPENAI_API_KEY` 只需任意非空字符串，不需要是真实密钥
- `--llm-base-url` 对于 vLLM 一般填写服务根路径，例如 `http://127.0.0.1:8000/v1`

#### 自定义服务地址示例

如果你使用的是自部署服务、企业网关或代理层，请按协议兼容性选择 provider，并通过 `--llm-base-url` 指向你的服务地址。

#### Anthropic 兼容服务

例如自部署或代理后的 Claude / Anthropic 兼容接口：

```bash
export ANTHROPIC_API_KEY="your-key"
msagent config --llm-provider anthropic --llm-base-url "https://example.com/anthropic" --llm-model "claude-sonnet-4-20250514"
```

#### Google / Gemini 服务

例如 Google AI Studio、Vertex AI 网关，或兼容 Gemini 协议的服务：

```bash
export GOOGLE_API_KEY="your-key"
msagent config --llm-provider google --llm-base-url "https://example.com/google" --llm-model "gemini-2.5-pro"
```

说明：

- 自部署服务请选择与其协议兼容的 provider，而不是使用不存在的 `custom` provider
- `--llm-base-url` 用于覆盖默认官方地址，指向你自己的服务入口或代理网关
- 对于不校验 API Key 的兼容服务，通常仍建议设置一个非空占位值，例如 `dummy`
- 历史配置里的 `gemini` 会被兼容处理为 `google`，但新配置建议统一使用 `google`

#### 查看当前配置

```bash
msagent config --show
```

### 4) 🖥️ 启动会话

进入交互式会话：

```bash
msagent
```

手动指定启动 agent：

```bash
msagent --agent Hermes
msagent --agent Minos
```

### 4.1) 🌐 启动 Web UI（可选，Beta功能）

启动：

```bash
msagent web
```

打开：

```text
http://127.0.0.1:3000
```

说明：

- 通过 `pip install -U mindstudio-agent` 安装后，也可以直接使用 `msagent web`
- `msagent web` 默认会同时启动：
  - API：`http://127.0.0.1:2024`
  - UI：`http://127.0.0.1:3000`
- 启动成功后会自动打开默认浏览器进入 UI
- 浏览器访问 UI 地址，不是 API 地址
- 首次打开时会自动预填：
  - `Deployment URL` = `http://127.0.0.1:2024`
  - `Assistant ID` = `msagent`
- 如果端口冲突，改端口启动即可
- 如果本地残留了旧的 Web 进程，可先手动清理：

```bash
pkill -f "next dev --turbopack" || true
pkill -f "langgraph dev" || true
```

常用命令：

```bash
msagent web --host 127.0.0.1 --port 2024 --ui-port 3000
msagent web --port 2025 --ui-port 3001
msagent web --no-open
msagent web --no-ui
```


### 5) 📚 按 Agent 查看说明与示例

不同能力的说明与示例已经按 Agent 拆分：

- `Hermes`：性能调优与 Profiling 分析 Agent 页面，见 [docs/agents/Hermes.md](docs/agents/Hermes.md)
- `Minos`：文档体验与上手审查 Agent 页面，见 [docs/agents/Minos.md](docs/agents/Minos.md)
---

## 🛠️ 参考文档

后续的配置、扩展、构建和版本说明已经拆分到独立文档，避免首页信息过载，也方便按代码演进单独维护：

- [配置与扩展](docs/configuration-and-extension.md)
  项目本地配置目录、MCP 配置、Skills 扩展与加载顺序。
- [编译与打包](docs/build-and-package.md)
  wheel 构建流程、构建脚本行为、常用构建参数与手动构建方式。
- [版本与兼容性](docs/version-and-compatibility.md)
  当前版本、Python 要求、内置 MCP 版本与 Provider 支持情况。

---

## ⌨️ 会话常见操作

进入交互式会话后，可以直接输入问题，也可以配合 `/` 命令和快捷键提升效率。

### `/` 命令

当前交互式会话支持以下 slash commands：

| 命令 | 说明 |
|---|---|
| `/help` | 查看当前支持的命令列表。 |
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

| 快捷键 | 说明 |
|---|---|
| `Ctrl+O` / `Enter` | 展开或折叠当前工具输出。 |
| `Left` / `Right` | 在不同 tool call 之间切换。 |
| `Up` / `Down` | 按行滚动。 |
| `PageUp` / `PageDown` | 按页滚动。 |
| `Home` / `End` | 跳到顶部或底部。 |
| `Esc` / `Ctrl+C` | 关闭查看器。 |

---

## 联系我们

欢迎加入飞书群交流：

<a href="https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=854v5833-c03a-484e-8aac-0637f0303dc4&qr_code=true">
  <img src="https://img.shields.io/badge/Feishu-3370FF?style=for-the-badge&logo=lark&logoColor=white" alt="Feishu Group"></a>

