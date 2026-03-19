<h1 align="center">🚀 msAgent</h1>

<p align="center"><strong>面向 Ascend NPU 场景的性能问题定位助手</strong></p>

**msAgent** 聚焦“发现瓶颈 -> 定位根因 -> 给出建议”的分析闭环。  
它结合 LLM 推理能力与可扩展工具链，帮助你把复杂 Profiling 信息快速转化为可执行的优化决策。

<p align="center">
  <img src="https://github.com/luelueFLY/images/blob/main/img/agent-gif-05.gif" alt="msAgent">
</p>

📌 文档导航：[最新消息](#最新消息) ｜ [版本说明](#版本说明) ｜ [使用效果展示](#使用效果展示)


## 最新消息

- 2026-03-11：v0.1 PyPi whl包待发布

## 🔍 支持的分析场景与扩展能力

- ⚙️ 单卡性能问题：高耗时算子、计算热点、重叠度不足等
- 🔗 多卡性能问题：快慢卡差异、通信效率瓶颈、同步等待等
- ⏱️ 下发与调度问题：下发延迟、CPU 侧调度阻塞等
- 🧩 集群性能问题：慢节点识别与从全局到单机的逐层定位
- 🔌 MCP 扩展：基于 Model Context Protocol 接入工具（默认随 wheel 安装启用 `msprof-mcp==0.1.4`）
- 🧠 Skills 扩展：自动加载 `<working-dir>/skills` 与内置 Skills；源码仓库中的内置 Skills 由 `resources/configs/default/skills/` 下的 `mindstudio-skills` 子模块提供，复用领域分析流程和知识（仓库：[mindstudio-skills](https://github.com/kali20gakki/mindstudio-skills)）
---

## ⚡ 快速上手

### 1) 🧰 准备环境

- Python `3.11+`
- 推荐使用 `uv`
- 至少准备一个可用的 LLM API Key

说明：
- 下文中的 `msagent` 适用于已安装命令行入口的场景。
- 如果你是源码运行（`git clone` + `uv sync`），请把示例里的 `msagent` 替换成 `uv run msagent`。
- Windows 示例默认使用 CMD；若使用 PowerShell，请把 `set KEY=value` 改为 `$env:KEY = "value"`。

### 2) 📦 安装

源码运行方式：

```bash
git clone --recurse-submodules https://github.com/kali20gakki/msAgent.git
cd msAgent
git submodule sync --recursive
# 如果要同步 mindstudio-skills 上游最新版本，再执行下一行
git submodule update --init --recursive --remote resources/configs/default/skills
uv sync
```

说明：

- `git clone --recurse-submodules` 会拉取主仓库当前记录的 Skills 版本。
- 如果你只需要初始化到主仓库锁定的 Skills 版本，可执行：

  ```bash
  git submodule update --init --recursive resources/configs/default/skills
  ```

- 如果你想同步 `mindstudio-skills` 仓库的最新提交，请执行：

  ```bash
  git submodule sync --recursive
  git submodule update --init --recursive --remote resources/configs/default/skills
  ```

如果你已经拿到 wheel 或已经发布到包源，也可以直接安装：

```bash
pip install -U mindstudio-agent
```

检查版本：

```bash
msagent --version
```

### 3) 🔐 配置默认 LLM

当前 `config` 子命令直接支持的 Provider 是：`openai`、`anthropic`、`gemini`、`google`、`custom`。

OpenAI 示例：

Linux / macOS：

```bash
export OPENAI_API_KEY="your-key"
msagent config --llm-provider openai --llm-model "gpt-5-mini-2025-08-07"
```

Windows（CMD）：

```cmd
set OPENAI_API_KEY=your-key
msagent config --llm-provider openai --llm-model "gpt-5-mini-2025-08-07"
```

Custom 示例：

Linux / macOS：

```bash
export CUSTOM_API_KEY="your-key"
msagent config --llm-provider custom --llm-base-url "https://example.com/chat/completions" --llm-model "my-model"
```

Windows（CMD）：

```cmd
set CUSTOM_API_KEY=your-key
msagent config --llm-provider custom --llm-base-url "https://example.com/chat/completions" --llm-model "my-model"
```



查看当前项目配置：

```bash
msagent config --show
```

### 4) 🖥️ 启动会话

进入交互式会话：

```bash
msagent
```


### 5) 📊 与 msAgent 一起性能调优

把 Profiling 目录路径和问题一起发给 msAgent，例如：

Linux / macOS：

```text
请分析 /path/to/profiler_output 的性能瓶颈，重点关注通信效率、重叠度和高耗时算子。
```

Windows：

```text
请分析 C:\path\to\profiler_output 的性能瓶颈，重点关注通信效率、重叠度和高耗时算子。
```

---

## 使用效果展示

| 场景     | 效果展示 |
|--------|---|
| MFU 计算 | <img src="https://github.com/luelueFLY/images/blob/main/img/op-mfu-calculate.png" alt="MFU 计算示例" width="800"> |
| 快慢卡诊断  | <img src="https://github.com/luelueFLY/images/blob/main/img/slow-rank-detect.png" alt="快慢卡诊断示例" width="800"> |

---



## 📚 常用命令

| 命令 | 说明 |
|---|---|
| `msagent` | 启动交互式会话 |
| `msagent "..."` | 发送单条消息并输出结果 |
| `msagent -r` | 恢复当前工作目录最近一次线程 |
| `msagent -m gemini-2.5-pro` | 以指定模型别名启动当前会话 |
| `msagent -a code-reviewer` | 以指定 Agent 启动当前会话 |
| `msagent --no-stream "..."` | 关闭流式输出 |
| `msagent config --show` | 查看当前项目配置 |
| `msagent config --llm-provider openai --llm-model "gpt-5-mini-2025-08-07"` | 把默认模型写入当前项目配置 |

---

## 🗂️ 完整命令参考

### 默认会话入口

当前 CLI 的“默认命令”就是聊天会话本身。也就是说，直接执行 `msagent` 会进入会话模式，根级 `--help` 只会显示显式暴露的 `config` 子命令。

会话入口实际支持的参数如下：

```bash
msagent [message] [--stream | --no-stream] [-w DIR] [-a AGENT] [-m MODEL_ALIAS] [-r] [--timer] [-am {semi-active,active,aggressive}] [-v]
```

| 参数 | 说明 |
|---|---|
| `message` | 可选位置参数；省略时进入交互式会话，传入时执行单轮请求 |
| `-w`, `--working-dir` | 指定工作目录；`msAgent` 会从该目录下的 `.msagent/` 读取或初始化配置 |
| `-a`, `--agent` | 指定 Agent 名称，例如 `general`、`code-reviewer` |
| `-m`, `--model` | 指定模型别名，而不是 Provider 原始模型名；别名来自 `.msagent/config.llms.yml` 和 `.msagent/llms/*.yml` |
| `-r`, `--resume` | 恢复最近一个线程；若同时传入 `message`，会在恢复后的线程上继续执行 |
| `--stream` | 流式输出回答，默认开启 |
| `--no-stream` | 关闭流式输出，等待完整回答后再输出 |
| `--timer` | 打印启动阶段耗时，便于排查初始化慢的问题 |
| `-am`, `--approval-mode` | 工具审批模式，可选 `semi-active`、`active`、`aggressive` |
| `-v`, `--verbose` | 开启控制台和 `.msagent/logs/app.log` 调试日志输出 |

审批模式说明：

| 模式 | 说明 |
|---|---|
| `semi-active` | 正常遵循审批规则 |
| `active` | 默认模式；跳过绝大多数审批，仅保留 `always_deny` |
| `aggressive` | 跳过所有审批规则 |

示例：

```bash
msagent -m gemini-2.5-pro
msagent -a code-reviewer
msagent -r "继续上一次分析，补充通信瓶颈优化建议"
msagent --no-stream "总结这个 Profiling 的主要瓶颈"
```

### `config` 子命令

`config` 用于查看和更新当前项目的默认模型配置：

```bash
msagent config [--show] [--llm-provider PROVIDER] [--llm-api-key KEY] [--llm-api-key-env ENV] [--llm-max-tokens N] [--llm-base-url URL] [--llm-model MODEL] [-w DIR] [-v]
```

| 参数 | 说明 |
|---|---|
| `--show`, `-s` | 显示当前项目配置；如果未传任何更新参数，也会自动执行展示 |
| `--llm-provider` | 只接受 `openai`、`anthropic`、`gemini`、`google`、`custom` |
| `--llm-model`, `-m` | 设置默认 Provider 对应的原始模型名，例如 `gpt-5-mini-2025-08-07` |
| `--llm-base-url` | 设置 Provider 地址；对于 `custom` 应填写完整请求 URL |
| `--llm-max-tokens` | 设置最大输出 token；`0` 表示交给模型或服务端默认值 |
| `--llm-api-key-env` | 设置读取 API Key 的环境变量名 |
| `--llm-api-key` | 只给当前进程临时注入 API Key，不会明文写入配置文件 |
| `-w`, `--working-dir` | 指定要修改的项目目录 |
| `-v`, `--verbose` | 输出更详细日志 |

补充说明：

- `gemini` 会被映射到内部 Provider `google`，默认读取的环境变量是 `GOOGLE_API_KEY`。
- `custom` 会直接向你配置的 URL 发送普通 HTTP 请求；不是强依赖 OpenAI SDK 的兼容模式。
- `custom` 默认读取 `CUSTOM_API_KEY`；如果配置了 `--llm-api-key-env`，则优先读取该环境变量。
- `custom` 只有在 API Key 非空时才会附带 `Authorization: Bearer <key>` 请求头；如果你的接口无需鉴权，可以不填 Key。
- `config` 会把结果写入 `<working-dir>/.msagent/config.llms.yml`，并让默认 Agent 指向其中的 `default` 模型别名。
- 仓库里还内置了 `deepseek`、`zhipuai`、`ollama`、`lmstudio`、`bedrock` 等模型别名，但它们不是通过 `config --llm-provider` 暴露的；更适合用 `-m <alias>`、`/model` 或手动编辑 `.msagent/llms/*.yml`。

示例：

```bash
msagent config --show
msagent config --llm-provider anthropic --llm-model "claude-sonnet-4-5"
msagent config --llm-provider openai --llm-base-url "https://api.deepseek.com" --llm-model "deepseek-chat" --llm-max-tokens 0
msagent config --llm-provider custom --llm-base-url "https://example.com/chat/completions" --llm-model "my-model"
msagent config --llm-provider custom --llm-api-key-env MY_CUSTOM_TOKEN --llm-base-url "https://example.com/chat/completions" --llm-model "my-model"
```

---

## 💬 会话内命令与快捷键

### Slash 命令

以下命令在交互式会话中可用：

| 命令 | 说明 |
|---|---|
| `/help` | 显示会话命令帮助 |
| `/hotkeys` | 显示当前输入框快捷键 |
| `/agents` | 交互式切换 Agent，并把选中 Agent 写回默认配置 |
| `/model` | 交互式切换当前 Agent 或子 Agent 的模型别名 |
| `/tools` | 浏览当前 Agent 已加载的工具 |
| `/skills` | 浏览当前 Agent 已加载的 Skills |
| `/mcp` | 交互式启用或禁用当前已配置的 MCP 服务器 |
| `/memory` | 打开 `.msagent/memory.md` 进行编辑 |
| `/graph` | 在终端渲染当前 Agent Graph |
| `/graph --browser` | 生成图像并尝试在浏览器中打开 |
| `/clear` | 清屏并切换到一个新的线程 ID |
| `/resume` | 从历史线程列表中选择并恢复 |
| `/replay` | 从当前线程某条历史用户消息处重放 |
| `/compress` | 压缩当前上下文到新线程 |
| `/todo` | 查看当前线程的 todo 列表 |
| `/todo 20` | 最多显示 20 条 todo |
| `/approve` | 管理 `always_allow` / `always_ask` / `always_deny` 规则 |
| `/exit` | 退出当前会话 |

说明：

- 当前代码里没有 `/new`、`/backend`、`/shell` 这些旧命令。
- `/mcp` 只负责切换已存在 MCP 服务的启用状态；新增或删除服务需要直接编辑配置文件。
- `/memory` 和 `/approve` 都会使用 `CLI__EDITOR` 指定的编辑器，默认是 `vim`。

### 输入框快捷键

| 快捷键 | 说明 |
|---|---|
| `Ctrl+C` | 若输入框有内容则清空；若输入框为空，连续按两次退出 |
| `Ctrl+J` | 插入换行 |
| `Shift+Tab` | 切换审批模式 |
| `Ctrl+B` | 切换 Bash 模式 |
| `Ctrl+K` | 打开快捷键面板 |
| `Tab` | 立即应用第一个补全结果 |
| `Enter` | 提交消息；若当前选中补全项则先应用补全 |

补充说明：

- Slash 命令和 `@文件` / `@图片` 引用都支持补全。
- 在 Agent、Model、Thread、Tool、Skill 等选择面板里，通常使用 `Up` / `Down` / `Enter` 操作。

### Bash 模式说明

按 `Ctrl+B` 后，输入框会切换到 Bash 模式。此时你输入的内容会直接通过：

```bash
bash -c "<your-command>"
```

在当前 `working_dir` 下执行，并把 `stdout` / `stderr` 原样打印出来。

注意：

- 这里执行的是本机真实 Shell 命令，不是 LLM 工具调用结果。
- 当前实现没有额外沙箱隔离，风险和你直接在终端执行命令一致。
- 该模式依赖系统存在 `bash`，更适合 Linux、macOS、WSL 或 Git Bash 环境。

---

## 🌱 环境变量参考

### LLM 鉴权与连接

| 环境变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | OpenAI Provider API Key |
| `ANTHROPIC_API_KEY` | Anthropic Provider API Key |
| `GOOGLE_API_KEY` | Google / Gemini Provider API Key |
| `CUSTOM_API_KEY` | `custom` Provider 的 API Key；非空时会作为 `Authorization: Bearer <key>` 发送 |
| `DEEPSEEK_API_KEY` | DeepSeek 内置别名所使用的 API Key |
| `ZHIPUAI_API_KEY` | 智谱内置别名所使用的 API Key |
| `AWS_ACCESS_KEY_ID` | Bedrock 鉴权 |
| `AWS_SECRET_ACCESS_KEY` | Bedrock 鉴权 |
| `AWS_SESSION_TOKEN` | Bedrock 临时凭证 |
| `CUSTOM_BASE_URL` | `custom` Provider 的默认请求 URL，建议填写完整接口地址 |
| `HTTP_PROXY` / `http_proxy` | HTTP 代理 |
| `HTTPS_PROXY` / `https_proxy` | HTTPS 代理 |

说明：

- `msagent config --llm-provider gemini` 实际读取的是 `GOOGLE_API_KEY`。
- `msagent config --llm-provider custom` 默认读取的是 `CUSTOM_API_KEY`；如果接口不需要鉴权，也可以不设置。
- `--llm-api-key` 只会注入当前进程，不会保存到配置文件。
- 配置文件不会明文保存 API Key。

### 设置项覆盖

`msAgent` 使用 `pydantic-settings` 读取环境变量，并支持 `.env` 文件。当前代码里没有强制 `MSAGENT_` 前缀，嵌套字段使用 `__` 分隔。

常用覆盖项：

| 环境变量 | 说明 |
|---|---|
| `CLI__EDITOR` | 设置 `/memory`、`/approve` 使用的编辑器 |
| `CLI__PROMPT_STYLE` | 修改输入提示符样式 |
| `CLI__ENABLE_WORD_WRAP` | 是否启用自动换行 |
| `CLI__MAX_AUTOCOMPLETE_SUGGESTIONS` | 设置补全候选上限 |
| `LLM__OLLAMA_BASE_URL` | 设置 Ollama 地址 |
| `LLM__LMSTUDIO_BASE_URL` | 设置 LM Studio OpenAI 兼容地址 |
| `LOG_LEVEL` | 设置日志级别 |
| `SUPPRESS_GRPC_WARNINGS` | 是否屏蔽 gRPC 警告，默认 `true` |

示例：

```bash
CLI__EDITOR=nvim
CLI__PROMPT_STYLE=">> "
CLI__MAX_AUTOCOMPLETE_SUGGESTIONS=20
LLM__OLLAMA_BASE_URL=http://localhost:11434
LOG_LEVEL=DEBUG
```

---

## 🛠️ 参考：配置与扩展

### 📁 项目本地配置目录

当前实现使用“项目本地配置”，所有运行时文件都放在：

```text
<working-dir>/.msagent/
```

首次运行时，`msAgent` 会把 `resources/configs/default/` 里的默认模板复制到该目录。常见文件如下：

| 文件 | 作用 |
|---|---|
| `.msagent/config.llms.yml` | 当前项目默认模型配置；`msagent config` 直接写这里 |
| `.msagent/llms/*.yml` | 附带的模型别名集合 |
| `.msagent/agents/*.yml` | Agent 定义，例如 `general`、`code-reviewer` |
| `.msagent/subagents/*.yml` | SubAgent 定义 |
| `.msagent/checkpointers/*.yml` | Checkpointer 配置 |
| `.msagent/sandboxes/*.yml` | 沙箱配置模板 |
| `.msagent/config.mcp.json` | MCP 服务器配置 |
| `.msagent/config.approval.json` | 工具审批规则 |
| `.msagent/config.checkpoints.db` | 会话 checkpoint 数据库 |
| `.msagent/.history` | 输入历史 |
| `.msagent/memory.md` | 用户偏好和项目上下文记忆 |

### 🤖 模型别名与 Agent

除了 `config` 写入的 `default` 别名外，仓库还自带一批可直接使用的模型别名，来源于 `.msagent/llms/*.yml`。示例包括：

- OpenAI: `gpt-5-mini-thinking`
- Anthropic: `sonnet-4.5`、`haiku-4.5`、`haiku-4.5-thinking`
- Google: `gemini-3-pro`、`gemini-2.5-pro`、`gemini-2.5-pro-thinking`
- DeepSeek: `deepseek-chat`
- ZhipuAI: `glm-4.6`、`glm-4.6-thinking`
- Bedrock / Ollama / LM Studio: 也有默认别名模板

使用方式：

- 临时指定会话模型：`msagent -m gemini-2.5-pro`
- 在会话中持久切换 Agent / SubAgent 模型：`/model`
- 临时指定 Agent：`msagent -a code-reviewer`
- 在会话中切换默认 Agent：`/agents`

默认模板里内置了至少两个 Agent：

- `general`
- `code-reviewer`

### 🔌 MCP 配置

默认模板会启用 `msprof-mcp`，并直接调用随当前 Python 环境安装的 `msprof-mcp` 可执行程序启动。

当前代码中的 MCP 使用方式是：

- 用 `/mcp` 在会话里切换已有服务的启用状态
- 用编辑器直接修改 `.msagent/config.mcp.json` 来新增、删除或细调服务器定义

配置文件格式示例：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"],
      "transport": "stdio",
      "env": {},
      "include": [],
      "exclude": [],
      "enabled": true,
      "stateful": false
    }
  }
}
```

常用字段：

- `command` / `url`
- `args`
- `transport`
- `env`
- `include` / `exclude`
- `enabled`
- `stateful`
- `repair_command` / `repair_timeout`
- `timeout` / `sse_read_timeout` / `invoke_timeout`

对于像 `msprof-mcp` 这类本地 `stdio` MCP，默认更推荐：

- `stateful: true`，避免每次工具调用都重新拉起服务进程
- 如果你是通过 `pip install mindstudio-agent` 或安装 wheel 使用，保持默认的 `"command": "msprof-mcp"` 即可
- 如需把 `msprof-mcp` 与当前环境解耦，仍可改成 `uvx --isolated --from msprof-mcp==0.1.4 msprof-mcp` 启动
- 只在需要强制刷新远端版本时才临时使用 `uvx --refresh`，不要把它作为常驻默认参数

### 🧠 Skills

Skills 会按以下候选目录自动加载：

- `<working-dir>/skills`
- 内置 Skills（源码仓库中对应 `resources/configs/default/skills/` 子模块）
- `<working-dir>/.msagent/skills`

源码仓库中的内置 Skills 来源于 `mindstudio-skills` 子模块：

- 子模块路径：`resources/configs/default/skills/`
- 上游仓库：`https://github.com/kali20gakki/mindstudio-skills`

如果你是 `git clone` 后从源码运行，建议至少执行一次以下命令初始化 Skills：

```bash
git submodule sync --recursive
git submodule update --init --recursive resources/configs/default/skills
```

如果你要同步 `mindstudio-skills` 的最新上游提交，请执行：

```bash
git submodule sync --recursive
git submodule update --init --recursive --remote resources/configs/default/skills
```

说明：

- 不带 `--remote`：同步到当前 `msAgent` 仓库记录的 Skills 版本，适合复现和保持版本一致。
- 带 `--remote`：同步到 `mindstudio-skills` 上游默认分支的最新提交。
- 执行 `--remote` 后，主仓库里的 submodule 指针会变更；如果你希望固定这个版本，记得一起提交该变更。

支持两种目录结构：

```text
skills/
  my-skill/
    SKILL.md
```

```text
skills/
  profiling/
    my-skill/
      SKILL.md
```

其中 `SKILL.md` 需要包含 frontmatter，至少提供：

```yaml
---
name: my-skill
description: 这个技能做什么
---
```

当前仓库里已经包含示例技能 `op-mfu-calculator`，会在无项目自定义 Skill 时作为兜底能力之一被加载。

---

## 🏗️ 编译与打包

### 打包 wheel（可直接 pip install）

Linux / macOS：

```bash
bash scripts/build_whl.sh
```

Windows（CMD）：

```cmd
git submodule sync --recursive
git submodule update --init --recursive --remote --force --depth 1 resources/configs/default/skills
uv build --wheel --out-dir dist .
```

如果你的 Windows 环境安装了 Git Bash / WSL，也可以直接执行 `bash scripts/build_whl.sh`。

构建脚本会默认执行 `git submodule update --init --recursive --remote --force --depth 1 resources/configs/default/skills`，同步 `mindstudio-skills` 上游最新提交后再打入 wheel 包。
如果你需要按主仓库里固定的 submodule 提交构建，可以临时设置 `SYNC_SKILLS_REMOTE=0`。

打包完成后会在 `dist/` 目录生成 `mindstudio_agent-*.whl`，可直接安装：

Linux / macOS：

```bash
pip install dist/mindstudio_agent-<version>-py3-none-any.whl
```

Windows（CMD）：

```cmd
pip install .\dist\mindstudio_agent-<version>-py3-none-any.whl
```

请将上面的 `<version>` 替换为实际构建出的 wheel 文件名。

从 TestPyPI 安装时，建议同时添加 PyPI 作为依赖源（部分依赖仅发布在 PyPI）：

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mindstudio-agent==0.1.0
```

---

---

## 版本说明

| 项目 | 说明 |
|---|---|
| 当前版本 | `0.1.0` |
| 包名 | `mindstudio-agent` |
| 命令行入口 | `msagent` |
| Python 要求 | `>=3.11` |
| 版本策略 | 遵循语义化版本（SemVer），补丁版本以兼容性修复为主，次版本新增功能保持向后兼容，主版本包含不兼容变更。 |

可通过以下命令查看本地安装版本：

```bash
msagent --version
```

---


## 🙏 引用与致谢

本项目在架构设计与实现思路上参考了 [`langrepl`](https://github.com/midodimori/langrepl) 项目，在此向其作者与贡献者表示感谢。
