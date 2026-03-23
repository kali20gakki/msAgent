<h1 align="center">🚀 msAgent</h1>

<p align="center"><strong>面向 Ascend NPU 场景的性能问题定位助手</strong></p>

**msAgent** 聚焦“发现瓶颈 -> 定位根因 -> 给出建议”的分析闭环。  
它结合 LLM 推理能力与可扩展工具链，帮助你把复杂 Profiling 信息快速转化为可执行的优化决策。

<p align="center">
  <img src="https://github.com/luelueFLY/images/blob/main/img/agent-gif-05.gif" alt="msAgent">
</p>

<p align="center">📌 文档导航：<a href="#最新消息">最新消息</a> ｜ <a href="#版本说明">版本说明</a> ｜ <a href="#使用效果展示">使用效果展示</a></p>


## 最新消息

- 2026-03-19：`mindstudio-agent` 已发布到 PyPI，推荐优先使用 `pip install -U mindstudio-agent` 安装

## 🔍 支持的分析场景与扩展能力

- ⚙️ 覆盖单卡、多卡到集群的性能分析
- 🔎 支持算子热点、通信瓶颈、快慢卡、慢节点、下发调度等常见问题定位
- 📊 支持 MFU 计算、快慢卡诊断等典型分析任务
- 🖼️ 具体示例提示词和效果截图可参考下文的 [使用效果展示](#使用效果展示)
- 🔌 支持 MCP 扩展，默认随 PyPI 包安装启用 [`msprof-mcp`](https://github.com/kali20gakki/msprof-mcp)
- 🧠 支持 Skills 扩展；源码仓库中的内置 Skills 由 [`mindstudio-skills`](https://github.com/kali20gakki/mindstudio-skills) 子模块提供
---

## ⚡ 快速上手

### 1) 🧰 准备环境

- Python `3.11+`
- 推荐使用 `uv`
- 至少准备一个可用的 LLM API Key
- glibc >= 2.34 (msprof-mcp trace_processor binary required)

### 2) 📦 安装
推荐优先使用 **PyPI 安装**。如果你需要跟踪最新源码、参与开发，或同步最新内置 Skills，再使用**源码运行**方式。

说明：
- 下文中的 `msagent` 默认指已安装的命令行入口
- 如果采用源码运行，请将示例中的 `msagent` 替换为 `uv run msagent`

#### 方式一：PyPI 安装（推荐）

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

支持的级别（从低到高）：`DEBUG` < `INFO` < `WARNING` < `ERROR` < `CRITICAL`
```
### 3) 🔐 配置 LLM

当前 `config` 子命令直接支持的 Provider 是：`openai`、`anthropic`、`gemini`、`google`、`custom`。

选型建议：
- OpenAI 兼容接口：使用 `openai`
- 非 OpenAI 兼容、自定义 HTTP 接口：使用 `custom`

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

#### 自定义 HTTP 接口

如果你的服务不是 OpenAI 兼容协议，而是直接请求某个自定义 HTTP 接口，请改用 `custom` provider：

```bash
export CUSTOM_API_KEY="your-key"
msagent config --llm-provider custom --llm-base-url "https://example.com/chat/completions" --llm-model "my-model"
```

如果你的自建接口不需要 API Key，可以不设置 `CUSTOM_API_KEY`，或先将其清空：

```bash
unset CUSTOM_API_KEY
msagent config --llm-provider custom --llm-base-url "http://127.0.0.1:8000/v1/chat/completions" --llm-model "your-model"
```

- `custom` provider 在未设置 `CUSTOM_API_KEY` 时，不会自动附带 `Authorization` 请求头

#### 查看当前配置

```bash
msagent config --show
```

### 4) 🖥️ 启动会话

进入交互式会话：

```bash
msagent
```


### 5) 📊 与 msAgent 一起性能调优


把 Profiling 目录路径和问题一起发给 msAgent，目前已有能力请参考：[使用效果展示](#使用效果展示)


---

## 使用效果展示

| 场景 | 示例提示词                                                           | 效果展示                                                                                                                 |
|---|-----------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| MFU 计算 | `请基于/path/to/kernel_details.csv计算matmul的MFU（910B3），并说明各项计算依据。`        | <img src="https://github.com/luelueFLY/images/blob/main/img/kernel-details-mfu-file.png" alt="MFU 计算示例" width="800"> |
| 快慢卡诊断 | `请分析 /path/to/cluster_profiling/ 中是否存在快慢卡问题，定位异常 rank，并给出可能原因。` | <img src="https://github.com/luelueFLY/images/blob/main/img/slow-rank-detect.png" alt="快慢卡诊断示例" width="800">         |
| profiling数据检查 | `请分析 /path/to/xxx_ascend_pt/ 数据是否采集正常。`                         | <img src="https://github.com/luelueFLY/images/blob/main/img/profiler-data-check.jpg" alt="数据完整性验证示例" width="800">    |
| msprof工具使用类咨询 | `msprof怎么编译出run包？`                                              | <img src="https://github.com/luelueFLY/images/blob/main/img/msprof-build.jpg" alt="工具咨询示例" width="800">              |
| db自定义内容转csv | `基于ascend_pytorch_profiler_0.db，帮我提取各个算子类型的总耗时并按降序输出到csv。`      | <img src="https://github.com/luelueFLY/images/blob/main/img/db-export.png" alt="数据导出示例" width="800">                 |
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
- 如果你是通过 `pip install mindstudio-agent` 或安装已构建的发布包使用，保持默认的 `"command": "msprof-mcp"` 即可
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
| 当前版本 | `0.1.0`（首个 PyPI 发布版本） |
| 发布方式 | 支持通过 `pip install -U mindstudio-agent` 直接安装 |
| 包名 | `mindstudio-agent` |
| 命令行入口 | `msagent` |
| Python 要求 | `>=3.11` |
| 默认内置扩展 | `msprof-mcp==0.1.4` |
| 版本策略 | 遵循语义化版本（SemVer），补丁版本以兼容性修复为主，次版本新增功能保持向后兼容，主版本包含不兼容变更。 |

### `0.1.0` 能力概览

- 支持通过 CLI 直接分析 Profiling 数据，既可进入交互式会话，也可执行单轮问题分析
- 支持单卡、多卡到集群的性能定位，覆盖算子热点、通信瓶颈、快慢卡、慢节点、下发调度等常见问题
- 支持 MFU 计算、快慢卡诊断、Profiling 数据检查等典型任务，具体示例可参考上文的 [使用效果展示](#使用效果展示)
- 支持 OpenAI 兼容 API 与 `custom` HTTP 接口配置，可按项目写入默认模型
- 支持 MCP 扩展，默认随安装启用 `msprof-mcp`
- 支持 Skills 扩展；源码仓库中的内置 Skills 由 [`mindstudio-skills`](https://github.com/kali20gakki/mindstudio-skills) 子模块提供

可通过以下命令查看本地安装版本：

```bash
msagent --version
```

---

## 联系我们

欢迎加入飞书群交流：

<a href="https://applink.feishu.cn/client/chat/chatter/add_by_link?link_token=854v5833-c03a-484e-8aac-0637f0303dc4&qr_code=true">
  <img src="https://img.shields.io/badge/Feishu-3370FF?style=for-the-badge&logo=lark&logoColor=white" alt="Feishu Group"></a>

---


## 🙏 引用与致谢

本项目在架构设计与实现思路上参考了 [`langrepl`](https://github.com/midodimori/langrepl) 项目，在此向其作者与贡献者表示感谢。
