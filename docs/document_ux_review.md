# 📝 `document-ux-review` 使用说明

`document-ux-review` 是一个面向“文档上手体验审查”的内置 skill，适合用来验证仓库的 `README`、安装文档或 quick start 是否真的能带新用户走通。

## 🤖 Agent 实际会怎么做

当你把仓库地址或本地仓库路径交给 Agent 后，它不会只做静态阅读，而是会按实际 `README` 文档一步步往下执行。

- 先从 `README` 开始，识别其中直接关联的安装、快速开始和运行文档
- 结合你提供的环境信息，优先复用已经准备好的基础环境，而不是重复安装
- 按文档中的真实步骤执行，检查命令、依赖、路径、配置和启动流程是否能走通
- 如果文档存在缺步骤、错误命令、隐含前提或平台差异，Agent 会把这些问题明确记录下来
- 如果为了继续执行不得不额外阅读脚本、源码、CI 或 Dockerfile，这类情况也会被记为文档完整性问题
- 最后输出一份结构化报告，说明哪些步骤成功、哪些步骤阻塞，以及对应的问题位置和改进建议

## 📌 使用前说明

- 通过 `pip install` 安装 `mindstudio-agent` 时，版本需 `>= 0.1.3`
- 如果是源码运行方式，一般可直接使用；建议同步上游 skills 依赖，确保 `resources/configs/default/skills` 内容是最新的

## 🚀 msagent 安装配置

`document-ux-review` 属于内置 skill。大多数情况下，只要 `msagent` 已正确安装、LLM 已配置完成，就可以直接使用，不需要在这里重复维护单独的安装步骤。

具体安装、LLM 配置和启动方式请直接参考 [README](../README.md) 的 `⚡ 快速上手` 章节。

### ✅ 使用前确认

为了确认 `document-ux-review` 已随默认配置启用，建议优先在 TUI 中直接检查：

- 方法一：启动 `msagent` 进入 TUI，查看欢迎页里的 `Skills` 一栏，确认其中包含 `document-ux-review`
- 方法二：在 TUI 中输入 `/skills`，确认当前可用 skill 列表中能看到 `document-ux-review`

## 💬 提示词示例

### 示例 1：`msmonitor` 实战示例

```text
请帮我体验并审查这个仓库的文档上手体验：https://gitcode.com/Ascend/msmonitor 。

本机环境：
- Ubuntu 20.04
- CANN 已安装，环境脚本：`/usr/local/Ascend/ascend-toolkit/set_env.sh`
- conda 虚拟环境已准备好，请优先使用：`msmonitor_ux_review`

请输出详细的中文 HTML 报告到 `/home/msmonitor`，重点说明在上述环境下，新用户能否按文档完成安装并进入可运行状态。
```

## 🖼️ 实际效果

可参考示例产物：[msmonitor-review-run.html](example/msmonitor-review-run.html)
