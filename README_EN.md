<h1 align="center" >MindStudio Agent</h1>

<div align="center">
<p><b><span style="font-size:20px;">One-Stop Debugging and Tuning for Ascend NPU Scenarios</span></b></p>

[![Quick Start](https://badgen.net/badge/Quick%20Start/QuickStart/blue)](docs/en/quick_start/msagent_quick_start.md)
[![AI Skill](https://badgen.net/badge/AI%20Skill/Skill/blue)](./skills/README.md)
[![AI Q&A (DeepWiki)](https://badgen.net/badge/AI%20Q%26A/DeepWiki/blue)](https://deepwiki.com/mindstudio-docs/master/5-msagent-ai-assisted-development-agent)
[![AI Q&A (ZRead)](https://badgen.net/badge/AI%20Q%26A/ZRead/blue)](https://zread.ai/mindstudio-docs/master)
[![Precise Search](https://badgen.net/badge/Exact%20Search/ReadTheDocs/blue)](https://mindstudio-docs-master.readthedocs.io)
[![Ascend Community](https://badgen.net/badge/Ascend%20Community/Community/blue)](https://www.hiascend.com/eng/developer/software/mindstudio)
[![Report Issues](https://badgen.net/badge/Report%20Issues/Issues/blue)](https://atomgit.com/Ascend/msagent/issues)

</div>

English | [简体中文](./README.md)

## ✨ What's New

<span style="font-size:14px;">

🔹 **[Aug 26, 2026]**: Added a one-click installation method (`curl | bash`/`irm | iex`) that uses uv for isolated installation, avoiding dependency conflicts with the existing environment. You are advised to use it. For details, see the [Installation Guide](docs/en/install_guide/msagent_install_guide.md).<br>
🔹 **[Jun 4, 2026]**: `Hermes` has been renamed to `Profiler`, `Zephyr` to `Quantizer`, and `Icarus` to `Operator`.<br>
🔹 **[May 21, 2026]**: v26.0.0 released, adding the `Icarus` agent to cover operator performance tuning scenarios.<br>
🔹 **[Apr 27, 2026]**: v26.0.0.alpha1 released, adding the `Accuracy`/`Zephyr` agents to cover accuracy tuning and model quantization scenarios.<br>
🔹 **[Apr 8, 2026]**: v0.1.3 released, completing the Deep Agents refactoring and enhancing the `Hermes`/`Minos` agents.<br>
🔹 **[Mar 19, 2026]**: `mindstudio-agent` has been released to PyPI. Install it with `pip install mindstudio-agent`.

</span>

## ℹ️ Introduction

MindStudio-Agent (referred to as `msAgent`) is an AI agent workbench for Ascend NPU development, debugging, and tuning scenarios. It combines the CLI, multi-model providers, MCP tools, built-in skills, and domain agents to help users locate issues faster and form actionable suggestions in tasks such as performance tuning, accuracy analysis, model quantization, operator optimization, documentation experience, and code review.

<p align="center">
  <img src="docs/en/figures/best_practices/msagent-hello.gif" alt="msAgent" width="720">
</p>

## ⚙️ Features

| Name | Core Capability |
|---|---|
| [**Profiler**](docs/en/agent_guide/profiler.md) | **[Performance tuning]** Focuses on Ascend Profiling analysis, covering single-device, multi-device, and cluster scenarios. It excels at locating performance issues such as fast/slow devices, slow nodes, MFU, communication bottlenecks, operator hot spots, and launch scheduling, and provides optimization suggestions. |
| [**Accuracy**](docs/en/agent_guide/accuracy.md) | **[Accuracy tuning]** Focuses on Ascend accuracy analysis and optimization, covering common accuracy issues such as RL training-inference consistency analysis and loss/gnorm NaN analysis. |
| [**Quantizer**](docs/en/agent_guide/quantizer.md) | **[Model quantization]** Focuses on msModelSlim quantization and compression scenarios, assisting with model adaptation feasibility, structural risk assessment, and basic adapter development. |
| [**Modeling**](docs/en/agent_guide/modeling.md) | **[Simulation and modeling]** Focuses on simulation and modeling scenarios for large models (LLM/VLM), handling issues such as performance modeling, single-point simulation, throughput planning, device profiling, and model integration preparation. |
| [**Operator**](docs/en/agent_guide/operator.md) | **[Operator tuning]** Focuses on Ascend NPU operator performance tuning, including in-depth operator performance analysis and end-to-end operator performance optimization, helping improve tuning efficiency and reduce development complexity. |
| [**Minos**](docs/en/agent_guide/minos.md) | **[Documentation experience and code review]** Focuses on README walkthrough, installation process verification, Quick Start experience, new-user onboarding, documentation usability assessment, and AtomGit PR review and review comment organization. |
| [**SpecTrainer**](docs/en/agent_guide/spectrainer.md) | **[Speculative decoding resampling]** Focuses on on-policy resampling (response regeneration) of training data for speculative decoding. It regenerates the assistant responses in multi-turn conversations round by round with a verifier model, producing pre-tokenized samples that can be used directly for training. |

> The skill script of **SpecTrainer** orchestrates the process based on the open-source [speculators](https://github.com/vllm-project/speculators) repository **v0.6.0**. For the skill description, see [skills/README.md](./skills/README.md).

## 🚀 Getting Started

For a 10-minute one-stop walkthrough of one-click installation, model connection, and starting msAgent to enter an interactive session, see [msAgent Quick Start](docs/en/quick_start/msagent_quick_start.md).

## 🧩 AI Skill

In addition to domain agents, `msagent` includes a set of reusable skills covering scenarios such as Profiling data analysis, operator performance tuning, accuracy overflow detection, documentation experience review, and code review. For the complete skill list, triggering methods, and dependency descriptions, see [Skills](./skills/README.md).

To integrate skills or `msprof-mcp` into external agents such as Trae, Claude, and Codex, see the [Integration Guide](./docs/en/user_guide/integration-guide.md).

## 📦 Installation Guide

The tool supports one-click installation, pip installation, and source code compilation. For its environment dependencies and installation methods, see the [msAgent Installation Guide](docs/en/install_guide/msagent_install_guide.md).

## 📘 User Guide

For detailed usage instructions, see the [msAgent User Guide](docs/en/user_guide/usemap.md).

## Security Notice

- `msagent` can run shell commands and load third-party MCP Servers and skills. Use it only in trusted environments.
- You are advised to pass credentials through environment variables. HTTP requests initiated by `msagent` itself have TLS certificate verification enabled by default.

## ❓ FAQ

For common issues and troubleshooting entry points, see [FAQ](docs/en/support/faq.md).

## 🌌 Intelligent Search

To help you look up information in the documentation more efficiently, use the following search entry points:<br>
🔹 [AI Q&A (DeepWiki)](https://deepwiki.com/mindstudio-docs/master): Ask questions in natural language to quickly understand the project architecture and module relationships.<br>
🔹 [AI Q&A (ZRead)](https://zread.ai/mindstudio-docs/master): Offers a better Chinese Q&A experience and helps you pinpoint feature usage and details.<br>
🔹 [Precise Search (ReadTheDocs)](https://mindstudio-docs-master.readthedocs.io): Full-text keyword search that takes you directly to interfaces, parameters, error messages, and other information.

## 🛠️ Contribution Guide

You are welcome to submit Issues, PRs, or new domain skills. For the complete process, development self-checks, and various contribution guidelines, see the [Contribution Guide](docs/en/contributing/contributing.md).

## ⚖️ Related Information

🔹 [Release Notes](https://atomgit.com/Ascend/msagent/releases)<br>
🔹 [License Notice](docs/en/legal/license_notice.md)<br>
🔹 [Security Statement](docs/en/legal/SECURITY.md)<br>
🔹 [Disclaimer](docs/en/legal/DISCLAIMER.md)

## 🤝 Suggestions and Communication

You are welcome to contribute to the community. If you have any questions or suggestions, please submit [Issues](https://atomgit.com/Ascend/msagent/issues). We will reply as soon as possible. Thank you for your support.

|                      Live Interaction (WeChat Group)                      |                      Official Updates (Official Account)                      | In-Depth Support (Assistant/Forum)                                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------- |
| <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*Scan the QR code to join the technical discussion group*</sub> | <img src="https://raw.gitcode.com/Ascend/docs/files/master/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*Scan the QR code to follow the official account*</sub> | Scan the QR codes to join the group and follow the official account, providing the fastest way to connect with MindStudio users and developers:<br> **Ask questions quickly:** Discuss technical issues with community members in real time.<br>**Stay up to date:** Get notifications about version releases and feature updates as soon as they are available.<br> **Share experience:** Exchange best practices and hands-on experience with developers.  <br> <br> **More support channels**: 👉 Ascend Assistant: [![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://atomgit.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 Ascend Forum: [![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 Acknowledgments

This tool is jointly contributed by the following departments of Huawei:<br>
🔹 Ascend Computing MindStudio Development Department<br>
🔹 Ascend Computing Ecosystem Enablement Department

Thank you to everyone in the community for every PR. Contributions are welcome!
