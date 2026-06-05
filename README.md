<h1 align="center" >MindStudio Agent</h1>

<div align="center">
<p><b><span style="font-size:24px;">面向 Ascend NPU 场景的一站式调试调优 Agent</span></b></p>

[![快速入门](https://badgen.net/badge/快速入门/QuickStart/blue)](#-快速入门)
[![精确搜索](https://badgen.net/badge/精确搜索/ReadTheDocs/blue)](https://mindstudio-agent.readthedocs.io/zh-cn/latest/)
[![安装指南](https://badgen.net/badge/安装指南/Install/blue)](#-安装指南)
[![配置文档](https://badgen.net/badge/配置文档/Docs/blue)](docs/zh/user_guide/configuration-and-extension.md)
[![昇腾社区](https://badgen.net/badge/昇腾社区/Community/blue)](https://www.hiascend.com/cn/developer/software/mindstudio)
[![报告问题](https://badgen.net/badge/报告问题/Issues/blue)](https://gitcode.com/Ascend/msagent/issues)

</div>

## ✨ 最新消息

<span style="font-size:14px;">

🔹 **[2026.06.04]**：`Hermes` 更名为 `Profiler`，`Zephyr` 更名为 `Quantizer`，`Icarus` 更名为 `Operator`。<br>
🔹 **[2026.05.21]**：`v26.0.0已发布，新增Icarus Agent，覆盖算子性能调优场景`。<br>
🔹 **[2026.04.27]**：`v26.0.0.alpha1` 发布，新增 `Accuracy` / `Zephyr` Agent，覆盖精度调优与模型量化场景。<br>
🔹 **[2026.04.08]**：`v0.1.3` 发布，完成 DeepAgents 重构并增强 `Hermes` / `Minos` Agent。<br>
🔹 **[2026.03.19]**：`mindstudio-agent` 已发布到 PyPI，推荐使用 `pip install -U mindstudio-agent` 安装。

</span>

## ℹ️ 简介

MindStudio-Agent（简称 `msagent`）是面向昇腾 Ascend NPU 开发、调试和调优场景的 AI Agent 工作台。它将 CLI、多模型 Provider、MCP 工具、内置 Skills 与领域 Agent 组合在一起，帮助用户在性能调优、精度分析、模型量化、算子优化、文档体验与代码审查等任务中更快定位问题并形成可执行建议。

<p align="center">
  <img src="docs/images/msagent-hello.gif" alt="msAgent" width="720">
</p>

## ⚙️ 功能介绍

| 名称 |  核心能力 |
|---|---|
| [**Profiler**](docs/zh/agent_guide/Profiler.md) | **【性能调优】**  聚焦 Ascend Profiling 分析，覆盖单卡、多卡、集群等场景，擅长快慢卡、慢节点、MFU、通信瓶颈、算子热点、下发调度等性能问题定位与优化建议。 |
| [**Accuracy**](docs/zh/agent_guide/Accuracy.md) | **【精度调优】** 聚焦 Ascend 精度分析与优化，覆盖 RL 训推一致性分析、loss / gnorm NaN 分析等常见精度问题。 |
| [**Quantizer**](docs/zh/agent_guide/Quantizer.md) | **【模型量化】**  聚焦 msModelSlim 量化与压缩场景，协助完成模型适配可行性、结构风险评估与基础适配器开发。 |
| [**Operator**](docs/zh/agent_guide/Operator.md) | **【算子调优】**  聚焦 Ascend NPU 算子性能调优，包括算子性能深度分析、端到端算子性能优化，辅助提升调优效率并降低开发难度。 |
| [**Minos**](docs/zh/agent_guide/Minos.md) | **【文档体验与代码审查】**  聚焦 README 走查、安装流程验证、Quick Start 体验、新手 onboarding、文档可用性评估，以及 GitCode PR 审查与评审意见整理。 |

## 🧩 Skills

除领域 Agent 外，`msagent` 还内置了一批可复用的 Skills，覆盖 Profiling 数据分析、算子性能调优、精度溢出检测、文档体验审查、代码审查等场景。完整技能清单、触发方式与依赖说明，请参见 《[Skills](./skills/README.md)》。

## 🚀 快速入门

快速体验核心功能，请参见《[msAgent快速入门](docs/zh/getting_started/quick_start.md)》。

## 📦 安装指南

介绍工具的环境依赖与安装方法，请参见《[msAgent安装指南](docs/zh/getting_started/install_guide.md)》。

## 📘 使用指南

工具的详细使用方法，请参见《[msAgent使用指南](docs/zh/user_guide/usemap.md)》。

## ❓ FAQ

常见问题与排查入口请参见 《[FAQ](docs/zh/user_guide/faq.md)》。

## 🌌 智能检索

为提升文档查阅效率，建议优先通过以下入口定位信息：

🔹 《[中文文档首页](docs/index.md)》：按快速入门、用户指南、Agent 指南和开发指南组织内容。<br>
🔹 《[配置与扩展](docs/zh/user_guide/configuration-and-extension.md)》：查询本地配置目录、MCP 配置、Skills 扩展与加载顺序。<br>
🔹 《[版本与兼容性](docs/zh/developer_guide/version-and-compatibility.md)》：查询版本要求、兼容策略与内置依赖。<br>
🔹 会话内直接询问 `msagent`：让对应 Agent 结合仓库文档、配置和上下文辅助定位问题。<br>

## 🛠️ 贡献指南

欢迎提交 Issue、PR 或补充新的领域 Skills。完整流程、开发自检与各类贡献指引见 《[贡献指南](docs/zh/developer_guide/contributing.md)》。

## ⚖️ 相关说明

🔹 《[版本与兼容性](docs/zh/developer_guide/version-and-compatibility.md)》<br>
🔹 《[安全声明](docs/zh/legal/SECURITY.md)》<br>
🔹 《[免责声明](docs/zh/legal/DISCLAIMER.md)》<br>
🔹 《[许可证声明](docs/zh/legal/LICENSE_intro.md)》<br>

## 🤝 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交 [Issues](https://gitcode.com/Ascend/mssanitizer/issues)，我们会尽快回复。感谢您的支持。

|                      即时互动（微信群）                      |                      官方资讯（公众号）                      | 深度支持（助手/论坛）                                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------- |
| <img src="https://raw.gitcode.com/mengguangxin/docs/files/dev_0526/common/Writing_Template/figures/qr_code_wechat_work.png" width="120"><br><sub>*扫码加入技术交流群*</sub> | <img src="https://raw.gitcode.com/mengguangxin/docs/files/dev_0526/common/Writing_Template/figures/qr_code_wechat_official_account.png" width="120"><br><sub>*扫码关注官方公众号*</sub> | 扫码入群并关注公众号，直达 MindStudio 用户与开发者最快捷的交流平台：<br> **快速提问：** 与社区小伙伴即时探讨技术问题<br>**掌握动态：** 第一时间获取版本发布与功能更新通知<br> **经验共享：** 与广大开发者交流最佳实践与实战心得  <br> <br> **更多支持渠道**：👉 昇腾助手：[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=flat-square&logo=wechat&logoColor=white)](https://gitcode.com/Ascend/msit/blob/master/docs/zh/figures/readme/xiaozhushou.png) 👉 昇腾论坛：[![Website](https://img.shields.io/badge/Website-%231e37ff?style=flat-square&logo=RSS&logoColor=white)](https://www.hiascend.com/forum/) |

## 🙏 致谢

本工具由华为公司下列部门联合贡献：<br>
🔹 昇腾计算 MindStudio 开发部<br>
🔹 昇腾计算生态使能部

感谢来自社区的每一个 PR，欢迎贡献！
