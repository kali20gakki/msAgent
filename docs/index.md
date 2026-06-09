# MindStudio-Agent 文档

MindStudio-Agent 是面向 Ascend NPU 场景的一站式调试调优 Agent。中文文档统一维护在 `docs/zh/` 下，按新用户上手、日常使用、Agent 能力和开发维护四类组织。

```{toctree}
:maxdepth: 2
:caption: 快速入门

zh/getting_started/install_guide
zh/getting_started/quick_start
```

```{toctree}
:maxdepth: 2
:caption: Agent 指南

zh/agent_guide/Profiler
zh/agent_guide/Accuracy
zh/agent_guide/Quantizer
zh/agent_guide/Modeling
zh/agent_guide/Minos
zh/agent_guide/Operator
```

```{toctree}
:maxdepth: 2
:caption: 用户指南

zh/user_guide/faq
zh/user_guide/configuration-and-extension
zh/user_guide/document-ux-review
zh/user_guide/context-compaction-guide
zh/user_guide/retry-middleware-guide
zh/user_guide/agent-tool-skill-filter-rules
```

```{toctree}
:maxdepth: 2
:caption: 开发指南

zh/developer_guide/contributing
zh/developer_guide/build-and-package
zh/developer_guide/version-and-compatibility
zh/developer_guide/arch_overview
zh/design/msagent_design
zh/developer_guide/tag-release
zh/developer_guide/readthedocs-local-build
```

```{toctree}
:maxdepth: 1
:caption: 法律与声明

zh/legal/index
zh/legal/SECURITY
zh/legal/DISCLAIMER
```

## 内置 Agent 与能力分工

| 名称 | 领域定位 | 说明 |
|---|---|---|
| **Profiler** | 性能调优 | 聚焦 Ascend Profiling 分析，覆盖单卡、多卡、集群等场景，擅长快慢卡、慢节点、MFU、通信瓶颈、算子热点、下发调度等性能问题定位与优化建议。详见 [Profiler 说明](zh/agent_guide/Profiler.md)。 |
| **Accuracy** | 精度调优 | 聚焦 Ascend 精度分析与优化，覆盖 RL 训推一致性分析、loss / gnorm NaN 分析等常见精度问题。详见 [Accuracy 说明](zh/agent_guide/Accuracy.md)。 |
| **Quantizer** | 模型量化 | 聚焦 msModelSlim 量化与压缩场景，协助完成模型适配可行性、结构风险评估与基础适配器开发。详见 [Quantizer 说明](zh/agent_guide/Quantizer.md)。 |
| **Modeling** | 仿真建模 | 聚焦大模型（LLM/VLM）仿真建模场景，承接性能建模、单点仿真、吞吐规划、设备画像与模型接入准备类问题。详见 [Modeling 说明](zh/agent_guide/Modeling.md)。 |
| **Minos** | 文档体验与代码审查 | 聚焦 README 走查、安装流程验证、Quick Start 体验、新手 onboarding、文档可用性评估，以及 GitCode PR 审查与评审意见整理。详见 [Minos 说明](zh/agent_guide/Minos.md)。 |
| **Operator** | 算子调优 | 聚焦 Ascend NPU 算子性能调优，包括算子性能深度分析、端到端算子性能优化，辅助提升调优效率并降低开发难度。详见 [Operator 说明](zh/agent_guide/Operator.md)。 |

## 推荐阅读路径

新用户建议先阅读 [入门安装指南](zh/getting_started/install_guide.md)，完成安装后再按 [快速入门指导](zh/getting_started/quick_start.md) 配置模型并启动会话。

日常使用中，优先从 [FAQ](zh/user_guide/faq.md) 和 [配置与扩展](zh/user_guide/configuration-and-extension.md) 定位常见问题；需要参与开发、发布或本地验证文档时，再进入 [贡献指南](zh/developer_guide/contributing.md) 与开发指南。
