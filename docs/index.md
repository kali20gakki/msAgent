# MindStudio-Agent 文档

MindStudio-Agent 是面向 Ascend NPU 场景的一站式调试调优 Agent。中文文档统一维护在 `docs/zh/` 下，按新用户上手、日常使用、Agent 能力和开发维护四类组织。

```{toctree}
:maxdepth: 2
:caption: 快速入门

zh/quick_start/msagent_quick_start
```

```{toctree}
:maxdepth: 2
:caption: 安装指南

zh/install_guide/msagent_install_guide
```

```{toctree}
:maxdepth: 2
:caption: Agent 指南

zh/agent_guide/profiler
zh/agent_guide/accuracy
zh/agent_guide/quantizer
zh/agent_guide/modeling
zh/agent_guide/minos
zh/agent_guide/operator
zh/agent_guide/spectrainer
```

```{toctree}
:maxdepth: 2
:caption: 用户指南

zh/user_guide/agent-tool-skill-filter-rules
zh/user_guide/configuration-and-extension
zh/user_guide/context-compaction-guide
zh/user_guide/document-ux-review
zh/user_guide/integration-guide
zh/user_guide/permission-approval-guide
zh/user_guide/retry-middleware-guide
zh/user_guide/usemap
```

```{toctree}
:maxdepth: 2
:caption: 典型案例

zh/best_practices/ascend-profiler-collect-adaption-practice
zh/best_practices/nan-overflow-detection-practice
zh/best_practices/op-fusion-replace-practice
zh/best_practices/op-mfu-profiler-practice
zh/best_practices/quantization-auto-tuning-practice
zh/best_practices/spike-root-cause-analysis-practice
```

```{toctree}
:maxdepth: 2
:caption: FAQ

zh/support/faq
```

```{toctree}
:maxdepth: 2
:caption: 开发指南

zh/development_guide/add-agent
zh/development_guide/build-and-package
zh/development_guide/version-and-compatibility
zh/development_guide/arch_overview
zh/development_guide/readthedocs-local-build
zh/development_guide/design/msagent_design
zh/development_guide/design/msprof_mcp_design
zh/development_guide/design/npu_snapshot_analysis
```

```{toctree}
:maxdepth: 2
:caption: 贡献指南

zh/contributing/contributing
```

```{toctree}
:maxdepth: 1
:caption: 法律与声明

zh/legal/disclaimer
zh/legal/SECURITY
```

## 内置 Agent 与能力分工

| 名称 | 领域定位 | 说明 |
|---|---|---|
| **Profiler** | 性能调优 | 聚焦 Ascend Profiling 分析，覆盖单卡、多卡、集群等场景，擅长快慢卡、慢节点、MFU、通信瓶颈、算子热点、下发调度等性能问题定位与优化建议。详见 [Profiler 说明](zh/agent_guide/profiler.md)。 |
| **Accuracy** | 精度调优 | 聚焦 Ascend 精度分析与优化，覆盖 RL 训推一致性分析、loss / gnorm NaN 分析等常见精度问题。详见 [Accuracy 说明](zh/agent_guide/accuracy.md)。 |
| **Quantizer** | 模型量化 | 聚焦 msModelSlim 量化与压缩场景，协助完成模型适配可行性、结构风险评估与基础适配器开发。详见 [Quantizer 说明](zh/agent_guide/quantizer.md)。 |
| **Modeling** | 仿真建模 | 聚焦大模型（LLM/VLM）仿真建模场景，承接性能建模、单点仿真、吞吐规划、设备画像与模型接入准备类问题。详见 [Modeling 说明](zh/agent_guide/modeling.md)。 |
| **Minos** | 文档体验与代码审查 | 聚焦 README 走查、安装流程验证、Quick Start 体验、新手 onboarding、文档可用性评估，以及 GitCode PR 审查与评审意见整理。详见 [Minos 说明](zh/agent_guide/minos.md)。 |
| **Operator** | 算子调优 | 聚焦 Ascend NPU 算子性能调优，包括算子性能深度分析、端到端算子性能优化，辅助提升调优效率并降低开发难度。详见 [Operator 说明](zh/agent_guide/operator.md)。 |
| **SpecTrainer** | 投机解码重采样 | 聚焦投机解码训练数据的 on-policy 重采样（响应重生成），把多轮对话数据用 verifier 模型逐轮重生成，产出可直接进训练的预分词样本。详见 [SpecTrainer 说明](zh/agent_guide/spectrainer.md)。 |
| **HostCPU** | 主机侧性能调优 | 聚焦 Host 侧（CPU / NUMA / 中断 / 绑核）配置审计、绑核方案求解与症状归因，覆盖新机器开箱、绑核未生效、利用率低、训练变慢等场景。详见 [HostCPU 说明](zh/agent_guide/hostcpu.md)。 |

## 推荐阅读路径

新用户建议先阅读 [安装指南](zh/install_guide/msagent_install_guide.md)，完成安装后再按 [快速入门](zh/quick_start/msagent_quick_start.md) 配置模型并启动会话。

日常使用中，优先从 [FAQ](zh/support/faq.md) 和 [配置与扩展](zh/user_guide/configuration-and-extension.md) 定位常见问题；需要参与开发、发布或本地验证文档时，再进入 [贡献指南](zh/contributing/contributing.md) 与开发指南。
