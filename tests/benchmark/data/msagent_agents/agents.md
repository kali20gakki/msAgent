# msAgent 内置 Agent 一览

msAgent 默认提供以下内置 agent（persona）：

- **Hermes**：Ascend NPU profiling 分析 agent，采用 msprof-mcp-first 工作流。它是默认 agent（default）。
- **Zephyr**：msModelSlim 模型分析与适配助手。
- **Icarus**：Ascend NPU 算子性能优化 agent。
- **Minos**：文档 onboarding 与 GitCode PR review agent。
- **Accuracy**：Ascend NPU 模型精度分析 agent。

说明：当用户不显式指定 `--agent` 时，msAgent 会使用默认 agent Hermes。
