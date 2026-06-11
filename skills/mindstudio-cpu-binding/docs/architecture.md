# 架构设计

## 1. 总体架构

`mindstudio-cpu-binding` 第一阶段建议采用分层架构，不直接绑定某一种最终形态。

架构图 Mermaid 源文件：`../assets/architecture-overview.mmd`

## 2. 组件职责

### 2.1 Agent 交互与编排层

负责：

- 询问最少必要信息：训练/推理、PID、NPU 设备、rank 映射、环境类型、优化目标。
- 触发采集或指导用户运行采集脚本。
- 读取 Host CPU Snapshot。
- 生成诊断报告。
- 输出保守方案和进阶方案。
- 明确风险、回滚方式和验证计划。

### 2.2 专家规则层

负责沉淀 CPU 绑核相关专家经验：

- 未绑核判断。
- 跨 NUMA 判断。
- Rank / NPU / NUMA 不匹配判断。
- 可用 CPU 数与线程数不匹配判断。
- PyTorch DataLoader、intra-op、inter-op、OpenMP 线程配置建议。
- 多 rank / 多实例 CPU range 冲突判断。
- latency 与 throughput 目标下的 SMT 使用策略。

### 2.3 Snapshot 数据层

负责承接采集结果，并对 Agent 屏蔽命令差异：

- 系统拓扑。
- NPU 拓扑。
- 进程线程信息。
- cgroup/cpuset 限制。
- PyTorch 环境变量与运行配置。
- CPU 使用率和线程 TopN 采样。

### 2.4 采集层

MVP 先用脚本，后续再演进为 MCP。

MVP 脚本应尽量只读、低侵入：

- `/proc/<pid>/status`
- `/proc/<pid>/task/*/status`
- `/proc/<pid>/task/*/stat`
- `/proc/<pid>/task/*/comm`
- `lscpu`
- `numactl -H`
- cgroup cpuset/cpu quota 文件
- NPU 拓扑命令或平台适配器
- PyTorch 相关环境变量
- 轻量 CPU 采样工具

## 3. 建议的 MVP 交付形态

MVP 推荐交付为：

```text
可运行采集脚本 + Snapshot JSON + Agent 分析提示词/Skill + 报告模板 + 示例报告
```

原因：

- 不需要一开始搭建完整 MCP 或 Plugin。
- 能快速用真实 NPU/PyTorch case 验证诊断准确性。
- 采集结果结构化后，后续可以平滑替换成 MCP。
- 报告模板固定后，专家可以快速 review 输出质量。

## 4. 形态演进建议

### 阶段 1：脚本 + 文档 + Agent Prompt

目标：验证问题 taxonomy 和报告价值。

交付：

- 采集脚本。
- Snapshot schema。
- 报告模板。
- 3-5 个真实/脱敏案例。
- Agent 诊断提示词。

### 阶段 2：Skill 化

目标：把专家流程固化成可复用能力。

交付：

- CPU affinity diagnosis skill。
- 固定问题分类。
- 固定报告结构。
- 数据缺口提示机制。

### 阶段 3：MCP 化

目标：将系统采集能力工具化、结构化、可控化。

交付：

- `collect_cpu_topology`
- `collect_npu_topology`
- `collect_process_affinity`
- `collect_cgroup_limits`
- `collect_runtime_config`
- `collect_cpu_runtime_sample`
- `generate_affinity_plan`

### 阶段 4：独立 Agent / Plugin

目标：面向团队规模化使用。

交付：

- 完整交互式诊断 Agent。
- 一键采集、分析、报告生成。
- 优化前后对比。
- 可选的安全应用与回滚。
- Plugin 打包分发。

## 5. 图表

### 5.1 核心组件类图

Mermaid 源文件：`../assets/component-class-diagram.mmd`

### 5.2 用户用例图

Mermaid 源文件：`../assets/use-case-diagram.mmd`

### 5.3 数据流图

Mermaid 源文件：`../assets/data-flow-diagram.mmd`

## 6. 文档映射

| 架构部分 | 对应文档 | 说明 |
|----------|----------|------|
| Agent 交互与编排层 | `agent-workflow.md` | 定义端到端交互、采集、诊断、报告、验证和状态机。 |
| 自动绑核与回滚 | `binding-rollback-design.md` | 定义执行后端、rollback-state、回滚流程和实验计划。 |
| HTML 报告输出 | `html-report-design.md` | 定义 HTML 页面结构、CPU/NPU/NUMA 拓扑关系可视化和跨平台查看要求。 |
| 专家规则层 | `diagnosis-rules.md` | 定义问题 Taxonomy、证据字段、判断逻辑和建议策略。 |
| Agent 报告模板 | `../templates/report-template.md` | 约束诊断报告内容结构和证据表达方式。 |
| Snapshot 数据层 | `snapshot-schema.md` | 定义采集器、Agent、MCP 之间的结构化数据契约。 |
| 采集层 | `collector-design.md` | 定义 MVP 采集器 CLI、模块、安全边界和失败策略。 |
