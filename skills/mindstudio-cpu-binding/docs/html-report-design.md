# HTML 报告设计

## 1. 目标

`mindstudio-cpu-binding` 的 HTML 报告用于直观展示当前 CPU 绑核状态、NUMA/NPU 映射、诊断问题和优化建议。报告应直接、清晰、可离线查看。

要求：

- 在 Linux 节点生成。
- 输出为单个自包含 HTML 文件。
- 不依赖外部 CDN、字体、图片或 JavaScript 包。
- 能在 Linux 和 Windows 浏览器中直接打开。
- 默认展示结论和关键证据，避免变成复杂 dashboard。

## 2. 页面结构

```text
report.html
├── 0. 摘要卡片
├── 1. 当前绑核状态
├── 2. CPU / NPU / NUMA 拓扑关系
├── 3. CPU / NUMA 逻辑 CPU 网格
├── 4. 关键进程与线程
├── 5. 问题发现
├── 6. 推荐优化方案
├── 7. 自动绑核与回滚
├── 8. 验证计划
└── 9. 信息缺口
```

## 3. 摘要卡片

摘要卡片放在页面顶部，回答用户最关心的问题：

```text
当前状态：明显不合理 / 部分不合理 / 基本合理 / 信息不足
最高风险：高 / 中 / 低 / 信息
主要问题：未绑核、跨 NUMA、线程池过载、CPU range 重叠
推荐动作：先应用保守 taskset 方案 / 先补充信息 / 保持现状并验证
是否可自动执行：是，仅低风险 taskset，需确认 / 否
```

## 4. 当前绑核状态

按进程和线程展示：

| 字段 | 说明 |
|------|------|
| PID | 目标进程。 |
| Rank / 实例 | 训练 rank 或推理实例。 |
| NPU | 对应 NPU。 |
| 当前 CPU Range | `Cpus_allowed_list`。 |
| cgroup 有效 CPU | `cpuset_cpus_effective`。 |
| 当前运行 NUMA | Top 线程所在 NUMA。 |
| 是否跨 NUMA | 是 / 否 / 信息不足。 |
| 推荐 CPU Range | 规则生成结果。 |

线程很多时只展示 Top CPU 线程和异常线程。

## 5. CPU / NPU / NUMA 拓扑关系

报告优先展示 Snapshot 驱动的拓扑关系，而不是在报告生成阶段重新执行 `lscpu` 或 `npu-smi`。拓扑关系由两层组成：

1. 轻量内联 SVG：展示 Server -> NUMA -> NPU 结构，以及 HCCS、PIX、PXB、PHB、SYS 等 NPU interconnect 链路。
2. NUMA 关系卡片：展示每个 NUMA 的 CPU range、本地 NPU、目标 PID/rank/worker、当前/有效/推荐 CPU range 和跨 NUMA 状态。

SVG 借鉴原 `topology_visualizer.py` 的节点配色和链路类型语义，但不依赖 pyvis、networkx、外部 JS 或 CDN。

每个 NUMA Node 渲染成一张关系卡：

```text
NUMA 0
- CPU Range: 0-31,64-95
- Core: physical=32, logical=64
- 本地 NPU:
  - NPU 0 / PCI 0000:81:00.0 / Local CPU 0-31,64-95 / Health ok
- 进程 / Rank / Worker:
  - PID 12345 / rank0 / NPU 0
  - 当前 CPU 0-127 / 有效 CPU 0-63 / 推荐 CPU 0-31
  - 状态：跨 NUMA
```

该 section 用于让用户直观看到：

- 每个 NUMA Node 的 CPU range。
- 每个 NPU 的 NUMA locality 和 local CPU range。
- 目标 PID、rank、worker 或实例映射到了哪个 NPU。
- 当前 CPU range、cgroup 有效 CPU range 和推荐 CPU range 的差异。
- 哪些进程当前 allowed CPU 覆盖多个 NUMA 节点。

如果 Snapshot 缺少 `npu_topology.devices`、`npu_topology.devices[*].numa_node` 或 `numa_topology.nodes`，该 section 必须显示信息缺口，而不是猜测拓扑关系。

## 6. CPU / NUMA 逻辑 CPU 网格

使用 HTML/CSS 网格展示 CPU：

```text
NUMA 0
[0] [1] [2] [3] ...
[64][65][66][67] ...

NUMA 1
[32][33][34][35] ...
[96][97][98][99] ...
```

每个逻辑 CPU 是一个小方块，方块状态通过颜色和边框表达。

### 颜色语义

| 状态 | 颜色/样式 |
|------|-----------|
| 普通 CPU | 浅灰色。 |
| 当前 allowed CPU | 蓝色。 |
| 推荐 CPU | 绿色。 |
| 当前 allowed 且推荐 | 青色。 |
| Top CPU 线程所在 CPU | 深色边框。 |
| 跨 NUMA 或冲突 CPU | 红色。 |
| cgroup 不允许 CPU | 灰色斜纹或降低透明度。 |
| SMT sibling | 虚线边框或同 core 标记。 |
| 信息缺失 | 灰色说明卡。 |

### 交互要求

第一版不要求复杂交互。可以使用简单 `title` tooltip 展示：

```text
CPU 48
NUMA: 1
Core: socket1-core16
SMT siblings: 48,112
Used by: PID 12345 / TID 12345
```

## 7. 关键进程与线程

按 `snapshot.key_processes` 展示对 Host CPU 绑核诊断有特殊意义的对象：

| 类别 | 说明 |
|------|------|
| 主调度进程 | 目标 PID、rank 主线程或模型服务入口进程。 |
| SQ 线程 | Ascend `devN_sq_task` 等固定运行时线程。 |
| NPU 固定线程 | 与特定 NPU 亲和或运行时绑定的线程。 |
| 通信线程 | HCCL、通信 worker 等线程。 |
| DataLoader 线程 | PyTorch DataLoader worker 或相关线程。 |
| Top CPU 线程 | 采样窗口内 CPU 使用率最高的关键线程。 |

该 section 应展示 PID、TID、名称、NPU、CPU%、NUMA 和分类来源。若缺少 `key_processes`，必须显示信息缺口，而不是把普通线程列表误当作关键线程。

## 8. 问题发现

每个问题渲染成独立卡片：

```text
[R001] 进程未绑定 CPU
严重程度：高
影响：稳定性 / 时延
证据：PID 12345 Cpus_allowed_list=0-127，机器 NUMA 节点数=2
判断：进程允许在全机 CPU 上运行，可能带来跨 NUMA 调度和 CPU migration。
建议：将 rank0 绑定到 NPU0 本地 NUMA 的 CPU 子集。
验证：观察 CPU migration、step time p99、NPU utilization 波动。
```

卡片按严重程度排序：high -> medium -> low -> info。

## 9. 推荐优化方案

至少展示两个方案。

### 保守方案

特点：

- 不修改系统级配置。
- 通过 `taskset` 或内部经过验证的绑核脚本临时调整目标 PID。
- 可立即回滚。
- 适合先验证。

展示字段：

| 对象 | 当前 CPU Range | 推荐 CPU Range | 命令 | 回滚命令 | 风险 |
|------|----------------|----------------|------|----------|------|

### 进阶方案

特点：

- 可能需要调整启动命令、`numactl`、PyTorch / OpenMP / DataLoader 配置。
- 通常需要重启业务。
- 第一版只生成建议，不自动执行。

展示字段：

| 对象 | 推荐配置 | 原因 | 风险 | 验证指标 |
|------|----------|------|------|----------|

## 10. 自动绑核与回滚区

该区域明确展示是否可自动执行：

```text
可自动执行：是
执行后端：taskset / internal-script
动作类型：低风险临时绑核
需要确认：是
```

执行前展示：

```text
PID 12345
执行后端: internal-script
当前 affinity: 0-127
目标 affinity: 0-31
回滚状态: rollback-state.json
回滚命令: taskset -cp 0-127 12345
风险: 可能因 CPU range 过窄导致线程竞争，建议先短时间验证。
```

如果动作不满足自动执行条件，显示：

```text
不可自动执行：推荐 CPU range 超出 cgroup cpuset_cpus_effective。
```

## 11. 验证计划

展示 before/after 对比表：

| 指标 | 优化前 | 优化后 | 变化 | 是否符合预期 |
|------|--------|--------|------|--------------|
| Throughput / QPS | | | | |
| Step time / p99 latency | | | | |
| NPU utilization | | | | |
| Host CPU utilization | | | | |
| Context switch | | | | |
| CPU migration | | | | |

## 12. 信息缺口

所有缺失字段必须集中展示：

```text
以下信息缺失，影响部分判断：
- npu_topology.devices[*].numa_node：不能判断 NPU locality。
- pytorch.dataloader.num_workers：不能判断 DataLoader 是否过载。
- baseline_metrics.training.step_time_ms_p99：不能量化优化收益。
```

## 13. 文件可移植性要求

HTML 文件必须满足：

- CSS 内联在 `<style>` 中。
- JavaScript 如有使用，内联在 `<script>` 中。
- Snapshot 摘要数据可以内嵌为 JSON script block。
- 不引用本地绝对路径。
- 不引用外部网络资源。
- 文件名建议：`mindstudio-cpu-binding-report-<timestamp>.html`。
