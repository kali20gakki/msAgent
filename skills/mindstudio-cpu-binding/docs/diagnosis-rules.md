# CPU 绑核诊断规则

## 1. 目标

本文定义 `mindstudio-cpu-binding` MVP 的问题 Taxonomy、证据字段、判断逻辑、建议策略和验证方式。诊断规则必须基于 `snapshot-schema.md` 中的字段，避免“凭经验猜测”。

规则输出应统一映射到报告模板中的“问题发现”和“推荐绑核方案”。

## 2. 规则输出格式

每条诊断发现使用统一结构：

```json
{
  "id": "cross_numa_execution",
  "title": "进程跨 NUMA 运行",
  "severity": "medium",
  "impact": ["throughput", "stability"],
  "evidence": [],
  "judgement": "...",
  "recommendations": [],
  "risk": "...",
  "verification": []
}
```

字段约束：

| 字段 | 说明 |
|------|------|
| `id` | 稳定规则 ID，使用 snake_case。 |
| `title` | 报告中的问题标题。 |
| `severity` | `high` / `medium` / `low` / `info`。 |
| `impact` | `throughput` / `latency` / `stability` / `isolation`。 |
| `evidence` | 必须引用 Snapshot 字段。 |
| `judgement` | 诊断判断，不超过 3 句话。 |
| `recommendations` | 可执行建议，可分保守/进阶。 |
| `risk` | 风险说明。 |
| `verification` | 优化后验证指标。 |

## 3. 规则分级

### high

满足以下任一条件：

- 目标进程在多 NUMA 机器上允许使用全量 CPU，且用户目标是 latency/stability。
- rank -> NPU -> NUMA 映射明确，但进程主要运行在远端 NUMA。
- cgroup 可用 CPU 少于 PyTorch/OpenMP/DataLoader 总线程需求，且 CPU 使用率高。
- 多实例或多 rank CPU range 明确重叠，并且当前目标是稳定性或低时延。

### medium

- 存在跨 NUMA 运行，但缺少 NPU locality 或性能基线。
- 线程数明显多于可用 CPU，但暂未观测到 throttling 或严重过载。
- PyTorch 线程池配置与 cpuset 不匹配，但影响程度待验证。

### low

- 有轻微优化空间，但当前证据不足以说明明显瓶颈。
- 信息缺失导致只能提示补充采集。

### info

- 当前配置合理。
- 仅输出环境摘要或验证建议。

## 4. R001：进程未绑定 CPU

### 触发条件

满足全部：

1. `processes[*].cpus_allowed_list` 等于或接近 `system.online_cpus`。
2. `numa_topology.nodes.length > 1`。
3. 目标进程是 NPU/PyTorch 训练或推理进程。

### 证据字段

- `processes[*].cpus_allowed_list`
- `system.online_cpus`
- `numa_topology.nodes[*].cpus`
- `workload.scenario`
- `workload.optimization_goal`

### 判断

进程允许在全机 CPU 上运行，Host 线程可能被调度到非本地 NUMA 或与其他 rank/实例竞争，导致缓存局部性下降、CPU migration 增加和性能抖动。

### 建议

保守方案：

- 如果 rank -> NPU 映射明确，将每个 rank 绑定到对应 NPU 本地 NUMA 的部分 CPU。
- 如果映射不明确，先补充 rank -> NPU 和 NPU -> NUMA 信息。

进阶方案：

- 按 rank 或实例划分不重叠 CPU range。
- 同步调整 PyTorch DataLoader、OpenMP、intra-op、inter-op 线程数。

### 验证

- CPU migration 是否下降。
- NPU utilization 是否更稳定。
- 训练 step time p90/p99 是否下降。
- 推理 p99 latency 是否下降。

## 5. R002：跨 NUMA 运行

### 触发条件

满足任一：

1. `runtime_sample.top_threads[*].numa_node` 分布在多个 NUMA 节点。
2. 同一进程的高 CPU 线程运行在多个 NUMA 节点。
3. `processes[*].current_cpu` 或线程 `current_cpu` 属于非推荐 NUMA。

### 证据字段

- `runtime_sample.top_threads[*].numa_node`
- `processes[*].threads[*].current_cpu`
- `cpu_topology.cpus[*].numa_node`
- `numa_topology.nodes[*].cpus`

### 判断

目标进程的关键线程跨 NUMA 调度。对于 NPU 训练/推理，若设备和内存访问更靠近某个 NUMA，跨 NUMA 可能增加访问延迟和抖动。

若 `snapshot.key_processes` 可用，应优先关注其中的主调度、SQ、通信和 DataLoader 线程；这些线程比普通低 CPU 后台线程更能代表 Host 侧瓶颈风险。

### 建议

- 将目标进程 CPU range 收敛到对应 NPU 本地 NUMA。
- 对多 rank 任务，为每个 rank 分配不同 NUMA 或同 NUMA 下不重叠 CPU range。
- 如果进程确实服务多个 NPU，不应强行绑定到单一 NUMA，应按进程模型拆分或保留更宽范围。

### 验证

- 对比优化前后 `top_threads` 的 NUMA 分布。
- 对比 step time / p99 latency。
- 对比 NPU utilization 波动。

## 6. R003：Rank / NPU / NUMA 不匹配

### 触发条件

满足全部：

1. `workload.rank_mapping[*].pid` 和 `workload.rank_mapping[*].npu_device` 可用。
2. `npu_topology.devices[*].numa_node` 可用。
3. 目标 PID 的 `cpus_allowed_list` 主要不属于该 NPU 的本地 NUMA，或高 CPU 线程主要运行在远端 NUMA。

### 证据字段

- `workload.rank_mapping`
- `npu_topology.devices[*].numa_node`
- `npu_topology.devices[*].local_cpus`
- `processes[*].cpus_allowed_list`
- `runtime_sample.top_threads`

### 判断

Rank 进程与其使用的 NPU 不在同一 NUMA locality，可能导致 Host 侧 runtime、数据准备或通信线程访问路径变差。

### 建议

- 为每个 rank 生成 `rank -> npu -> numa -> cpu_range` 映射。
- 优先使用 NPU 本地 NUMA 的物理 core。
- 如果追求低时延，优先避免 SMT sibling 共享；如果追求吞吐，可在验证后使用 SMT。

### 验证

- 单 rank step time 是否下降。
- 多 rank 间 step time 方差是否降低。
- NPU utilization 是否更稳定。

## 7. R004：绑核范围过宽

### 触发条件

满足任一：

1. 单 rank 进程允许使用超过一个 NUMA 的 CPU，且该 rank 只使用一个 NPU。
2. 推理单实例允许使用全机 CPU，但目标是 latency 或 stability。
3. 多实例进程 CPU range 大量重叠。

### 证据字段

- `processes[*].cpus_allowed_list`
- `workload.process_model`
- `workload.optimization_goal`
- `workload.rank_mapping`

### 判断

绑核范围过宽会增加调度不确定性，使关键线程与其他进程、rank 或实例争抢 CPU。

### 建议

- 训练：按 rank 分配本地 NUMA CPU 子集。
- 推理：按实例分配独占 physical core range。
- 对 latency/stability 优先任务，避免多个实例共享 SMT sibling。

### 验证

- CPU migration 下降。
- p99 latency 或 step time 抖动下降。
- 实例间性能方差下降。

## 8. R005：绑核范围过窄

### 触发条件

满足全部：

1. 有效 CPU 数小于目标进程需求估算。
2. `runtime_sample.process_cpu_percent_total` 接近或超过有效 CPU 数 × 100%。
3. `processes[*].num_threads` 明显大于有效 CPU 数。

需求估算：

```text
effective_cpu_count = cpuset 或 cpus_allowed_list 中的逻辑 CPU 数
estimated_thread_demand = max(
  torch_num_threads,
  OMP_NUM_THREADS,
  DataLoader num_workers + 1,
  top active threads count
)
```

### 证据字段

- `processes[*].cpus_allowed_list`
- `cgroup.process_groups[*].cpuset_cpus_effective`
- `processes[*].num_threads`
- `pytorch.threading.*`
- `pytorch.dataloader.num_workers`
- `runtime_sample.process_cpu_percent_total`

### 判断

当前可用 CPU 可能不足，导致线程竞争、context switch 增加或输入 pipeline 变慢。

### 建议

- 增加 CPU range，或减少 PyTorch/OpenMP/DataLoader 线程数。
- 如果是推理多实例，减少实例数或重新划分 core。
- 如果是容器环境，检查 CPU request/limit 和 cpuset。

### 验证

- CPU utilization 是否从满载下降到合理区间。
- context switch 是否下降。
- NPU 利用率是否上升。

## 9. R006：PyTorch 线程池过载

### 触发条件

满足任一：

1. `OMP_NUM_THREADS` 大于有效 CPU 数。
2. `torch_num_threads` 大于有效 CPU 数。
3. 多 rank 下每个 rank 的 `OMP_NUM_THREADS` 之和大于本地 NUMA 可用 CPU 数。
4. `DataLoader num_workers + torch_num_threads` 明显大于进程有效 CPU 数。

### 证据字段

- `pytorch.threading.omp_num_threads`
- `pytorch.threading.torch_num_threads`
- `pytorch.threading.torch_num_interop_threads`
- `pytorch.dataloader.num_workers`
- `processes[*].cpus_allowed_list`
- `cgroup.process_groups[*].cpuset_cpus_effective`

### 判断

PyTorch、OpenMP 或 DataLoader 线程配置超过实际可用 CPU，可能造成 oversubscription 和调度竞争。

### 建议

训练场景：

- 每 rank 的 `OMP_NUM_THREADS` 不应默认开满全机 CPU，应按 rank 分配 CPU 数设置。
- `DataLoader num_workers` 应结合数据预处理强度和 CPU range 验证。
- 对 NPU-heavy 训练，避免 Host 线程池占满本地 NUMA 全部 core。

推理场景：

- 低时延优先时，减少 intra-op 线程和 worker 竞争。
- 多实例时，每实例线程数不应超过实例独占 core 数。

### 验证

- context switch 下降。
- step time 或 p99 latency 稳定性改善。
- Host CPU 使用更均匀。

## 10. R007：cgroup/cpuset 与应用绑核冲突

### 触发条件

满足任一：

1. `processes[*].cpus_allowed_list` 包含不在 `cgroup.process_groups[*].cpuset_cpus_effective` 中的 CPU。
2. 用户推荐或当前启动命令中的 CPU range 超出 cgroup 有效范围。
3. `cpu_quota_us` 或 `cpu_max` 显示有配额限制，且 `nr_throttled` 增长。

### 证据字段

- `processes[*].cpus_allowed_list`
- `cgroup.process_groups[*].cpuset_cpus_effective`
- `cgroup.process_groups[*].cpu_max`
- `cgroup.process_groups[*].nr_throttled`
- `cgroup.process_groups[*].throttled_usec`

### 判断

容器或 K8s cgroup 限制可能使应用层绑核无效，或者 CPU quota 导致周期性 throttling。

### 建议

- 以 `cpuset_cpus_effective` 作为真实 CPU 上限生成建议。
- K8s 场景优先检查 Guaranteed QoS、CPU request/limit 和 CPU Manager static policy。
- 不建议在应用内绑定到 cgroup 不允许的 CPU。

### 验证

- throttling 计数是否不再增长。
- p99 latency 抖动是否下降。
- 实际线程 CPU 是否落在有效 cpuset 内。

## 11. R008：多 rank / 多实例 CPU range 重叠

### 触发条件

满足全部：

1. `workload.process_model` 为 `multi-rank` 或 `multi-instance`。
2. 至少两个目标 PID 的有效 CPU range 有明显交集。
3. 优化目标为 `latency`、`stability` 或 `isolation`，或者运行采样显示 CPU 竞争。

### 证据字段

- `workload.process_model`
- `processes[*].pid`
- `processes[*].rank`
- `processes[*].cpus_allowed_list`
- `runtime_sample.cpu_usage_by_numa`

### 判断

多个 rank 或实例共享 CPU range，可能导致性能抖动和实例间干扰。

### 建议

- 为每个 rank/实例分配不重叠 CPU range。
- 推理低时延场景优先使用不共享 SMT sibling 的物理 core。
- 训练吞吐场景可在验证后允许同 NUMA 内合理共享 SMT sibling。

### 验证

- rank 间 step time 方差。
- 实例间 p99 latency 方差。
- CPU 使用分布是否符合预期。

## 12. R009：SMT 使用策略与目标不匹配

### 触发条件

满足任一：

1. `system.smt_enabled = true` 且低时延推理实例共享同一 physical core 的 sibling。
2. 绑核建议中使用了 SMT sibling，但用户目标是 latency 或 stability。
3. 绑核建议完全避开 SMT，但用户目标是 throughput 且 CPU 资源不足。

### 证据字段

- `system.smt_enabled`
- `cpu_topology.physical_cores[*].logical_cpus`
- `workload.optimization_goal`
- `processes[*].cpus_allowed_list`

### 判断

SMT 对吞吐和低时延目标的影响不同。低时延场景共享 sibling 可能造成尾延迟抖动；吞吐场景合理使用 SMT 可能提升 CPU 利用率。

### 建议

- latency/stability：优先使用独占 physical core，避免实例共享 sibling。
- throughput：先使用 physical core，不足时再逐步加入 sibling，并用基线验证收益。

### 验证

- p99 latency。
- throughput。
- CPU utilization。
- 实例间性能方差。

## 13. R010：信息不足

### 触发条件

`availability.missing` 包含关键字段。

关键字段：

- `npu_topology.devices[*].numa_node`
- `workload.rank_mapping`
- `processes[*].threads[*].current_cpu`
- `cgroup.process_groups[*].cpuset_cpus_effective`
- `pytorch.threading.*`
- `runtime_sample.top_threads`
- `key_processes.main_scheduler_pids`
- `key_processes.sq_task_threads`
- `key_processes.communication_threads`

### 判断

当前 Snapshot 信息不足以支持完整诊断。报告必须列出缺口，并说明哪些结论不能下。

### 建议

- 补充 NPU topology。
- 补充 rank -> NPU 映射。
- 补充运行期 CPU 采样。
- 补充 PyTorch 线程池和 DataLoader 配置。

### 验证

补齐字段后重新生成报告。

## 14. 推荐方案生成原则

### 14.1 CPU range 选择

优先级：

1. cgroup `cpuset_cpus_effective`。
2. NPU 本地 NUMA `local_cpus`。
3. 物理 core 优先于 SMT sibling。
4. 多 rank/实例不重叠。
5. 根据目标选择是否使用 SMT。

### 14.2 训练场景

- 每个 rank 优先绑定到其 NPU 本地 NUMA。
- 多 rank 同 NUMA 时平均切分 physical cores。
- DataLoader 较重时保留额外 CPU 给 worker。
- `OMP_NUM_THREADS` 不应超过 rank 分配 CPU 数。

### 14.3 推理场景

- latency/stability 优先时，每个实例独占 physical cores。
- tokenizer/preprocess/postprocess 与模型 worker 可以分区。
- 多实例避免 CPU range 重叠。
- 若追求 throughput，可增加 SMT sibling 并验证。

### 14.4 风险分级

低风险：

- 只生成建议。
- 临时 `taskset -cp` 调整目标进程。
- 调整下次启动时环境变量。

中风险：

- 重启训练/推理进程应用新启动命令。
- 调整容器 cpuset 或 K8s resource 配置。

高风险：

- 修改 IRQ affinity。
- 修改 CPU governor。
- 修改 kernel boot 参数。
- 线上服务自动重启。

MVP 不执行高风险动作。
