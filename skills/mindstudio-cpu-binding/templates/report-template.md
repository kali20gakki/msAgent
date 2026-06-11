# CPU 绑核优化分析报告模板

## 0. 报告摘要

```text
诊断对象：
- 场景：PyTorch 训练 / PyTorch 离线推理 / LLM Serving
- 框架：PyTorch / PyTorch + NPU 后端 / vLLM-Ascend / SGLang / 其他
- 服务形态：离线任务 / 在线服务 / benchmark / 多实例服务
- 目标进程：PID / 进程名 / 启动命令 / API server / scheduler / engine worker / rank / instance
- 运行环境：裸机 / Docker / K8s / Slurm / 其他
- NPU 设备：设备 ID / Rank 映射 / Worker 映射 / Instance 映射
- 优化目标：samples/s / step time / batch throughput / QPS / tokens/s / TTFT / TPOT / p99 latency / 稳定性 / 资源隔离
- 采集时间：
- 采集时长：

总体结论：
- 当前 CPU 绑核状态：合理 / 部分不合理 / 明显不合理 / 信息不足
- 主要问题：
  1. ...
  2. ...
  3. ...
- 推荐动作：
  1. ...
  2. ...
  3. ...
- 预期收益：吞吐提升 / 时延降低 / 抖动降低 / NPU 利用率提升 / Host 侧 CPU 竞争降低
- 风险等级：低 / 中 / 高
```

## 1. 当前 CPU 绑定状态

```text
进程级绑定：
- PID：
- Cpus_allowed_list：
- Mems_allowed_list：
- 当前实际运行 CPU 分布：
- 是否受 cgroup/cpuset 限制：
- 容器/K8s 可用 CPU：
- CPU quota / throttling 情况：

线程级绑定摘要：
| TID | 线程名 | Allowed CPUs | 当前 CPU | CPU 使用率 | NUMA Node | 备注 |
|-----|--------|--------------|----------|------------|-----------|------|
|     |        |              |          |            |           |      |
```

重点展示关键线程、Top CPU 线程和异常线程；线程很多时展示摘要和 Top N。

## 2. CPU / NPU / NUMA 拓扑关系

```text
拓扑关系摘要：
- NUMA Node 数：
- 逻辑 CPU 数：
- NPU 数：
- 目标进程数：
- 当前跨 NUMA 进程数：

NUMA 关系卡片：
- NUMA 0
  - CPU Range：...
  - 物理 Core 数：...
  - 逻辑 CPU 数：...
  - 本地 NPU：
    - NPU 0 / PCI ... / Local CPU ... / Health ...
  - 进程 / Rank / Worker：
    - PID ... / rank... / NPU ... / 当前 CPU ... / 有效 CPU ... / 推荐 CPU ... / 状态：本地 NUMA / 跨 NUMA / 待确认

- NUMA 1
  - CPU Range：...
  - 本地 NPU：...
  - 进程 / Rank / Worker：...

拓扑提示：
- PID ... 当前 allowed CPU 覆盖多个 NUMA 节点。

信息缺口：
- 缺少 npu_topology.devices 时，不能展示 NPU -> NUMA 关系。
- 缺少 npu_topology.devices[*].numa_node 时，不能判断 NPU locality。
```

该 section 由 Snapshot 和诊断计划渲染，不在报告生成阶段重新执行 `lscpu` 或 `npu-smi`。报告先用轻量内联 SVG 展示 Server -> NUMA -> NPU 关系和 NPU interconnect，再用 NUMA 关系卡片展示 PID/rank/worker 与推荐 CPU range。SVG 和卡片都必须自包含在 `report.html` 中，不依赖 pyvis、networkx、外部 JS 或 CDN。如果缺少 NPU 拓扑、NUMA 拓扑或 rank/device 映射，必须明确写入“信息缺口”，不要假设。

## 3. CPU / NUMA 逻辑 CPU 网格

```text
逻辑 CPU 网格：
- 普通 CPU：灰色
- 当前 allowed CPU：蓝色
- 推荐 CPU：绿色
- 当前 allowed 且推荐：青色

NUMA 0: [0] [1] [2] ...
NUMA 1: [32] [33] [34] ...
```

SMT sibling 信息：
| Physical Core | Logical CPUs | NUMA Node |

|---------------|--------------|-----------|
|               |              |           |

## 4. 运行时 CPU 使用与竞争情况

```text
进程 CPU 使用摘要：
- 进程总 CPU 使用率：
- Top CPU 线程：
  | TID | 线程名 | CPU% | 当前 CPU | NUMA | Context Switch | 备注 |
  |-----|--------|------|----------|------|----------------|------|

系统 CPU 使用摘要：
- 各 NUMA CPU 使用率：
  - NUMA 0:
  - NUMA 1:
- 当前绑定范围内 CPU 使用率：
- runqueue / load average：
- context switch 情况：
- 是否存在 CPU 过载：
- 是否存在明显 CPU 迁移：
```

MVP 阶段优先使用轻量指标：CPU 使用率、Top 线程、context switch、当前 CPU、NUMA 分布。

## 5. 问题发现

每个问题使用固定结构：

```text
问题 N：<问题标题>

严重程度：高 / 中 / 低
影响目标：吞吐 / 时延 / 稳定性 / 资源隔离

现象：
- ...

证据：
- ...

判断：
- ...

建议：
- ...

风险：
- ...

验证方式：
- ...
```

第一阶段优先覆盖的问题类型：

1. 进程未绑核。
2. 绑核范围过宽。
3. 绑核范围过窄。
4. 跨 NUMA 运行。
5. Rank / Worker / Instance / NPU / NUMA 不匹配。
6. 多 rank、多 worker 或多实例 CPU range 重叠。
7. SMT sibling 使用不符合 latency/throughput 目标。
8. PyTorch / Runtime / Serving 线程数超过可用 CPU。
9. cgroup cpuset 与应用绑核冲突。
10. CPU quota / throttling。
11. DataLoader / tokenizer / scheduler / API server / engine worker 与主进程或 runtime 线程竞争。

## 6. 推荐绑核方案

### 6.1 保守方案

```text
目标：
- 在不修改系统级配置的前提下，减少跨 NUMA 和 CPU 竞争。

建议：
| 对象 | 当前 CPU Range | 推荐 CPU Range | 推荐 Mems | 原因 |
|------|----------------|----------------|-----------|------|
| PID  |                |                |           |      |

推荐命令：
taskset -cp <cpu-list> <pid>

或：
numactl --cpunodebind=<node> --membind=<node> <command>

风险：
- 低
- 只影响目标进程
- 可通过重启或重新设置 affinity 恢复
```

### 6.2 进阶方案

```text
目标：
- 同时优化进程绑核、PyTorch/DataLoader 或 LLM serving 线程配置、实例隔离和 NUMA locality。

建议：
| 对象 | 推荐配置 | 原因 |
|------|----------|------|
| Rank / 主进程 | CPU ... |      |
| DataLoader workers | CPU ... |      |
| API server | CPU ... |      |
| Scheduler | CPU ... |      |
| Tokenizer workers | CPU ... |      |
| Engine / worker process | CPU ... |      |
| Runtime / 通信线程 | CPU ... |      |
| OMP_NUM_THREADS | ... |      |
| MKL_NUM_THREADS | ... |      |
| torch.set_num_threads | ... |      |
| serving runtime workers | ... |      |

推荐启动方式：
OMP_NUM_THREADS=... MKL_NUM_THREADS=... \
numactl --cpunodebind=... --membind=... \
python train.py ...

风险：
- 中
- 需要重启进程
- 需要结合 workload 验证
```

## 7. 推荐 PyTorch / Runtime / Serving 线程配置

```text
通用线程池：
- OMP_NUM_THREADS=
- MKL_NUM_THREADS=
- OPENBLAS_NUM_THREADS=
- GOTO_NUM_THREADS=
- KMP_AFFINITY=
- KMP_BLOCKTIME=

PyTorch：
- torch.set_num_threads(...)
- torch.set_num_interop_threads(...)
- DataLoader num_workers=
- DataLoader pin_memory=
- DataLoader prefetch_factor=

LLM Serving：
- API server 线程/进程 CPU：
- Scheduler 线程/进程 CPU：
- Tokenizer worker CPU：
- Engine / worker process CPU：
- TP / DP / PP rank -> NPU 映射：
- Prefill / decode worker 是否需要隔离 CPU：待确认 / 已确认
- Queueing、scheduler、tokenizer 是否存在 CPU 竞争：待确认 / 已确认

NPU/PyTorch 后端：
- Rank -> NPU device 映射：
- NPU runtime 相关线程是否需要独立 CPU：待确认 / 已确认
- 通信线程是否靠近 NPU/NIC 所在 NUMA：待确认 / 已确认
```

如果未检测到相关框架配置，不要强行给具体值，应标注“需确认”。

## 8. 验证计划

```text
优化前基线：
- PyTorch 训练 throughput / samples per second：
- step time 平均值 / p50 / p90 / p99：
- PyTorch 离线推理 batch throughput：
- 推理 QPS：
- 推理 latency p50 / p90 / p99：
- LLM serving input tokens/s：
- LLM serving output tokens/s：
- TTFT p50 / p90 / p99：
- TPOT p50 / p90 / p99：
- queueing latency：
- prefill / decode throughput：
- timeout / error rate：
- NPU utilization：
- Host CPU utilization by NUMA：
- API server / scheduler / tokenizer / engine worker CPU utilization：
- context switch：
- CPU migration：
- NUMA remote memory：

优化后验证：
1. 应用保守方案；
2. 运行相同 workload；
3. 使用相同压测或采集方式；
4. 采集相同时长；
5. 对比以下指标：

| 指标 | 优化前 | 优化后 | 变化 | 是否符合预期 |
|------|--------|--------|------|--------------|
| Throughput / QPS / tokens/s | | | | |
| Step time / latency p99 | | | | |
| TTFT / TPOT | | | | |
| Queueing latency | | | | |
| Timeout / error rate | | | | |
| NPU utilization | | | | |
| CPU utilization by NUMA | | | | |
| API server / scheduler / tokenizer / engine CPU | | | | |
| Context switch | | | | |
| CPU migration | | | | |
```

## 9. 风险与回滚

```text
风险等级：
- 保守方案：低 / 中 / 高
- 进阶方案：低 / 中 / 高

潜在风险：
- CPU 绑定过窄导致线程饥饿。
- 多 NPU 共享进程时，绑定到单 NUMA 可能反而变差。
- DataLoader worker 过少导致输入不足。
- Tokenizer、scheduler 或 API server CPU 过窄导致 TTFT、queueing latency 或 p99 变差。
- Engine worker CPU 与其他实例重叠导致 decode tokens/s 抖动。
- 线程池过小导致 CPU 算子或 serving runtime 调度变慢。
- K8s 环境下应用绑核超出 cpuset 无效。
- 使用 SMT sibling 可能影响低时延场景。
- membind 可能导致内存不足或远端访问异常。

回滚方式：
taskset -cp <original-cpu-list> <pid>

如果是启动命令变更：
- 恢复原启动命令和原环境变量。
```

## 10. 信息缺口

```text
以下信息缺失，可能影响判断准确性：
- 未获取 NPU -> NUMA 映射。
- 未确认 rank / worker / instance -> NPU 映射。
- 未获取容器 cgroup cpuset。
- 未采集 workload 运行期 CPU 使用率。
- 未获取优化目标：throughput / latency / stability。
- 未获取 PyTorch / Runtime / Serving 线程配置。
- 未获取 LLM serving 的 TTFT / TPOT / tokens/s / queueing latency 基线。
- 未确认 API server / scheduler / tokenizer / engine worker 进程关系。
- 未获取优化前性能基线。

建议补充：
1. ...
2. ...
3. ...
```
