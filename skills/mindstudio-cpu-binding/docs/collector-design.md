# MVP 采集器设计

## 1. 目标

MVP 采集器负责在目标机器上以只读方式采集 Host CPU 绑核分析所需信息，并输出符合 `snapshot-schema.md` 的 JSON 文件。它不做诊断、不修改系统、不自动优化，只负责稳定、低侵入、可追溯地收集证据。

当前仓库阶段尚未实现本文件描述的完整 `collector/collect.py`。已实现的是 `scripts/topology_collect.py` 和 `scripts/process_discovery.py` 两个只读原型，用于先验证拓扑采集解析和进程发现的数据契约；完整 Snapshot collector 后续应复用这些 parser 和输出结构。

## 2. 交付形态

建议第一版交付为一个 Python CLI：

```text
mindstudio-cpu-binding/collector/
├── collect.py
├── adapters/
│   ├── linux_cpu.py
│   ├── linux_proc.py
│   ├── linux_cgroup.py
│   ├── npu_topology.py
│   └── runtime_env.py
└── README.md
```

当前阶段先定义完整 collector 设计；以下带 `samples/*.txt` 的命令仅用于仓库离线示例验证，真实 Linux NPU 节点请使用后面的 live 只读命令或完整 collect 命令。以下命令默认在 Skill 目录执行：

```bash
cd skills/mindstudio-cpu-binding
python scripts/cli.py collect-topology --lscpu-file samples/lscpu.sample.txt --npu-smi-topo-file samples/npu-smi-topo.sample.txt --out out/topology.json
python scripts/cli.py discover-processes --ps-file samples/ps.sample.txt --npu-smi-info-file samples/npu-smi-info.sample.txt --out out/processes.json
```

真实 Linux NPU 设备上可用独立原型入口验证 live 输出：

```bash
cd skills/mindstudio-cpu-binding
python scripts/topology_collect.py --out out/topology.json
python scripts/process_discovery.py --out out/processes.json
```

## 3. CLI 入口设计

当前完整 Snapshot 采集原型入口为：

```bash
cd skills/mindstudio-cpu-binding
python scripts/cli.py collect \
  --pid 12345 \
  --scenario training \
  --framework pytorch \
  --device-type npu \
  --optimization-goal throughput \
  --rank-map rank0=12345:npu0,rank1=12346:npu1 \
  --sample-seconds 10 \
  --out out/snapshot.json
```

### 参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `--pid` | 是 | 目标 PID，可重复。 |
| `--scenario` | 否 | `training` / `inference` / `unknown`。 |
| `--framework` | 否 | MVP 默认为 `pytorch`。 |
| `--device-type` | 否 | MVP 默认为 `npu`。 |
| `--optimization-goal` | 否 | `throughput` / `latency` / `stability` / `isolation`。 |
| `--rank-map` | 否 | rank、PID、NPU 设备映射。 |
| `--sample-seconds` | 否 | 运行期采样时长，默认 10 秒。 |
| `--output` | 是 | Snapshot JSON 输出路径。 |
| `--raw-dir` | 否 | 原始命令输出目录。 |

## 4. 采集原则

- 默认只读。
- 不要求 root；缺少权限时记录 `availability.errors`。
- 不执行 `taskset -p` 修改、`numactl` 启动、sysctl、写 cgroup 文件等动作。
- 不依赖 ftrace/eBPF/perf。
- 外部命令缺失时降级到 `/proc`、`/sys` 文件。
- 原始输出落盘到 `raw_refs`，便于问题追溯。

## 5. 采集模块

## 5.1 linux_cpu.py

### 目标

采集系统、CPU、NUMA、SMT 拓扑。

### 数据来源

优先：

```bash
lscpu
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ,MHZ
numactl -H
```

备用：

```text
/sys/devices/system/cpu/online
/sys/devices/system/cpu/cpu*/topology/core_id
/sys/devices/system/cpu/cpu*/topology/physical_package_id
/sys/devices/system/cpu/cpu*/topology/thread_siblings_list
/sys/devices/system/node/node*/cpulist
/sys/devices/system/node/node*/meminfo
```

### 输出字段

- `system.*`
- `cpu_topology.cpus`
- `cpu_topology.physical_cores`
- `numa_topology.nodes`
- `numa_topology.distance_matrix`

### 失败处理

- `numactl` 不存在时，仍从 `/sys/devices/system/node` 获取 NUMA CPU list。
- 无法获取频率时，`max_mhz/current_mhz` 为 `null`。

## 5.2 linux_proc.py

### 目标

采集目标进程和线程 affinity、状态、当前 CPU、context switch、线程名。

### 数据来源

```text
/proc/<pid>/status
/proc/<pid>/stat
/proc/<pid>/cmdline
/proc/<pid>/task/<tid>/status
/proc/<pid>/task/<tid>/stat
/proc/<pid>/task/<tid>/comm
```

可选命令：

```bash
ps -eLo pid,tid,psr,pcpu,stat,comm
```

### 字段映射

| 来源 | Snapshot 字段 |
|------|---------------|
| `/proc/<pid>/status` `Cpus_allowed_list` | `processes[*].cpus_allowed_list` |
| `/proc/<pid>/status` `Mems_allowed_list` | `processes[*].mems_allowed_list` |
| `/proc/<tid>/stat` processor | `threads[*].current_cpu` |
| `/proc/<tid>/status` ctxt switches | `threads[*].voluntary_ctxt_switches` / `nonvoluntary_ctxt_switches` |
| `/proc/<tid>/comm` | `threads[*].name` |

### role_hint 推断

MVP 仅做保守推断：

| 条件 | role_hint |
|------|-----------|
| TID 等于 PID | `main` |
| 线程名包含 `DataLoader` / `worker` | `dataloader` |
| 线程名包含 `omp` / `OpenMP` | `openmp_worker` |
| 线程名包含 `blas` / `mkl` | `blas_worker` |
| 线程名包含 `hccl` / `comm` / `communication` | `communication` |
| 其他 | `unknown` |

## 5.3 linux_cgroup.py

### 目标

采集目标 PID 的 cgroup v1/v2 CPU 和 cpuset 限制。

### 数据来源

```text
/proc/<pid>/cgroup
/sys/fs/cgroup/**/cpuset.cpus
/sys/fs/cgroup/**/cpuset.cpus.effective
/sys/fs/cgroup/**/cpuset.mems
/sys/fs/cgroup/**/cpuset.mems.effective
/sys/fs/cgroup/**/cpu.max
/sys/fs/cgroup/**/cpu.stat
/sys/fs/cgroup/**/cpu.cfs_quota_us
/sys/fs/cgroup/**/cpu.cfs_period_us
/sys/fs/cgroup/**/cpuacct.usage
```

### 输出字段

- `cgroup.version`
- `cgroup.process_groups[*].path`
- `cgroup.process_groups[*].cpuset_cpus_effective`
- `cgroup.process_groups[*].cpuset_mems_effective`
- `cgroup.process_groups[*].cpu_max`
- `cgroup.process_groups[*].cpu_quota_us`
- `cgroup.process_groups[*].cpu_period_us`
- `cgroup.process_groups[*].nr_periods`
- `cgroup.process_groups[*].nr_throttled`
- `cgroup.process_groups[*].throttled_usec`

### 失败处理

- 容器中可能无法访问完整 `/sys/fs/cgroup`，记录 `availability.errors`。
- cgroup v1/v2 字段不一致时，只填可用字段。

## 5.4 npu_topology.py

### 目标

采集 NPU 设备列表、逻辑 ID、PCI 标识、NUMA 节点、本地 CPU。

### 数据来源

由于 NPU 平台差异较大，MVP 支持三种方式：

1. 平台命令自动采集。
2. 用户提供映射文件。
3. 用户通过 CLI 参数提供 rank -> NPU 映射，NUMA 信息标记缺失。

### 自动采集候选

```bash
npu-smi info
npu-smi info -t topo
```

如果实际环境命令不同，应通过 adapter 扩展，不在 Agent 层硬编码。

### 用户映射文件格式

```json
{
  "vendor": "ascend",
  "devices": [
    {
      "device_id": "0",
      "pci_bus_id": "0000:81:00.0",
      "numa_node": 0,
      "local_cpus": "0-31,64-95"
    }
  ]
}
```

CLI 示例：

```bash
cd skills/mindstudio-cpu-binding
python scripts/cli.py collect --pid 12345 --device-type npu --out out/snapshot.json
```

### 输出字段

- `npu_topology.vendor`
- `npu_topology.devices[*].device_id`
- `npu_topology.devices[*].pci_bus_id`
- `npu_topology.devices[*].numa_node`
- `npu_topology.devices[*].local_cpus`
- `npu_topology.source`

### 失败处理

- 无法采集时不要猜测 NUMA，写入：

```json
{
  "npu_topology": {
    "vendor": "unknown",
    "devices": [],
    "source": null
  }
}
```

并在 `availability.missing` 中记录 `npu_topology.devices[*].numa_node`。

## 5.5 runtime_env.py

### 目标

采集 PyTorch、Runtime、LLM Serving 和线程池相关配置。

### 数据来源

外部进程可读信息：

```text
/proc/<pid>/environ
/proc/<pid>/cmdline
```

可识别环境变量：

```text
LOCAL_RANK
RANK
WORLD_SIZE
MASTER_ADDR
MASTER_PORT
OMP_NUM_THREADS
MKL_NUM_THREADS
OPENBLAS_NUM_THREADS
GOTO_NUM_THREADS
KMP_AFFINITY
KMP_BLOCKTIME
ASCEND_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES
```

用户补充信息：

```bash
--torch-num-threads 16
--torch-num-interop-threads 2
--dataloader-workers 8
--dataloader-pin-memory true
--dataloader-prefetch-factor 2
```

### 限制

从外部进程通常无法可靠读取：

- `torch.get_num_threads()` 当前值。
- `torch.get_num_interop_threads()` 当前值。
- DataLoader 实例参数。

因此这些字段应优先由用户提供、启动脚本静态分析或应用内探针补充。MVP 外部采集不到时必须标记缺失。

## 5.6 runtime_sample.py

### 目标

低侵入采样 CPU 使用情况和 Top 线程。

### 数据来源

优先：

```bash
pidstat -t -p <pid> 1 <seconds>
mpstat -P ALL 1 <seconds>
```

备用：

```text
/proc/stat
/proc/<pid>/task/<tid>/stat
```

### 输出字段

- `runtime_sample.sample_seconds`
- `runtime_sample.process_cpu_percent_total`
- `runtime_sample.cpu_usage_by_numa`
- `runtime_sample.top_threads`
- `runtime_sample.cpu_migration_observed`

### cpu_migration_observed 判断

MVP 简单判断：同一 TID 在采样窗口中 `current_cpu` 出现多个值，即认为发生迁移。

## 6. Snapshot 生成流程

```text
1. 解析 CLI 参数
2. 初始化 collection 和 workload
3. 采集 system / cpu_topology / numa_topology
4. 采集 npu_topology
5. 采集 processes / threads
6. 采集 cgroup
7. 采集 pytorch env
8. 运行 runtime sample
9. 汇总 availability.missing / errors / warnings
10. 写出 snapshot.json 和 raw_refs
```

## 7. 原始数据目录建议

```text
snapshot-output/
├── snapshot.json
└── raw/
    ├── lscpu.txt
    ├── lscpu-e.txt
    ├── numactl-H.txt
    ├── proc-12345-status.txt
    ├── proc-12345-cmdline.txt
    ├── proc-12345-task/
    │   ├── 12345-status.txt
    │   ├── 12345-stat.txt
    │   └── 12345-comm.txt
    ├── cgroup-12345.txt
    ├── npu-topology.txt
    ├── pidstat.txt
    └── mpstat.txt
```

## 8. 安全边界

MVP 采集器禁止：

- 写 `/proc`、`/sys`、`/sys/fs/cgroup`。
- 执行 `taskset -cp` 修改进程 affinity。
- 执行 `numactl` 启动业务。
- 修改环境变量后重启进程。
- 修改 K8s、Docker、Slurm 配置。
- 使用 ftrace/eBPF/perf。
- 发送数据到外部服务。

## 9. 采集失败策略

采集失败不能导致整个 Snapshot 不可用，除非目标 PID 不存在。

| 失败项 | 行为 |
|--------|------|
| 目标 PID 不存在 | 退出失败。 |
| `numactl` 不存在 | 使用 `/sys`，记录 warning。 |
| `npu-smi` 不存在 | `npu_topology.devices=[]`，记录 missing。 |
| 无权限读取 environ | PyTorch env 标记 partial。 |
| 无法访问 cgroup 文件 | cgroup 标记 partial。 |
| `pidstat` 不存在 | 使用 `/proc` 采样。 |

## 10. MVP 验收标准

采集器第一版完成后应满足：

1. 对单 PID PyTorch/NPU 进程输出合法 Snapshot JSON。
2. 对多 PID rank 进程输出 rank/process/device 映射。
3. 即使缺少 NPU 命令，也能生成带 `availability.missing` 的报告输入。
4. 不需要 root 即可采集大部分字段。
5. 不修改任何系统状态。
6. Agent 能基于输出触发至少以下规则：
   - 进程未绑核。
   - 跨 NUMA 运行。
   - PyTorch 线程池过载。
   - cgroup/cpuset 冲突。
