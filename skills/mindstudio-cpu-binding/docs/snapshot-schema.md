# Host CPU Snapshot Schema

## 1. 目标

Host CPU Snapshot 是 `mindstudio-cpu-binding` MVP 的核心数据契约。它把采集脚本获得的 CPU/NUMA/NPU/进程线程/cgroup/PyTorch 信息整理成统一 JSON，使 Agent、Skill、MCP 或后续插件都可以基于同一份结构化输入生成诊断报告。

设计原则：

- **只读优先**：MVP 只采集状态，不修改系统。
- **证据优先**：每个诊断结论都应能追溯到 Snapshot 字段。
- **缺失可表达**：拿不到的数据用 `null`、空数组或 `availability` 说明，不由 Agent 猜测。
- **NPU + PyTorch 优先**：字段优先覆盖 NPU 训练/推理、PyTorch rank、DataLoader、线程池和环境变量。
- **可演进**：MVP 用脚本生成，后续可由 MCP 工具直接返回同样结构。

## 2. 顶层结构

```json
{
  "schema_version": "0.1.0",
  "collection": {},
  "workload": {},
  "system": {},
  "cpu_topology": {},
  "numa_topology": {},
  "npu_topology": {},
  "processes": [],
  "cgroup": {},
  "pytorch": {},
  "key_processes": {},
  "runtime_sample": {},
  "baseline_metrics": {},
  "availability": {},
  "raw_refs": {}
}
```

## 3. collection

描述采集动作本身。

```json
{
  "collection": {
    "timestamp": "2026-05-26T10:30:00+08:00",
    "duration_seconds": 10,
    "collector_name": "mindstudio-cpu-binding-collector",
    "collector_version": "0.1.0",
    "hostname": "node-001",
    "user": "runner",
    "privilege": "user",
    "command": "python collect.py --pid 12345 --scenario training --framework pytorch --device-type npu",
    "warnings": [
      "numactl command not found; NUMA memory details unavailable"
    ]
  }
}
```

字段说明：

| 字段 | 类型 | 说明 |
|------|------|------|
| `timestamp` | string | 采集开始时间。 |
| `duration_seconds` | number | 采样持续时间。 |
| `collector_name` | string | 采集器名称。 |
| `collector_version` | string | 采集器版本。 |
| `hostname` | string | 节点名。 |
| `privilege` | string | `user` / `root` / `container-user`。 |
| `warnings` | array | 采集阶段的非致命问题。 |

## 4. workload

描述用户声明的工作负载意图。它不是采集事实，但会影响诊断规则。

```json
{
  "workload": {
    "scenario": "training",
    "framework": "pytorch",
    "device_type": "npu",
    "optimization_goal": "throughput",
    "target_pids": [12345, 12346],
    "process_model": "multi-rank",
    "rank_mapping": [
      {
        "rank": 0,
        "pid": 12345,
        "npu_device": "0"
      },
      {
        "rank": 1,
        "pid": 12346,
        "npu_device": "1"
      }
    ],
    "deployment": {
      "environment": "baremetal",
      "container_id": null,
      "kubernetes": null,
      "slurm": null
    }
  }
}
```

字段建议值：

| 字段 | 建议值 |
|------|--------|
| `scenario` | `training` / `inference` / `unknown` |
| `optimization_goal` | `throughput` / `latency` / `stability` / `isolation` / `unknown` |
| `process_model` | `single-process` / `multi-rank` / `multi-instance` / `unknown` |
| `deployment.environment` | `baremetal` / `docker` / `kubernetes` / `slurm` / `unknown` |

## 5. system

描述操作系统和机器级信息。

```json
{
  "system": {
    "os": "Linux",
    "kernel": "5.15.0-xx",
    "architecture": "aarch64",
    "cpu_model": "Kunpeng/Intel/AMD/...",
    "total_logical_cpus": 128,
    "total_physical_cores": 64,
    "sockets": 2,
    "smt_enabled": true,
    "online_cpus": "0-127",
    "isolated_cpus": null
  }
}
```

## 6. cpu_topology

描述 CPU、core、socket、NUMA、SMT sibling 关系。

```json
{
  "cpu_topology": {
    "cpus": [
      {
        "cpu": 0,
        "core_id": 0,
        "socket_id": 0,
        "numa_node": 0,
        "online": true,
        "max_mhz": 3000.0,
        "current_mhz": 2800.0,
        "smt_siblings": [0, 64]
      }
    ],
    "physical_cores": [
      {
        "core_key": "socket0-core0",
        "socket_id": 0,
        "core_id": 0,
        "numa_node": 0,
        "logical_cpus": [0, 64]
      }
    ]
  }
}
```

## 7. numa_topology

描述 NUMA 节点 CPU 和内存。

```json
{
  "numa_topology": {
    "nodes": [
      {
        "node": 0,
        "cpus": "0-31,64-95",
        "logical_cpu_count": 64,
        "physical_core_count": 32,
        "memory_total_mb": 262144,
        "memory_free_mb": 180000
      },
      {
        "node": 1,
        "cpus": "32-63,96-127",
        "logical_cpu_count": 64,
        "physical_core_count": 32,
        "memory_total_mb": 262144,
        "memory_free_mb": 175000
      }
    ],
    "distance_matrix": [
      [10, 20],
      [20, 10]
    ]
  }
}
```

## 8. npu_topology

描述 NPU 与 NUMA locality。不同厂商命令不同，MVP 允许通过平台适配器填充。当前原型通过 `scripts/topology_collect.py` 解析 `npu-smi info -t topo`，将 NPU 间互联统一写为 `links[*].target` 和 `links[*].type`，其中 `type` 可为 `HCCS`、`HCCS_SW`、`PIX`、`PXB`、`PHB`、`SYS`、`SIO` 或 `NA`。

```json
{
  "npu_topology": {
    "vendor": "ascend",
    "devices": [
      {
        "device_id": "0",
        "logical_id": "0",
        "pci_bus_id": "0000:81:00.0",
        "numa_node": 0,
        "local_cpus": "0-31,64-95",
        "health": "ok",
        "links": [
          {
            "target": "1",
            "type": "HCCS",
            "status": "ok"
          }
        ]
      }
    ],
    "source": "npu-smi-or-platform-adapter"
  }
}
```

如果暂时无法获取 NPU NUMA 信息：

```json
{
  "npu_topology": {
    "vendor": "unknown",
    "devices": [],
    "source": null
  }
}
```

并在 `availability.missing` 中记录。

## 9. processes

描述目标进程和线程。MVP 至少采集目标 PID，训练多 rank 时采集所有 rank PID。

```json
{
  "processes": [
    {
      "pid": 12345,
      "ppid": 12000,
      "command": "python train.py --local_rank=0",
      "comm": "python",
      "state": "running",
      "rank": 0,
      "npu_device": "0",
      "cpus_allowed_list": "0-127",
      "mems_allowed_list": "0-1",
      "current_cpu": 48,
      "num_threads": 96,
      "voluntary_ctxt_switches": 1200,
      "nonvoluntary_ctxt_switches": 340,
      "cpu_percent": 650.0,
      "threads": [
        {
          "tid": 12345,
          "name": "python",
          "state": "running",
          "cpus_allowed_list": "0-127",
          "mems_allowed_list": "0-1",
          "current_cpu": 48,
          "numa_node": 1,
          "cpu_percent": 90.0,
          "voluntary_ctxt_switches": 100,
          "nonvoluntary_ctxt_switches": 30,
          "role_hint": "main"
        },
        {
          "tid": 12410,
          "name": "DataLoader",
          "state": "sleeping",
          "cpus_allowed_list": "0-127",
          "mems_allowed_list": "0-1",
          "current_cpu": 5,
          "numa_node": 0,
          "cpu_percent": 45.0,
          "voluntary_ctxt_switches": 500,
          "nonvoluntary_ctxt_switches": 80,
          "role_hint": "dataloader"
        }
      ]
    }
  ]
}
```

`role_hint` 可由线程名、命令行、用户输入或 PyTorch 环境推断，允许为 `unknown`。

建议值：

- `main`
- `dataloader`
- `runtime`
- `communication`
- `blas_worker`
- `openmp_worker`
- `inference_worker`
- `unknown`

## 10. cgroup

描述目标进程所在 cgroup 的 CPU 限制。

```json
{
  "cgroup": {
    "version": "v2",
    "process_groups": [
      {
        "pid": 12345,
        "path": "/sys/fs/cgroup/user.slice/...",
        "cpuset_cpus_effective": "0-31",
        "cpuset_mems_effective": "0",
        "cpu_max": "max 100000",
        "cpu_quota_us": null,
        "cpu_period_us": null,
        "cpu_weight": 100,
        "nr_periods": 10000,
        "nr_throttled": 0,
        "throttled_usec": 0
      }
    ]
  }
}
```

对于 cgroup v1：

```json
{
  "cgroup": {
    "version": "v1",
    "process_groups": [
      {
        "pid": 12345,
        "cpuset_cpus_effective": "0-31",
        "cpuset_mems_effective": "0",
        "cpu_quota_us": -1,
        "cpu_period_us": 100000,
        "nr_periods": 10000,
        "nr_throttled": 10,
        "throttled_usec": 500000
      }
    ]
  }
}
```

## 11. pytorch

描述 PyTorch、NPU 后端和线程相关配置。

```json
{
  "pytorch": {
    "detected": true,
    "version": "2.x",
    "npu_backend": {
      "detected": true,
      "name": "torch_npu",
      "version": "x.y.z"
    },
    "distributed": {
      "enabled": true,
      "world_size": 8,
      "rank": 0,
      "local_rank": 0,
      "backend": "hccl"
    },
    "threading": {
      "torch_num_threads": 16,
      "torch_num_interop_threads": 2,
      "omp_num_threads": "16",
      "mkl_num_threads": null,
      "openblas_num_threads": null,
      "kmp_affinity": null,
      "kmp_blocktime": null
    },
    "dataloader": {
      "num_workers": 8,
      "pin_memory": true,
      "prefetch_factor": 2,
      "persistent_workers": true,
      "source": "user-provided-or-detected"
    },
    "env": {
      "LOCAL_RANK": "0",
      "RANK": "0",
      "WORLD_SIZE": "8",
      "OMP_NUM_THREADS": "16"
    }
  }
}
```

MVP 中部分 PyTorch 运行时字段可能无法从外部进程可靠读取，可通过用户补充或启动命令解析填充。无法确认时写 `null`，并加入信息缺口。

## 12. key_processes

描述对诊断有特殊意义的关键进程和线程，便于报告把主调度线程、SQ 线程、通信线程、DataLoader 线程和高 CPU 线程单独展示。

```json
{
  "key_processes": {
    "discovery_sources": ["target_pids", "npu-smi", "sq_pattern"],
    "main_scheduler_pids": [12345],
    "sq_task_threads": [
      {
        "pid": 3830,
        "tid": 3830,
        "name": "dev0_sq_task",
        "npu_id": 0
      }
    ],
    "npu_fixed_threads": [],
    "dataloader_threads": [
      {
        "pid": 12345,
        "tid": 12410,
        "name": "DataLoader",
        "cpu_percent": 45.0
      }
    ],
    "communication_threads": [
      {
        "pid": 12346,
        "tid": 12510,
        "name": "hccl",
        "cpu_percent": 35.0
      }
    ],
    "top_threads": [
      {
        "pid": 12345,
        "tid": 12345,
        "name": "python",
        "key_class": "main_scheduler",
        "cpu_percent": 90.0
      }
    ],
    "npu_smi_host_pids": [12345, 12346],
    "user_extra_matches": []
  }
}
```

`key_processes` 是派生索引，不替代 `processes[*].threads` 和 `runtime_sample.top_threads`。报告应优先用它解释“哪些线程值得关注”，再回链到原始进程/线程字段作为证据。

## 13. runtime_sample

描述短时间采样结果。MVP 使用轻量采样，不依赖 perf/ftrace。

```json
{
  "runtime_sample": {
    "sample_seconds": 10,
    "process_cpu_percent_total": 650.0,
    "system_loadavg": [12.5, 10.2, 8.8],
    "cpu_usage_by_numa": [
      {
        "node": 0,
        "avg_cpu_percent": 72.0,
        "max_cpu_percent": 98.0,
        "busy_cpus": "0-15"
      },
      {
        "node": 1,
        "avg_cpu_percent": 35.0,
        "max_cpu_percent": 80.0,
        "busy_cpus": "48-52"
      }
    ],
    "top_threads": [
      {
        "pid": 12345,
        "tid": 12345,
        "name": "python",
        "cpu_percent": 90.0,
        "current_cpu": 48,
        "numa_node": 1
      }
    ],
    "cpu_migration_observed": true,
    "notes": []
  }
}
```

## 14. baseline_metrics

性能基线由用户或外部 benchmark 提供，不要求采集脚本自动获得。

```json
{
  "baseline_metrics": {
    "training": {
      "throughput_samples_per_second": 1200.5,
      "step_time_ms_avg": 220.0,
      "step_time_ms_p50": 210.0,
      "step_time_ms_p90": 260.0,
      "step_time_ms_p99": 310.0
    },
    "inference": null,
    "device": {
      "npu_utilization_percent_avg": 78.0,
      "npu_utilization_percent_min": 40.0
    }
  }
}
```

## 15. availability

描述采集完整性，避免 Agent 过度推断。

```json
{
  "availability": {
    "complete": false,
    "missing": [
      "npu_topology.numa_node",
      "pytorch.dataloader.num_workers",
      "baseline_metrics.training.step_time_ms_p99"
    ],
    "partial": [
      "runtime_sample.cpu_usage_by_numa"
    ],
    "errors": [
      {
        "component": "npu_topology",
        "message": "npu-smi command not found"
      }
    ]
  }
}
```

## 16. raw_refs

保留原始命令输出文件路径或摘要，便于追溯。

```json
{
  "raw_refs": {
    "lscpu_e": "raw/lscpu-e.txt",
    "numactl_H": "raw/numactl-H.txt",
    "proc_status": "raw/proc-12345-status.txt",
    "thread_status_dir": "raw/proc-12345-task/",
    "cgroup": "raw/cgroup-12345.txt",
    "npu_topology": "raw/npu-topology.txt"
  }
}
```

## 17. MVP 最小必需字段

要生成第一版可用报告，最少需要：

```text
workload.scenario
workload.optimization_goal
workload.target_pids
system.total_logical_cpus
system.smt_enabled
numa_topology.nodes[*].cpus
npu_topology.devices[*].numa_node 或明确缺失
processes[*].pid
processes[*].cpus_allowed_list
processes[*].mems_allowed_list
processes[*].threads[*].current_cpu
processes[*].threads[*].cpu_percent
cgroup.process_groups[*].cpuset_cpus_effective
pytorch.threading.omp_num_threads
pytorch.dataloader.num_workers 或明确缺失
runtime_sample.top_threads
availability
```

如果缺少 NPU -> NUMA 或 rank -> NPU 映射，报告仍可判断“是否未绑核/是否受 cgroup 限制/线程数是否过载”，但不能下结论说“Rank 与 NPU NUMA 不匹配”。
