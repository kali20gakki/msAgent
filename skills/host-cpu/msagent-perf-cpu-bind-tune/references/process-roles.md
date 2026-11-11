# 进程/线程分类规则

## 进程类型

| 类型 | 识别 | 绑核决策 |
| --- | --- | --- |
| 训练主进程 | `npu-smi` 卡占用 + `cmdline` 含 `torchrun` / `msrun` / `verl` / `MindSpeed` | 业务区，按 rank 分核；RL 按 actor/rollout 角色分核 |
| 推理进程 | `npu-smi` 卡占用 + `cmdline` 含 `vllm serve` / `sglang.launch_server` | 业务区，按实例分核；调度/分词线程与计算核分离 |
| DataLoader worker | 主进程 fork，不占卡；comm 与主进程同为 `python`，**当前仅按线程 `comm` 关键字识别，PPID 识别待实现**；未占卡的 worker 不作为独立 unit | 同 node 独立 IO 子核集 |
| HCCL 通信线程 | 线程 `comm` 含 `hccl` | 通信核集（1~2 核） |
| NPU runtime 线程 | `comm` 含 `AOE_RTKB` | 与 comm 同核集；`acl_thread` 归 compute（见下表） |
| OMP 计算线程 | 主进程线程池，受 `OMP_*` 控制 | 计算主体 |
| 监控/采集 | `npu-smi` 自身、`node_exporter`、`telescope` | 干扰隔离区（仅同 cgroup 可迁） |
| 容器 runtime | `containerd-shim` / `runc` | 系统中断区，不主动迁移 |
| 日志/agent | `fluentd` / `filebeat` | 干扰隔离区 |
| 编译/交互 | `gcc` / `pip` / `jupyter` | 告警并建议移出，不自动处理 |

## 线程角色映射（discover.classify_thread）

| 角色 | comm 关键字 | 说明 |
| --- | --- | --- |
| `comm` | `hccl`、`aoe_rtkb` | HCCL 通信 + NPU runtime；独立 1~2 核 |
| `io` | `zmq`、`tokenizer`、`detokenizer`、`dataloader`、`network` | 消息队列/分词/数据加载；独立 1 核 |
| `compute` | `vllm`、`caffe`、`python`、`pt_autograd`、`acl_thread`、`compl`、`omp` | 计算主体 |
| `main` | 其他（`plog`、`trace`、`monitor`、`ras`、`watchdog` 等） | 主线程/运行时管理，保留 1 核 |

## 绑定执行语义

| 角色 | apply | 执行方式 |
| --- | --- | --- |
| main | exec | 绑主线程（`tid == pid`） |
| comm | exec | 执行时枚举 `/proc/<pid>/task`，comm 命中关键字后细绑 |
| io | exec | 同上 |
| compute | exec | 绑定到 compute 核集（与 `OMP_NUM_THREADS` 核数一致，OMP 在该核集内自绑）；旧计划标 `inherit` 时用 `--bind-compute` 强制绑定 |

所有目标核先与 cgroup `cpuset.cpus.effective` 取交集；plan 不保存 tid 快照。

## Primary NPU 判定

`npu-smi info` 中同一 pid 可能出现在多张卡上。按进程内存取最大条目作为 primary NPU：

- 48GB 条目 = 主卡；
- 117MB 条目 = 控制/共享内存，不作为主卡；
- `npus` 保留全部卡列表，便于报告展示。

## 权限降级

- `numa_maps` 不可读（非进程 owner / 容器权限）→ `view_limited`，远端页占比标 `not_observable`；
- `Cpus_allowed_list` 不可读 → 该 unit 跳过，并在报告中标注；
- `npu-smi` 不可用/失败/无设备 → 直接 blocked，不生成 plan。
