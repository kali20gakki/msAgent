# 权限降级与沙箱策略

## 1. 数据来源（不依赖外部工具）

本 skill 的拓扑、设备亲和与进程数据全部来自 sysfs / procfs / npu-smi，不需要任何额外安装的工具：

| 目标 | 主来源 | 降级来源 |
| --- | --- | --- |
| NUMA node 与 CPU 列表 | `/sys/devices/system/node/node*/cpulist` | CPU 侧 `node*` 链接；无 node 目录时退化为单 node |
| node 距离 | `/sys/devices/system/node/node*/distance` | 缺失时按同 node=10、跨 node=20 估算 |
| node 内存 | `/sys/devices/system/node/node*/meminfo` | 缺失时留空，不影响分核 |
| NPU→CPU 亲和 | `npu-smi info -t topo` 的 CPU Affinity | PCI `local_cpulist` > PCI `numa_node` |
| NPU→node | 由 NPU 的 CPU 亲和反推 | `/sys/bus/pci/devices/<bus_id>/numa_node` |
| NIC→node | `/sys/class/net/<if>/device/numa_node` | PCI `numa_node` |
| 存储→node | `/sys/block/<dev>/device/numa_node` | PCI `numa_node` |
| 进程 CPU 亲和 | `/proc/<pid>/status` `Cpus_allowed_list` | 无（必读） |
| 进程允许内存 node | `Mems_allowed_list` / cgroup `cpuset.mems.effective` | 无 |
| 进程实际页分布 | `/proc/<pid>/numa_maps` | 最近运行 CPU（弱证据） |
| 进程最近运行 CPU | `/proc/<pid>/stat` 第 39 字段 | 无 |

`numa_maps` 不可读时：
- 远端页占比标 `not_observable`；
- 用 `Cpus_allowed_list` + `last_cpu` 作为弱证据，标 `evidence_level=affinity+last_cpu`；
- 不伪造远端页比例。

## 2. 普通用户登录没有权限

### 可读 / 不可读

| 数据 | 普通用户（非目标 owner） | 说明 |
| --- | --- | --- |
| `/proc/<pid>/status` | 通常可读 | `Cpus_allowed_list` / `Mems_allowed_list` / `Uid` |
| `/proc/<pid>/cmdline` | 通常可读 | hidepid 时不可读 |
| `/proc/<pid>/cgroup` | 通常可读 | 容器 v1/v2 |
| `/proc/<pid>/task/*/comm` | 通常可读 | 线程角色 |
| `/proc/<pid>/task/*/stat` | 通常可读 | `last_cpu` 弱证据 |
| `/proc/<pid>/numa_maps` | 通常不可读 | 需同用户或 root；标 `view_limited` |
| `/sys/fs/cgroup/.../cpuset.*` | 可能不可读 | 读取失败时留空 |
| `taskset -pc` 他人进程 | 不可写 | 需 root / `CAP_SYS_NICE` |
| `npu-smi info` | 可能需 root | 失败时区分权限/未安装/无设备 |

### 降级策略

- 不因普通用户权限直接失败；能读多少读多少，plan 标 `privilege` 与 `evidence_level`；
- `numa_maps` 不可读 → `view_limited`，远端页占比 `not_observable`；
- 目标进程非当前用户 → `cross_user_control`，`requires_privilege=true`，`execution_mode=delegated`；
- `taskset` 只生成命令，不自动执行；由用户/管理员确认后执行；
- `npu-smi` 权限不足 → `status=blocked`，建议 `sudo` 或加入 `HwHiAiUser` 组，本 skill 不自动提权；
- `proc_hidden`（hidepid）→ 标 `view_limited`，必要时只用 `npu-smi` 卡↔进程做拓扑级方案。

### plan 字段

```json
"privilege": {
  "current_user": "<your-user>",
  "current_uid": 1000,
  "is_root": false,
  "target_uids": [0],
  "own_processes": false,
  "can_read_numa_maps": false,
  "can_taskset": false,
  "requires_privilege": true,
  "execution_mode": "delegated",
  "notes": ["目标进程属于其他用户/root；需 sudo 或同用户执行"]
}
```

`execution_mode`：
- `self`：目标都是当前用户进程，`taskset` 可直接执行；
- `delegated`：目标含其他用户/root，命令交给管理员；
- `plan_only`：事前无进程，只出预规划。

## 3. 沙箱：安全前提下减少确认

### 风险分级与确认矩阵

| 级别 | 操作 | 示例 | 确认 |
| --- | --- | --- | --- |
| R0 | 本机只读 | `/proc`、`/sys`、`npu-smi info`、`lscpu`、`ip`、`lsblk` | 自动允许 |
| R1 | 远端只读 | SSH 执行 `npu-smi info`、`cat`、`lscpu` | 主机级一次授权 + 只读白名单 |
| R2 | 本地方案写 | `plan.py --out ./out` | 工作区内自动允许 |
| R3 | 远端临时写 | 上传 bundle 到 `/tmp/bindtune-<random>`（仅示意；运行产物由 `plan.py --out` 指定） | 每次会话一次确认，校验和 + 用后删除 |
| R4 | 变更执行 | `taskset`、cgroup、sysctl | 永不自动；逐 plan 显式确认 |

### 一次性配置（减少重复确认）

1. **SSH key + ssh-agent**：`ssh-keygen -t ed25519` + `ssh-copy-id`，不再使用密码/askpass 文件；
2. **连接复用**：`~/.ssh/config`
   ```sshconfig
   Host a2
     HostName 192.0.2.10
     User <user>
     IdentityFile ~/.ssh/id_ed25519
     ControlMaster auto
     ControlPath ~/.ssh/cm-%r@%h:%p
     ControlPersist 10m
   ```
3. **主机指纹固定**：首次连接核对 fingerprint 后写入 `known_hosts`；
4. **只读包装**：`remote_readonly.sh` 只接受白名单命令（`npu-smi info`、`lscpu`、`cat /proc/...`、`cat /sys/...`），拒绝写操作；
5. **单次批量采集**：一次 SSH 调用跑完所有只读采集，避免逐条命令确认；
6. **验证用 dry-run**：`plan.py --dry-run --print-plan` 不落盘、不执行，最安全；
7. **单次 bundle**：需要完整运行时，只上传一个 tar 包到 `/tmp/bindtune-<random>`，一次确认，跑完清理。

### 安全边界

- 本 skill 默认 plan-only；`taskset` / cgroup / sysctl 永不自动执行；
- 执行前校验 pid 存在、进程归属、`before` 原值；
- 不使用 root 做只读采集；不在日志/文档中写密码；
- 变更命令与回滚命令成对输出，交主会话确认后执行；
- 建议沙箱只放行特定主机与工作区，不要使用 `ssh *` 通配。
