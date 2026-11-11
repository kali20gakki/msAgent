# 检查域与阈值表（默认表）

> 三级覆盖：本默认表 → `profiles/<机型>.json` → 用户 `--expected`（后者优先级最高）。
> 判定态：OK / WARN / CRIT / FACT（仅报事实）。阈值均为示例值，可被 profile 覆盖。

## A. NUMA 与拓扑（9 项）

| 项 | 检查项 | 数据源 | 判定 |
| - | ----- | ------ | ---- |
| A1 | node 数与内存分布 | /sys/devices/system/node/*/meminfo | node 间可用内存差 > 20% → WARN |
| A2 | distance 矩阵 | /sys/devices/system/node/*/distance | FACT：本地=10，跨 node ≥ 21 标注远端 |
| A3 | SNC 子 NUMA 簇 | /sys/devices/system/node/* 分区特征 | FACT：SNC 开启时输出子簇划分 |
| A4 | SMT 兄弟 + L3 domain | cpu*/topology/{thread_siblings_list,shared_cpu_list} | FACT：兄弟对与 L3 分组 |
| A5 | NPU→NUMA 亲和 | npu-smi info -t topo | FACT：有四分支——有 CPU Affinity → 卡→核集；无亲和且 fabric 为 UB/HCCS_SW → 「无（UB/等距）」（合法形态，不告警不标定）；其余 → 不可得并建议标定 |
| A6 | NIC→NUMA | /sys/class/net/*/device/numa_node | FACT：网卡→node 映射 |
| A7 | 存储→NUMA | /sys/block/*/device/numa_node | FACT：存储→node 映射 |
| A8 | numa_balancing | /proc/sys/kernel/numa_balancing | = 1 且严格绑核场景 → WARN |
| A9 | zone_reclaim_mode | /proc/sys/vm/zone_reclaim_mode | = 1 → WARN |

## B. 绑核与 CPU 隔离（6 项）

| 项 | 检查项 | 侧 | 数据源 | 判定 |
| - | ----- | -- | ------ | ---- |
| B1 | Cpus_allowed_list | 现状 | /proc/<pid>/status | 目标进程允许核 = 全机核集 → WARN |
| B2 | cpuset 截断 | 配置 | cgroup cpuset.cpus.effective vs 期望 | 期望核 ⊄ effective → CRIT；无期望时 FACT |
| B3 | 关键线程亲和 | 现状 | /proc/<pid>/task/*/status | 关键线程与主线程允许核完全重叠 → FACT |
| B4 | SMT 兄弟占用 | 现状 | topology 兄弟表 + 实际占用核 | 业务线程与杂务共享物理核 → FACT |
| B5 | isolcpus 等内核隔离 | 配置 | /proc/cmdline | FACT：隔离核集输出 |
| B6 | 容器可见核 vs 申请核 | 配置 | cpuset.cpus.effective | 可见核数 < 申请核数 → WARN |

## C. 大页与内存（6 项）

| 项 | 检查项 | 数据源 | 判定 |
| - | ----- | ------ | ---- |
| C1 | 大页预留（HugePages） | /proc/meminfo | Total = 0 → WARN（未配）；Total>0 且 Free=0 → WARN（不足） |
| C2 | 透明大页模式（THP） | /sys/kernel/mm/transparent_hugepage/enabled | never → WARN |
| C3 | 透明大页整理（defrag） | .../defrag | always → WARN |
| C4 | swappiness | /proc/sys/vm/swappiness | > 10 且训练大内存 → WARN |
| C5 | swap 设备 | /proc/swaps | 存在活跃 swap → WARN |
| C6 | overcommit | /proc/sys/vm/overcommit_memory | = 2 且 ratio 偏低 → WARN |

## D. 频率与电源（5 项）

| 项 | 检查项 | 数据源 | 判定 |
| - | ----- | ------ | ---- |
| D1 | CPU 频率策略（governor） | scaling_governor | = powersave（省电模式，CPU 跑不满频率）→ CRIT |
| D2 | 当前频率 vs max | scaling_cur_freq vs cpuinfo_max_freq | < 80% max → WARN |
| D3 | 能效偏好（EPP） | energy_performance_preference | power/balance_power（偏省电）→ WARN |
| D4 | CPU 空闲睡眠深度（C-state） | cpuidle/state*/name | 允许 C6+ 深睡（唤醒延迟大）→ WARN |
| D5 | 温度 | sensors（缺失时 FACT） | 触及阈值 → WARN |

## E. 中断与软中断（7 项）

| 项 | 检查项 | 数据源 | 判定 |
| - | ----- | ------ | ---- |
| E1 | 每核中断分布 | /proc/interrupts | 单核 > 30% 或 CPU0 > 50% → WARN |
| E2 | 业务核 IRQ 占比 | interrupts + Cpus_allowed_list | > 20% → WARN |
| E3 | irqbalance 状态 | ps | 未运行且分布不均 → WARN |
| E4 | IRQ smp_affinity | /proc/irq/*/smp_affinity_list | FACT：亲和集输出 |
| E5 | 网卡队列数 | /sys/class/net/*/queues | 合并队列 < 绑定核数 → WARN |
| E6 | 收包分流（RPS/RFS） | queues/*/rps_cpus | 未配置且多核收包 → WARN |
| E7 | softirq 分布 | /proc/softirqs | NET_RX/TIMER 集中在少数核 → WARN |

## F. cgroup 与配额（5 项）

| 项 | 检查项 | 数据源 | 判定 |
| - | ----- | ------ | ---- |
| F1 | CPU 配额 | cpu.max / cpu.cfs_quota_us | quota < 申请 CPU 数 → WARN |
| F2 | 限流计数（throttled） | cpu.stat | 累计 > 0 → WARN |
| F3 | cpuset.mems 限定 | cpuset.mems.effective | FACT：mems 限定输出 |
| F4 | cpu.weight | cpu.weight | 显著低于同机其他作业 → WARN |
| F5 | memory.max | memory.max vs RSS | max ≤ RSS → WARN |

## G. 运行时环境变量（6 项，事中）

| 项 | 检查项 | 数据源 | 判定 |
| - | ----- | ------ | ---- |
| G1 | OMP_NUM_THREADS | /proc/<pid>/environ | > 可用核数 → WARN |
| G2 | OMP_PROC_BIND/PLACES | environ | 已绑核但 PROC_BIND=false → WARN |
| G3 | ASCEND_RT_VISIBLE_DEVICES | environ | 与期望卡集不符 → WARN |
| G4 | HCCL_* 线程参数 | environ | 与分配核数不匹配 → WARN |
| G5 | MALLOC_ARENA_MAX | environ | 未设 → INFO（FACT 降级） |
| G6 | TBB/GE 线程数 | environ | > 可用核 → WARN |

## H. NPU 适配（4 项）

| 项 | 检查项 | 数据源 | 判定 |
| - | ----- | ------ | ---- |
| H1 | 设备可见数（双口径） | npu-smi info -l（封装权威）/ -m（die 映射）/ /dev/davinci* | 封装数 ≠ 申请卡数 → CRIT；逻辑 die 数与 davinci 设备数并列事实（双 die 机型 8 封装 = 16 die） |
| H3 | 驱动/CANN 版本 | npu-smi info | FACT：版本输出 |
| H4 | 互联拓扑与 fabric | npu-smi info -t topo | FACT：双形态（NPUx 封装级 / Phy-ID die 级矩阵）+ fabric（UB/HCCS_SW/HCCS/PCIe）+ 同封装 SIO die 对 |
| H5 | NPU 频率/温度 | npu-smi | 触及阈值或频率明显低于标称 → WARN |

**双 die 与 UB 等距说明**：
- 双 die 机型（如 910C/A3）一颗物理封装含 2 个 die，软件设备按 die 编号（davinci0-15）；绑核时同封装 die 对（SIO 片内直连）应放同一 NUMA node，跨 node 拆分会让 die 间协同走远端。
- UB（灵衢）/HCCS_SW 交换互联机型，NPU 与所有 CPU 等距，**本来就没有** NPU→NUMA 亲和——A5 输出「无（UB/等距）」属正常形态，不需要标定；绑核按 node 均衡分配（global_slice）。

## quick 档

A1/A2/A5、B1/B2、C1/C2、D1、E1/E2、F1/F2、G1/G2、H1/H4 + 报告元信息。

## 收益定级（高/中/低）

- 高：直接消除 CRIT 级高危且修复路径确定（如 CPU 频率策略为 powersave、cpuset 截断）；
- 中：有改善但幅度依赖负载（大页、THP、E 域中断）；
- 低：方向性建议。
