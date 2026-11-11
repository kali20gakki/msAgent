---
name: msagent-perf-cpu-bind-tune
description: >-
  昇腾 NPU 多卡任务的 Host CPU 绑核方案求解器：覆盖 PyTorch 训练/离线推理与
  vLLM-Ascend、SGLang 推理服务的 rank/worker/engine 分核，基于 NUMA 拓扑与
  卡↔进程映射输出可执行、可回滚、可验证的绑核计划（taskset 前缀 / cpuset /
  OMP 环境变量），自动判定事前（env 片段）与事中（在线 sched_setaffinity）双形态。
  触发：绑核方案、核怎么分、per-rank 绑核、NUMA 绑定方案、cpuset 划分、多卡分核、
  CPU range 冲突、TTFT 或吞吐受 CPU 影响。
  不用于：绑核没生效/绑核异常诊断（→ perf-cpu-config-audit）；运行期是否落核
  （→ perf-cpu-tracing）；变慢根因归因（→ perf-cpu-perf-triage）。
metadata:
  layer: analysis
---

# msagent-perf-cpu-bind-tune

粗粒度绑核与 NUMA 亲和求解：给定 NPU/CPU 拓扑（事中另含实际业务进程/线程），
输出每个 unit（训练 rank / 推理实例）及角色的核集、命令、回滚、验证脚本与 HTML 报告。

**只出方案，不执行**。`taskset` 等动作由主会话确认后执行；运行期落核验证交 tracing，
效果复检交 perf-triage / config-audit。

## 形态判定（自动，无需用户输入）

1. `preflight` 只探测 `npu-smi` 可执行文件是否存在；
2. `npu_query` 调用 `npu-smi info`，按“任一 NPU 有占用进程”判定：
   - 有任务 → **事中（主场景）**：按实际业务卡↔进程、进程树、线程角色、affinity/numa_maps 求解；
   - 无任务 → **事前（兜底）**：按拓扑 + 框架画像预规划，进程级偏差标 `not_observable`；
   - `npu-smi` 不可用 / 调用失败 / 无 NPU 设备 → **blocked**：不执行绑核分析，不生成 plan/HTML。
3. 同时判定 locality 模式：`npu-smi info -t topo` 是否带 `CPU Affinity` 列；`/sys/bus/ub/devices` 可见则标记 `fabric=ub`；超节点矩阵（`SIO`/`HCCS_SW`）标 `fabric=hccs_supernode`。

## 快速开始

> `apply_plan.py` / `verify_config.py` 属求解层脚本；仅合入采集层时命令不可用，待求解层提供后执行。

```bash
# 事中：A2 上已有 VLLM Worker
# 默认输出到 <workspace>/workspaces/<host>/<timestamp>/
python3 scripts/plan.py

# 显式指定输出目录（旧行为 / CI）
python3 scripts/plan.py --out ./out

# 调预留核（默认 system 2 + interference 1）
python3 scripts/plan.py --irq-reserve 2 --interference-reserve 1

# 查看方案（plan-only，不执行；建议先 --dry-run）
python3 scripts/apply_plan.py <plan.json> --dry-run

# 确认后执行：主线程 + comm/io + compute（按 plan 的 apply 语义）
python3 scripts/apply_plan.py <plan.json> --roles comm,io
# 旧计划标 compute=inherit 时，需要才强制一起绑
python3 scripts/apply_plan.py <plan.json> --roles comm,io --bind-compute

# 回滚（按 before.Cpus_allowed_list 恢复）
python3 scripts/apply_plan.py <plan.json> --rollback

# 现状 vs 目标 gap 分析（plan-only 预期有 FAIL；apply 后应全 PASS）
python3 scripts/verify_config.py <plan.json> --threads
```

## 工作空间与输出

- **skill 代码**：本 SKILL.md 所在目录（`scripts/`、`references/`、`resources/`）。
- **工作空间**：Agent 运行时设置的工作路径（cwd），由宿主或远程执行插件决定；skill 不猜测自身安装位置。
- **运行产物**：`<workspace>/workspaces/<host>/<timestamp>/`，包含 plan / report / verify 脚本；skill 目录只放代码、references、resources，不写运行产物。
- **覆盖方式**：`--out <dir>` 显式指定；`HOSTCPU_WORKSPACE=<root>` 覆盖工作空间根。
- **远端执行**：skill 拷贝到远端何处、以什么 cwd 执行，由远程执行插件控制；skill 只按传入的 cwd/参数落盘，不自行选择远端目录。
- **离线回归测试**：仓库 `tests/skills/msagent-perf-cpu-bind-tune/`（开发期使用，不随 skill 分发）。

输出：

- `hostcpu_plan_<host>_<ts>.json`：plan（units / roles / commands / before / rollback / verify / zones / deviations）
- `hostcpu_verify_<host>_<ts>.sh`：配置生效验证脚本
- `hostcpu_report_<host>_<ts>.html`：NUMA 拓扑与设备卡片、NUMA 亲和矩阵（NPU/NIC/存储 × node，本地/远端时延/不可观测）、node×node 时延矩阵、unit→核分配、角色核分配条、偏差与结论、验证与回滚

## 关键设计

- **NPU 锚点**：`npu-smi info` 的卡↔进程映射是权威锚点；同 pid 多卡条目按进程内存取 primary NPU，避免把 117MB 控制内存误判为主卡。
- **两种绑核策略**（按能力自动选择，写入 `fabric` / `locality_mode`；命名对齐 vllm-ascend CPU Binding）：
  - `topo_affinity`：`npu-smi info -t topo` 带 `CPU Affinity` 且可解析（A2 / A200T A3 Box8 等；A5 上 UB 与亲和列可同时存在）→ 按 NPU 主机口就近核组求解，`per_npu` 每卡独占、`group` 同组 unit 均分；
  - `global_slice`：无 `CPU Affinity`（超节点矩阵 `UB`/`HCCS_SW`/`SIO`）或亲和不可用 → 按封装分组把可见 CPU 池均分到各 node（双 die 机型同封装 die 的 unit 锁定同一 node），切片对齐 NUMA 边界、unit 间不重叠；NIC/存储/IRQ 仍按各自 node 就近。
- **判定按能力而非代际**：只依据 topo 的 `CPU Affinity` 列、关系符号、亲和重复度，以及 sysfs 的 `numa_node` / `local_cpulist` / `/sys/bus/ub/devices`；A2/A3/A5 仅用于预期校验与文案，实测不符只告警、不切策略。不依赖 `npu-smi info -t product`（部分产品/版本无此子命令）。
- **`fabric` 与 `CPU Affinity` 是两条独立信号**：`fabric` 看关系符号（`UB` / `HCCS*` / `SIO` / `PHB`），描述 NPU 间平面；`CPU Affinity` 描述 NPU 主机口到本机 CPU 组的就近关系。A5 可为 `fabric=ub` 且带亲和列，此时仍走 `topo_affinity`。
- **亲和粒度**（`affinity_granularity`）：多个 NPU 共享同一核区间时为 `group`（如 A5 的 NPU0/1/6/7→288-383、NPU2-5→96-191），绑核需在同组 unit 间再均分该区间，不能按单卡独占整段。
- **NPU→CPU 亲和**：直接使用 `npu-smi info -t topo` 的 `CPU Affinity` 列（A2 实机为 0/1→144-167、2/3→96-119、4/5→0-23、6/7→48-71），不假设 NPU 与 node 顺序对应；`Phy-ID` 矩阵按表头判断是否存在该列，不再视作解析失败。
- **按 node 划区**：每个 node 预留 system_irq 与 interference，剩余业务区在同 node 的多个 unit 间均分。
- **角色分核**：compute / comm / io / main；通信与 NPU runtime 独立核，ZMQ/tokenizer/dataloader 等 IO 独立核。
- **local-first**：优先把 unit 放到 NPU 所在 node；若 plan 核数少于当前亲和，标 `capacity_gap`，不静默扩到跨 node。
- **权限降级**：`numa_maps` 不可读时标 `not_observable`，不伪造远端页占比；`npu-smi` 缺失直接 blocked。

## 权限降级 / 普通用户 / 沙箱

- **数据来源**：拓扑与设备亲和全部取自 sysfs（`/sys/devices/system/node`、CPU `node*`、NIC/存储 `numa_node`）+ `npu-smi info -t topo`；NPU→CPU 降级顺序为 `topo CPU Affinity → PCI local_cpulist → PCI numa_node`，不依赖任何额外安装的工具。
- **普通用户权限**：`Cpus_allowed_list`、`Mems_allowed_list`、`cmdline`、`cgroup`、线程 `comm/stat` 通常可读；他人进程 `numa_maps` 不可读时标 `view_limited`，远端页占比 `not_observable`，用 `last_cpu` 作弱证据。`taskset` 他人进程需 sudo → `execution_mode=delegated`，plan 只出命令、不自动执行。
- **apply 属主校验**：`apply_plan.py` 实际执行（非 `--dry-run`）前逐一校验目标进程属主——目标 uid 与当前用户不一致且当前非 root 时拒绝并提示 delegated 路径，防止 plan 生成后进程属主变化导致误绑他人进程；dry-run 放行用于展示动作清单。
- **沙箱与确认**：R0 本机只读自动允许；R1 远端只读一次主机授权 + 只读白名单；R2 本地方案写自动允许；R3 远端临时写一次确认；R4 `taskset`/cgroup/sysctl 永不自动。推荐 SSH key + `ControlMaster` + 单次批量采集 + `plan.py --dry-run --print-plan`，减少重复确认。

## 边界

- 不做配置审计/诊断 → `config-audit`；
- 不做运行期落核验证 → `tracing --plan`；
- 不做效果复检 → `perf-triage` / `config-audit`；
- 不做 IRQ 亲和、大页、频率、cgroup 配额 → `config-audit`；
- 不做 NPU 微架构分析、代码/算子改写；
- 与旧技能 `mindstudio-cpu-binding` 的分工：旧技能输出诊断报告与保守/进阶建议，本技能负责采集与可执行方案（plan / 命令 / 回滚 / 验证）；Host CPU 场景后续由 `msagent-perf-cpu-*` 系列替代旧技能，迁移期两者并存。

## 目录

```
scripts/
  plan.py            # 入口：参数校验 + 流程编排（形态判定在 preflight，可行性在 feasibility）
  preflight.py       # npu-smi 存在性 / 容器 / 工具探测
  npu_query.py       # npu-smi 唯一封装点：设备 / 卡↔进程 / topo
  topology.py        # NUMA / CPU / SMT / L3 / NPU / NIC / 存储
  procfs.py          # status / cmdline / cgroup / affinity / numa_maps / threads
  roles.py           # 线程角色规则唯一来源（comm 关键字 → comm/io/compute/main）
  discover.py        # units + 偏差
  model.py           # unit→node/L3 建模
  zone_alloc.py      # 按 node 划区与业务区均分
  unit_alloc.py      # unit 内角色分核 + thread_binding
  feasibility.py     # C1-C5
  emit_plan.py       # plan / 命令 / 回滚 / 验证脚本
  apply_plan.py      # 主会话确认后执行：按 comm 枚举线程 + cgroup 裁剪 + 回滚
  verify_config.py   # 现状 gap / 生效校验
  report.py          # HTML 报告
  workspace.py       # 工作空间/输出目录解析
references/
  process-roles.md   # 进程/线程分类规则
  framework-profiles.md
  mechanisms.md      # taskset / 启动包装 / cgroup / 调度器
  checks.md          # C1-C5
  permissions.md     # 权限降级 / 普通用户 / 沙箱确认策略
resources/
  plan_template.json
  report_template.html   # 页面骨架（占位符由 report.py 填充）
  report.css             # 报告样式
```
