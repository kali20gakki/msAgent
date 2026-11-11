# 绑核机制与命令

> 与旧技能 `mindstudio-cpu-binding` 的分工：旧技能输出绑核诊断与建议（报告式），
> 本技能负责采集与可执行方案（plan / 命令 / 回滚 / 验证）；Host CPU 场景后续由
> `msagent-perf-cpu-*` 系列替代旧技能，迁移期两者并存。

## 机制选型

| 机制 | 时机 | 适用 | 说明 |
| --- | --- | --- | --- |
| `taskset -pc` | 进程已存在（事中） | 单机排障、事后修正 | 立即生效；pid 可能复用，执行前需校验 |
| 启动包装 | 进程诞生前（事前） | 单机/容器 | `taskset` 前缀 + OMP env（`OMP_PROC_BIND` / `OMP_PLACES`） |
| cgroup cpuset | 容器创建时 | 容器/集群 | 进程/线程 fork 均受限，天然批量 |
| 调度器级 | 调度时 | 集群 | k8s CPU Manager / slurm --cpu-bind；本期只出清单，不生成模板 |

## 时机

```text
t0 容器创建    -> cgroup cpuset 圈定容器 CPU 集
t1 entrypoint  -> 启动包装：taskset 前缀 + OMP_PROC_BIND
t2 进程 spawn  -> 线程继承 affinity
t3 特殊线程    -> 仅对通信/IO 线程按 comm 匹配二次细绑
```

## 命令契约

> 版本说明：`apply_plan.py` / `verify_config.py` 属求解层脚本。仅合入采集层时这些脚本不存在，
> 下列命令只作契约参考，待求解层提供后可直接执行。

```bash
# 执行（plan-only 只生成该命令，不执行；需主会话确认）
python3 <skill>/scripts/apply_plan.py <plan.json> --roles comm,io
python3 <skill>/scripts/apply_plan.py <plan.json> --roles comm,io --bind-compute
python3 <skill>/scripts/apply_plan.py <plan.json> --dry-run

# 回滚（按 before.Cpus_allowed_list 恢复主线程与全部线程）
python3 <skill>/scripts/apply_plan.py <plan.json> --rollback

# 校验（apply 前后均可；plan-only 时预期 FAIL）
python3 <skill>/scripts/verify_config.py <plan.json> --threads
```

- Linux 只有线程级亲和：`main` 绑 `tid==pid`；`comm`/`io` 在执行时枚举 `/proc/<pid>/task` 并按 comm 匹配；
- `compute` 默认按 plan 的 `apply=exec` 绑定到 compute 核集（与 `OMP_NUM_THREADS` 一致）；
  旧计划标 `inherit` 时用 `--bind-compute` 强制绑定；
- 目标核先与 cgroup `cpuset.cpus.effective` 取交集，越界部分不会生效；
- plan 不保存 tid 快照，避免执行时线程重建导致绑错；
- 不按 pid 做集群批量；集群走 cpuset/启动包装，避免 pid 复用误绑。

## 验证

- 配置生效：`verify_config.py <plan.json> --threads`（主线程 + 按 comm 枚举线程，与 cgroup 裁剪后的期望比对）
- 单点核对：`awk '/^Cpus_allowed_list:/{print $2}' /proc/<pid>/status`、`/proc/<pid>/task/<tid>/status`
- 运行落核：`tracing --plan <plan.json>`
- 效果：`sched_delay_p99_us`、`remote_page_ratio`、`migration_per_s`
