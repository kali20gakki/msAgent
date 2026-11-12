# 量化规则（两级口径）

两级 = **定性级（配置判型）** + **定量级（互斥时间分解：打扰 / 等待 / 效率三段）**。
阈值引用链与 config-audit 一致：默认表 → 机型 profile → 用户覆盖。

## 1. 定性级：配置判型

- 输入：config-audit 的 `findings.json`（R 规则）+ `actions[]`；
- 判型：存在 `confidence ≥ 0.7` 的 R 规则结论，或存在 `risk=high` 的待改项
  → `config_primary_candidate = true`；
- 影响面：取 actions 的最高 risk 映射（high→高 / medium→中 / low→低 / 无→无）；
- **不硬折算数值**：配置类以"判型 + 影响等级"呈现，不换算成百分比（避免与时间分解重复计量）。

## 2. 定量级：三段互斥时间分解

> 本期仅交付定性级。下表为运行时维口径，数据源随 tracing 技能接入后启用；
> 缺运行时产物时打扰 / 等待均为 `null`，不得以 0 代替。

| 维度 | 计算口径 | 数据来源 |
| ---- | ---- | ---- |
| 打扰 | `union(中断段 ∪ 软中断段 ∪ 超阈 sched 延迟区间) / 观测窗` | 运行时产物 `accounting.disturb_ns` |
| 等待 | `业务线程睡眠区间并集 / 观测窗` | 运行时产物 `wait.total_ns` |
| 效率 | 第一期 `not_observable`；第二期 perf-collect 提供低效活跃时长 | —— |
| 残余 | `100% − (打扰 + 等待 + 效率)` | 计算 |

**互斥保障**：
- 打扰与等待取不同事件域（`prev_state=R` 的被迫切出 vs 非 runnable 睡眠），天然不重叠；
- 区间内部先做并集（`_union_ns`），避免同源重叠重复计量；
- 各自 `min(时长, 观测窗)` 截断，加总不超过 100%；
- busy-wait（用户态自旋 / OMP 自旋）不睡眠 → 不入等待维，随活跃时间归效率维（第二期）。

## 3. 等待维唤醒源归类（运行时数据接入后启用）

| 类别 | waker comm 特征 | 置信度 |
| ---- | ---- | ---- |
| IO 等待 | `kworker*` / `irq/*` / `jbd2` / `kblockd` / `xfsaild` / `kswapd` / `kcompactd` | inference |
| 定时器 | `swapper*` / `migration*` / `rcu*` | inference |
| 网络 | `ksoftirqd*` / `napi*` / `kstr*` | inference |
| 锁/同步 | 其他业务进程 comm（含同进程其他线程） | inference |
| 长阻塞 | 窗尾未再被唤醒（waker 为空，区间以窗尾截断） | inference |
| 上游数据 | 无法与"锁"区分，报告中并入锁/同期待人工判定 | inference |

归类是**相关性推断**：唤醒方 comm 能说明"谁把它叫醒"，不能证明"它为何必须等"。
报告中所有等待类 Finding 必须标 `inference`，不得表述为根因定论。

## 4. 多卡与多线程口径

- **多线程**：等待占比用区间**时间并集**（并行睡眠不重复累计），上限自然 ≤ 100%；
  同时给出 `sum_ns`（合计睡眠时长）供观察并行度，不用于占比；
- **多卡**：主结论取 `disturb_ratio + wait_ratio` 最高的卡（暴露最严重卡），
  其余卡明细在 `phase_verdict_1.json`；报告标注 `basis=card<N>`。

## 5. 残余与未收敛

- 无运行时数据时 `residual_pct = null`，不计算收敛，也不宣称「没有问题」；
- `residual_pct = 100 − (disturb_pct + wait_pct)`（效率维缺位时其时间自动落残余）；
- `residual_pct > 30` → `converged = false`，结论标「未收敛」，**不强行归类**；
- 第二期 perf-collect 交付后，效率维数值化，残余应显著下降；
  若仍 > 30%，提示存在未覆盖的归因维度（需人工介入）。

## 6. 数值呈现

- 占比保留 1 位小数；
- `null` 表示不可观测（而非 0），报告中显示「—」并给出原因；
- 报告数字一律取自 `attribution.json`，不得二次计算（与「报告与 Action 同源」一致）。
