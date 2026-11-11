# 可行性检查 C1-C5

| 检查 | 判据 | 处理 | 实现状态 |
| --- | --- | --- | --- |
| C1 核预算不足 | unit 可用核 < 3（comm 1 + io 1 + compute 1） | 写入 `unresolved` | 已实现（`feasibility.check`） |
| C2 L3 跨域 | unit 核集不落在同一 L3 domain | `l3_split=true` 告警 | 已实现（`feasibility.check`） |
| C3 unit 核集冲突 | 两个 unit 的 cpuset 相交 | 写入 `unresolved` | 已实现（`feasibility.check`） |
| C4 SMT | 检测到 SMT 且策略为 exclusive | `smt_shared=true` 提示 | 已实现（`feasibility.check`） |
| C5 容量回退 | plan 核数 < 当前亲和核数 | 标 `capacity_gap`；local-first 不静默跨 node | 已实现（`feasibility.check`） |

## 降级链（确定性）

```text
D1 启用 SMT 兄弟（无 SMT 时跳过）
D2 逻辑线程共享（oversubscribed=true，明确不推荐）
D3 跨 node（cross_numa=true，取拓扑距离最近的 node）
D4 仍不足 -> unresolved（check=C1）
```

> 实现状态：D1-D4 降级链**本期未实现**，当前只做 C1-C5 判定并输出告警/`unresolved`，
> 不做自动降级决策；后续版本按上表顺序补齐。

## local-first 原则

- 优先把 unit 放到 NPU 所在 node；
- 同一 node 多个 NPU/unit 时均分业务核；
- 若均分后核数少于当前亲和，保留 `capacity_gap` 告警，不自动扩到跨 node；
- 需要保留 CPU 容量时，由用户显式选择后续 `capacity` 策略（本期未实现）。
