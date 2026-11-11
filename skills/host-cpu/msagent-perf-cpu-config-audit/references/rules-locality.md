# 组合诊断规则（R 系列）：多检查项联合判定

> R = Rule 编号。单个检查项只给单点判定，R 规则把多项事实组合成综合结论，
> 命中后输出 Finding 并附 suggested_cpuset。编号在整体设计时一次性分配：
> R003/R004（NUMA 亲和偏差类）已移交 bind-tune，编号保留不复用，
> 因此本文档为 R001/R002/R005~R010 共 8 条。每条定义见下表。

| 编号 | 规则 | 判据（由检查项组合） | 结论 | suggested_cpuset 来源 |
| ---- | ---- | ------------------- | ---- | --------------------- |
| R001 | 进程完全未绑核 | B1 affinity = 全机 且 进程活跃 | 线程全机漂移、cache 失效 | 按 NPU 所在 node 推导 node 核集 |
| R002 | cpuset 截断 | B2 期望核 ⊄ effective | 绑核参数被容器截掉 | effective 与期望的交集 |
| R005 | 数据线程抢占计算核 | dataloader 与计算线程核集相交 | 数据线程抢计算核 | 相交核之外为 dataloader 划独立核 |
| R006 | 通信线程与计算冲突 | HCCL 与计算线程核集相交 且 通信占比高 | 通信延迟、计算被打断 | 相交核外划 1~2 物理核作通信核集 |
| R007 | 中断落在业务核 | E4 亲和相交 且 E2 业务核 IRQ > 20% | 中断打扰业务线程 | 业务核集减去 IRQ 高占比核 |
| R008 | SMT 兄弟冲突 | 关键线程与杂务 thread_siblings 相同 | 物理核执行单元争抢 | 兄弟对中取空闲侧 |
| R009 | 多 rank 核集重叠 | 各 rank cpuset 两两相交 且 核数不足 | rank 间争抢 | 按 node 均分不相交核集 |
| R010 | 隔离核未用 / 未完全绑入 | B5 隔离核 ∩ 业务核 = ∅（隔离核闲置），或业务核 ⊄ 隔离核（部分越界） | 隔离失效或核浪费 | 隔离核集或其补集 |

## Finding 输出要求

每条 R 规则命中输出一个 Finding（8 字段契约的前 4 字段由本 skill 填充）：

```json
{
  "finding_id": "F-R001-<host>-<ts>",
  "problem": "进程完全未绑核",
  "evidence": ["B1: pid 12345 Cpus_allowed_list=0-127", "..."],
  "impact": "线程全机漂移、cache 失效",
  "confidence": 0.9,
  "root_cause": "未配置绑核",
  "actions": [],
  "suggested_cpuset": "0-31"
}
```

Action 一律转出 bind-tune（本 skill 不产出绑核 Action），Finding 必须附 evidence 与
suggested_cpuset。
