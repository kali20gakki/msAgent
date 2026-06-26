# 分析方法论

本文档定义 Ascend NPU Memory Snapshot 分析中的检测算法、判定规则和阈值定义。

## 一、碎片率定义

### 整体碎片率

```
碎片率 = (Reserved - Allocated) / Reserved × 100%
```

其中：
- **Reserved**：所有 segment 的 `total_size` 之和
- **Allocated**：所有 segment 的 `allocated_size` 之和

### 逐段碎片率

```
逐段碎片率 = (segment.total_size - segment.allocated_size) / segment.total_size × 100%
```

### 碎片构成

| 碎片类型 | 来源 | 计算方式 |
|---------|------|---------|
| 空闲空间 (Free) | 已分配但未使用的 segment 空间 | SUM(total_size) - SUM(allocated_size) |
| 待释放 (Pending) | 状态为 `active_pending_free` 的 block | SUM(blocks.size) WHERE state = 'active_pending_free' |

### 假性碎片判定

若待释放占比 > 50%，说明碎片主因是异步释放延迟，非真正的分配器碎片。

## 二、碎片率阈值

| 等级 | 图标 | 碎片率范围 | 说明 |
|------|------|-----------|------|
| 正常 | 🟢 | < 5% | 内存利用率良好 |
| 偏高 | 🟡 | 5% ~ 15% | 建议关注，可定期调用 `empty_cache()` |
| 严重 | 🔴 | > 15% | 需优化，可能存在分配器策略问题 |

## 三、泄漏检测算法

### 算法 A：单调增长检测

**原理**：沿 traces 表按 trace_index 顺序统计 segment_alloc 和 segment_free 的累积差值。

**判定条件**：
1. 计算 `net_segments = COUNT(segment_alloc) - COUNT(segment_free)` 的累积趋势
2. 若 trend 单调递增且无回落 → 标记为疑似泄漏

**SQL 实现**：
```sql
SELECT
    trace_index,
    SUM(CASE WHEN action = 'segment_alloc' THEN 1 ELSE 0 END) OVER (ORDER BY trace_index) -
    SUM(CASE WHEN action = 'segment_free' THEN 1 ELSE 0 END) OVER (ORDER BY trace_index) AS net_segments
FROM traces
WHERE action IN ('segment_alloc', 'segment_free')
ORDER BY trace_index;
```

### 算法 B：长生命周期检测

**原理**：查找 `active_allocated` 状态且对应 alloc 事件在时间线前 20% 的 block。

**判定条件**：
1. 计算总 trace 事件数 N
2. 查找 block 关联的 alloc 事件的 trace_index < 0.2 × N
3. 且该 block 仍为 `active_allocated` 状态

### 算法 C：堆栈归因

**原理**：将算法 A 和 B 的结果按 stack_id 聚合，按累计大小降序输出。

**规则**：
- 按 `stack_id` 分组，SUM(size) 为累计增长量
- 取 TOP 3 作为泄漏嫌疑输出
- 每个嫌疑附带堆栈帧的关键路径

### 风险等级

| 等级 | 图标 | 条件 |
|------|------|------|
| 高风险 | 🔴 | 单调增长 + 长生命周期 block ≥ 5 |
| 中风险 | 🟡 | 单调增长 或 长生命周期 block ≥ 3 |
| 低风险 | 🟢 | 无明显增长趋势，长生命周期 block < 3 |

## 四、OOM 判定规则

### OOM 事件识别

```sql
SELECT * FROM traces WHERE action = 'oom';
```

### OOM 上下文分析范围

回溯 OOM 事件前 50 条 trace 事件。

### OOM 根因推断

| 条件 | 推断 |
|------|------|
| 前 5 次分配中同一堆栈出现 ≥ 3 次 | 该堆栈为疑似根因 |
| 前 5 次分配累计 > OOM 时 device_free 的 3 倍 | 一连串大分配导致 |
| OOM 时 device_free > 1GB | 单次过大分配请求，非累积泄漏 |

## 五、峰值分析规则

### 峰值定位算法（时序重放）

1. 沿 traces 表按 trace_index 顺序遍历
2. 遇到 segment_alloc → 累加 Reserved/Allocated
3. 遇到 segment_free → 扣减 Reserved/Allocated
4. 记录每个时刻的 (Reserved, Allocated, Active) 三元组
5. 找出最大值及其对应的 trace_index

### 基线定义

trace 序列前 10% 事件对应的内存状态。

### 峰值贡献者排序

按 segment 的 `total_size` 降序排列，取 TOP 5。

## 六、扩容分析规则

### 新增 Segment 判定

B 中存在但 A 中不存在的 `address` 的 segment。

### 扩容 Segment 判定

A 和 B 中 `address` 相同，但 B 的 `total_size` > A 的 `total_size`。

### 扩容频率判定

| Segment 数量 | 评估 |
|-------------|------|
| < 100 | 正常 |
| 100 ~ 200 | 偏多 |
| > 200 | 频繁 |

## 七、健康状态判定

| 状态 | 图标 | 判定条件 |
|------|------|---------|
| 健康 | 🟢 | 所有指标在阈值内，无 OOM |
| 需关注 | 🟡 | 1~2 个指标超阈值，或碎片率 5%~15% |
| 严重 | 🔴 | 3+ 个指标超阈值，或存在 OOM，或碎片率 >15% |

## 八、默认阈值汇总

| 指标 | 🟢 正常 | 🟡 关注 | 🔴 严重 |
|------|---------|---------|---------|
| 碎片率 | < 5% | 5%~15% | > 15% |
| Segment 数量 | < 100 | 100~200 | > 200 |
| OOM 事件 | 0 | - | > 0 |