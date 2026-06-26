---
name: ascend-npu-snapshot-analyzer
description: 分析 PyTorch memory snapshot pickle 文件（_dump_snapshot 导出的内存快照），提供内存峰值、碎片、泄漏、OOM 检测与交互式 HTML 报告。仅当用户明确提及 memory snapshot、内存快照、_dump_snapshot、pickle 内存文件 或需要对 snapshot pickle 做内存分析时才触发，不处理一般的 NPU 性能分析、SQLite 查询或通用内存问题。
keywords: [memory snapshot, 内存快照, _dump_snapshot, _record_memory_history, snapshot pickle, 内存碎片, memory leak NPU, OOM snapshot, segment block]
---

# Ascend NPU Memory Snapshot Analyzer

## 1. 技能目标

分析 `torch_npu.npu.memory._dump_snapshot()` 导出的 pickle 文件，提供多维度的内存分析能力和交互式可视化报告。

## 2. 输入规范

- **输入数据**：`torch_npu.npu.memory._dump_snapshot()` 或 `torch.cuda.memory._dump_snapshot()` 导出的 pickle 文件（支持 dict 和 list 两种格式）
- **前置条件**：需要先通过 `snapshot_to_db.py` 将 pickle 转换为 SQLite DB

## 3. 分析能力

### Track A：快速通道（CTE 宏）

以下 CTE 宏可直接嵌入 SQL 查询，覆盖 80% 常见分析场景：

```sql
-- 宏1：设备内存概览
WITH device_overview AS (
    SELECT
        d.device_index,
        SUM(s.total_size)      AS reserved_bytes,
        SUM(s.allocated_size)  AS allocated_bytes,
        SUM(s.active_size)     AS active_bytes,
        ROUND((SUM(s.total_size) - SUM(s.allocated_size)) * 100.0 / NULLIF(SUM(s.total_size), 0), 2) AS frag_pct,
        COUNT(s.id)            AS segment_count,
        SUM(s.is_expandable)   AS expandable_segments
    FROM segments s
    JOIN devices d ON s.device_id = d.id
    GROUP BY d.device_index
)

-- 宏2：块状态分布
WITH block_state_dist AS (
    SELECT
        d.device_index,
        b.state,
        COUNT(b.id)      AS block_count,
        SUM(b.size)      AS total_size,
        AVG(b.size)      AS avg_size
    FROM blocks b
    JOIN segments s ON b.segment_id = s.id
    JOIN devices d ON s.device_id = d.id
    GROUP BY d.device_index, b.state
)

-- 宏3：扩容事件时间线
WITH expansion_timeline AS (
    SELECT
        d.device_index,
        t.trace_index,
        t.action,
        t.size,
        t.addr,
        cs.frames_json
    FROM traces t
    JOIN devices d ON t.device_id = d.id
    LEFT JOIN call_stacks cs ON t.stack_id = cs.id
    WHERE t.action IN ('segment_alloc', 'segment_map', 'segment_free', 'segment_unmap')
    ORDER BY d.device_index, t.trace_index
)

-- 宏4：大块分配 TOP N
WITH top_allocations AS (
    SELECT
        d.device_index,
        b.size,
        b.requested_size,
        b.state,
        cs.frames_json
    FROM blocks b
    JOIN segments s ON b.segment_id = s.id
    JOIN devices d ON s.device_id = d.id
    LEFT JOIN call_stacks cs ON b.stack_id = cs.id
    WHERE b.state = 'active_allocated'
    ORDER BY b.size DESC
    LIMIT 20
)
```

### Track B：深度分析（脚本调用）

```bash
# 总体概览
python scripts/snapshot_analyze.py snapshot.db --mode overview

# 峰值分析（时序重放 + 峰值贡献者）
python scripts/snapshot_analyze.py snapshot.db --mode peak

# 碎片分析（整体 + 逐段 + 假性碎片）
python scripts/snapshot_analyze.py snapshot.db --mode fragment

# 泄漏检测（单调增长 + 长生命周期 + 堆栈归因）
python scripts/snapshot_analyze.py snapshot.db --mode leak

# OOM 分析（上下文回溯 + 根因推断）
python scripts/snapshot_analyze.py snapshot.db --mode oom

# 跨快照对比（ATTACH DATABASE）
python scripts/snapshot_analyze.py snapshot.db --mode compare --ref other.db

# 全模式 + HTML 报告
python scripts/snapshot_analyze.py snapshot.db --mode all -o report.html
```

## 4. 工作流

### 初始分析流程

1. 用户提供 snapshot pickle 文件路径
2. **转换**: `python scripts/snapshot_to_db.py snapshot.pkl`
3. **总览**: `python scripts/snapshot_analyze.py snapshot.db --mode overview`
4. 根据总览结果，按需执行深度分析

### 问题诊断流程

```
用户问题
    │
    ├─ "峰值/最高内存" → --mode peak
    ├─ "碎片/内存利用率" → --mode fragment
    ├─ "泄漏/不释放" → --mode leak
    ├─ "OOM/崩溃" → --mode oom
    ├─ "对比/差异" → --mode compare --ref other.db
    └─ "全面分析" → --mode all -o report.html
```

## 5. 输出规范

### 问题 → 证据 → 建议 三段式

每个分析结果包含：
- **问题描述**：现象是什么
- **数据证据**：具体数值和来源
- **可执行建议**：优先级 + 预期效果 + 参考链接

### 报告模板路由

分析结果按五层金字塔组织：
1. 总览 & 一键结论（健康状态 + 核心指标卡片）
2. 设备详情（各设备对比 + Segment 类型分布）
3. 深度分析（峰值/碎片/泄漏/OOM 按需展开）
4. 堆栈归因（TOP 10 堆栈 + 分配量占比）
5. 优化建议（高/中优先级 + 类别标签 + 预期效果）

详见 `references/analysis_templates.md`。

## 6. 数据格式参考

详见 `references/snapshot_schema.md`。

## 7. 分析方法论

详见 `references/analysis_methodology.md`。

## 8. 脚本工具

| 脚本 | 用途 |
|------|------|
| `scripts/snapshot_to_db.py` | pickle → SQLite DB 转换 |
| `scripts/snapshot_queries.py` | SQL 查询函数库（供 Agent 直接调用） |
| `scripts/snapshot_analyze.py` | 高层分析（6 种模式 + HTML 报告） |

## 9. 使用示例

```bash
# 1. 转换
python scripts/snapshot_to_db.py snapshot_before.pkl

# 2. 总览
python scripts/snapshot_analyze.py snapshot_before.db --mode overview

# 3. 碎片分析
python scripts/snapshot_analyze.py snapshot_before.db --mode fragment

# 4. 泄漏检测
python scripts/snapshot_analyze.py snapshot_before.db --mode leak

# 5. 生成完整报告
python scripts/snapshot_analyze.py snapshot_before.db --mode all -o report.html
```

## 10. 约束与限制

- 仅支持离线 pickle 文件，不支持实时采集
- Phase 1 不解析 device_traces 的时序关联（仅存储）
- 大文件 (>1GB) 导入时建议使用 `--no-indexes` 先导入，后手动建索引