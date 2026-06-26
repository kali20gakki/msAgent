#!/usr/bin/env python3
"""
SQL 查询库 —— 供 snapshot_analyze.py 和 Agent 直接调用的查询函数
"""

import sqlite3
from typing import Any, Dict, List, Optional


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_device_overview(db_path: str) -> List[Dict[str, Any]]:
    """各设备内存概览"""
    conn = _connect(db_path)
    rows = conn.execute("""
        SELECT
            d.device_index,
            SUM(s.total_size)      AS reserved_bytes,
            SUM(s.allocated_size)  AS allocated_bytes,
            SUM(s.active_size)     AS active_bytes,
            ROUND(
                (SUM(s.total_size) - SUM(s.allocated_size)) * 100.0 / NULLIF(SUM(s.total_size), 0), 2
            ) AS frag_pct,
            COUNT(s.id)            AS segment_count,
            SUM(s.is_expandable)   AS expandable_segments
        FROM segments s
        JOIN devices d ON s.device_id = d.id
        GROUP BY d.device_index
        ORDER BY d.device_index
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_block_state_dist(db_path: str, device_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """块状态分布"""
    conn = _connect(db_path)
    query = """
        SELECT
            d.device_index,
            b.state,
            COUNT(b.id)      AS block_count,
            SUM(b.size)      AS total_size,
            AVG(b.size)      AS avg_size
        FROM blocks b
        JOIN segments s ON b.segment_id = s.id
        JOIN devices d ON s.device_id = d.id
    """
    params: tuple = ()
    if device_index is not None:
        query += " WHERE d.device_index = ?"
        params = (device_index,)
    query += " GROUP BY d.device_index, b.state ORDER BY d.device_index, b.state"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_segment_type_dist(db_path: str, device_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """Large/Small 段分布"""
    conn = _connect(db_path)
    query = """
        SELECT
            d.device_index,
            s.segment_type,
            COUNT(s.id)      AS segment_count,
            SUM(s.total_size) AS total_size
        FROM segments s
        JOIN devices d ON s.device_id = d.id
    """
    params: tuple = ()
    if device_index is not None:
        query += " WHERE d.device_index = ?"
        params = (device_index,)
    query += " GROUP BY d.device_index, s.segment_type ORDER BY d.device_index, s.segment_type"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_expansion_events(db_path: str, device_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """扩容事件列表"""
    conn = _connect(db_path)
    query = """
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
    """
    params: tuple = ()
    if device_index is not None:
        query += " AND d.device_index = ?"
        params = (device_index,)
    query += " ORDER BY d.device_index, t.trace_index"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_top_allocations(
    db_path: str, device_index: Optional[int] = None, limit: int = 20
) -> List[Dict[str, Any]]:
    """TOP N 大块分配（堆栈归因）"""
    conn = _connect(db_path)
    query = """
        SELECT
            d.device_index,
            b.size,
            b.requested_size,
            b.state,
            s.address AS segment_address,
            cs.frames_json
        FROM blocks b
        JOIN segments s ON b.segment_id = s.id
        JOIN devices d ON s.device_id = d.id
        LEFT JOIN call_stacks cs ON b.stack_id = cs.id
        WHERE b.state = 'active_allocated'
    """
    params: tuple = ()
    if device_index is not None:
        query += " AND d.device_index = ?"
        params = (device_index,)
    query += " ORDER BY b.size DESC LIMIT ?"
    params = params + (limit,)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_oom_events(db_path: str, device_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """OOM 事件列表"""
    conn = _connect(db_path)
    query = """
        SELECT
            d.device_index,
            t.trace_index,
            t.device_free,
            t.size,
            cs.frames_json
        FROM traces t
        JOIN devices d ON t.device_id = d.id
        LEFT JOIN call_stacks cs ON t.stack_id = cs.id
        WHERE t.action = 'oom'
    """
    params: tuple = ()
    if device_index is not None:
        query += " AND d.device_index = ?"
        params = (device_index,)
    query += " ORDER BY t.trace_index"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trace_range(
    db_path: str, device_index: int, start: int, end: int
) -> List[Dict[str, Any]]:
    """时间窗口事件查询"""
    conn = _connect(db_path)
    rows = conn.execute(
        """
        SELECT
            t.trace_index,
            t.action,
            t.addr,
            t.device_free,
            t.size,
            t.stream,
            cs.frames_json
        FROM traces t
        JOIN devices d ON t.device_id = d.id
        LEFT JOIN call_stacks cs ON t.stack_id = cs.id
        WHERE d.device_index = ? AND t.trace_index BETWEEN ? AND ?
        ORDER BY t.trace_index
        """,
        (device_index, start, end),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_fragmentation_detail(db_path: str, device_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """逐段碎片详情"""
    conn = _connect(db_path)
    query = """
        SELECT
            d.device_index,
            s.address,
            s.total_size,
            s.allocated_size,
            s.active_size,
            ROUND(
                (s.total_size - s.allocated_size) * 100.0 / NULLIF(s.total_size, 0), 2
            ) AS frag_pct,
            s.total_size - s.allocated_size AS waste_bytes,
            s.segment_type,
            s.is_expandable,
            cs.frames_json
        FROM segments s
        JOIN devices d ON s.device_id = d.id
        LEFT JOIN call_stacks cs ON s.stack_id = cs.id
    """
    params: tuple = ()
    if device_index is not None:
        query += " WHERE d.device_index = ?"
        params = (device_index,)
    query += " ORDER BY waste_bytes DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_peak_timeline(db_path: str, device_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """获取时序内存变化，用于峰值分析"""
    conn = _connect(db_path)

    query = """
        SELECT
            d.device_index,
            t.trace_index,
            t.action,
            t.size,
            t.addr
        FROM traces t
        JOIN devices d ON t.device_id = d.id
        WHERE t.action IN ('segment_alloc', 'segment_free', 'segment_map', 'segment_unmap')
    """
    params: tuple = ()
    if device_index is not None:
        query += " AND d.device_index = ?"
        params = (device_index,)
    query += " ORDER BY d.device_index, t.trace_index"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trace_count(db_path: str, device_index: Optional[int] = None) -> int:
    """获取 trace 事件总数"""
    conn = _connect(db_path)
    query = "SELECT COUNT(*) AS cnt FROM traces"
    params: tuple = ()
    if device_index is not None:
        query += " t JOIN devices d ON t.device_id = d.id WHERE d.device_index = ?"
        params = (device_index,)
    row = conn.execute(query, params).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_device_count(db_path: str) -> int:
    """获取设备数量"""
    conn = _connect(db_path)
    row = conn.execute("SELECT COUNT(*) AS cnt FROM devices").fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_segment_count(db_path: str, device_index: Optional[int] = None) -> int:
    """获取 segment 数量"""
    conn = _connect(db_path)
    query = "SELECT COUNT(*) AS cnt FROM segments"
    params: tuple = ()
    if device_index is not None:
        query += " s JOIN devices d ON s.device_id = d.id WHERE d.device_index = ?"
        params = (device_index,)
    row = conn.execute(query, params).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_block_count(db_path: str, device_index: Optional[int] = None) -> int:
    """获取 block 数量"""
    conn = _connect(db_path)
    query = """
        SELECT COUNT(*) AS cnt FROM blocks b
        JOIN segments s ON b.segment_id = s.id
        JOIN devices d ON s.device_id = d.id
    """
    params: tuple = ()
    if device_index is not None:
        query += " WHERE d.device_index = ?"
        params = (device_index,)
    row = conn.execute(query, params).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_net_segment_trend(db_path: str, device_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """获取 net segment 变化趋势（用于泄漏检测）"""
    conn = _connect(db_path)
    query = """
        SELECT
            d.device_index,
            trace_index,
            SUM(CASE WHEN action IN ('segment_alloc', 'segment_map') THEN 1 ELSE 0 END)
                OVER (PARTITION BY d.device_index ORDER BY trace_index ROWS UNBOUNDED PRECEDING) -
            SUM(CASE WHEN action IN ('segment_free', 'segment_unmap') THEN 1 ELSE 0 END)
                OVER (PARTITION BY d.device_index ORDER BY trace_index ROWS UNBOUNDED PRECEDING) AS net_segments
        FROM traces t
        JOIN devices d ON t.device_id = d.id
    """
    params: tuple = ()
    if device_index is not None:
        query += " WHERE d.device_index = ?"
        params = (device_index,)
    query += " ORDER BY d.device_index, trace_index"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_long_lived_blocks(
    db_path: str, device_index: Optional[int] = None, threshold_pct: float = 0.2
) -> List[Dict[str, Any]]:
    """检测长生命周期 block（其所属 segment 的创建事件在时间线前 threshold_pct）"""
    conn = _connect(db_path)
    total_traces = get_trace_count(db_path, device_index)
    threshold_index = int(total_traces * threshold_pct)

    query = """
        SELECT
            d.device_index,
            b.id AS block_id,
            b.size,
            b.requested_size,
            b.state,
            s.address AS segment_address,
            cs.frames_json
        FROM blocks b
        JOIN segments s ON b.segment_id = s.id
        JOIN devices d ON s.device_id = d.id
        LEFT JOIN call_stacks cs ON b.stack_id = cs.id
        JOIN (
            SELECT addr, MIN(trace_index) AS min_trace_index
            FROM traces
            WHERE action IN ('segment_alloc', 'segment_map')
            GROUP BY addr
        ) seg_trace ON s.address = seg_trace.addr
        WHERE b.state = 'active_allocated'
          AND seg_trace.min_trace_index < ?
    """
    params: tuple = (threshold_index,)
    if device_index is not None:
        query += " AND d.device_index = ?"
        params = params + (device_index,)
    query += " ORDER BY b.size DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stack_attribution(db_path: str, device_index: Optional[int] = None) -> List[Dict[str, Any]]:
    """按堆栈聚合分配量"""
    conn = _connect(db_path)
    query = """
        SELECT
            cs.id AS stack_id,
            cs.frames_json,
            SUM(b.size) AS total_size,
            COUNT(b.id) AS block_count,
            ROUND(SUM(b.size) * 100.0 / (
                SELECT SUM(b2.size) FROM blocks b2
                JOIN segments s2 ON b2.segment_id = s2.id
                JOIN devices d2 ON s2.device_id = d2.id
                WHERE 1=1
    """
    params: tuple = ()
    if device_index is not None:
        query += " AND d2.device_index = ?"
        params = (device_index,)
    query += """
            ), 2) AS pct
        FROM blocks b
        JOIN segments s ON b.segment_id = s.id
        JOIN devices d ON s.device_id = d.id
        LEFT JOIN call_stacks cs ON b.stack_id = cs.id
    """
    if device_index is not None:
        query += " WHERE d.device_index = ?"
        params = params + (device_index,)
    query += " GROUP BY cs.id ORDER BY total_size DESC LIMIT 10"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_peak_alloc_events(
    db_path: str, device_index: Optional[int] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """获取最大的 segment_alloc / segment_map 事件及其堆栈，用于峰值归因"""
    conn = _connect(db_path)
    query = """
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
        WHERE t.action IN ('segment_alloc', 'segment_map')
    """
    params: tuple = ()
    if device_index is not None:
        query += " AND d.device_index = ?"
        params = (device_index,)
    query += " ORDER BY t.size DESC LIMIT ?"
    params = params + (limit,)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_top_blocks_with_stack(
    db_path: str, device_index: Optional[int] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    """获取最大的 active_allocated block 及其堆栈，用于峰值归因"""
    conn = _connect(db_path)
    query = """
        SELECT
            d.device_index,
            b.size,
            b.requested_size,
            b.state,
            s.address AS segment_address,
            s.total_size AS segment_total_size,
            cs.frames_json
        FROM blocks b
        JOIN segments s ON b.segment_id = s.id
        JOIN devices d ON s.device_id = d.id
        LEFT JOIN call_stacks cs ON b.stack_id = cs.id
        WHERE b.state = 'active_allocated'
    """
    params: tuple = ()
    if device_index is not None:
        query += " AND d.device_index = ?"
        params = (device_index,)
    query += " ORDER BY b.size DESC LIMIT ?"
    params = params + (limit,)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def execute_sql(db_path: str, sql: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """执行只读 SQL 查询（供 Agent 直接调用）"""
    sql_stripped = sql.strip().upper()
    if not (sql_stripped.startswith("SELECT") or sql_stripped.startswith("WITH") or sql_stripped.startswith("EXPLAIN")):
        raise ValueError("仅允许 SELECT / WITH / EXPLAIN 查询，拒绝: " + sql[:80])
    conn = _connect(db_path)
    rows = conn.execute(sql, params or ()).fetchall()
    conn.close()
    return [dict(r) for r in rows]