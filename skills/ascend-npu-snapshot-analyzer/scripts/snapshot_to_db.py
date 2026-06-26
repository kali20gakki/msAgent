#!/usr/bin/env python3
"""
Ascend NPU Memory Snapshot pickle → SQLite 转换工具

将 torch_npu.npu.memory._dump_snapshot() 导出的 pickle 文件转换为 SQLite 数据库，
以支持高效的 SQL 查询和跨快照对比。

用法:
    python snapshot_to_db.py snapshot.pkl                # 自动生成 snapshot.db
    python snapshot_to_db.py snapshot.pkl -o custom.db   # 指定输出路径
    python snapshot_to_db.py snapshot.pkl --no-indexes   # 跳过索引创建（调试用）
"""

import argparse
import hashlib
import json
import os
import pickle  # nosec B403
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _pickle_load(filepath: str) -> Dict[str, Any]:
    """加载 pickle 文件，兼容 list 和 dict 两种格式"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with open(filepath, "rb") as f:
        data = pickle.load(f)  # nosec B403

    if isinstance(data, dict):
        return {
            "segments": data.get("segments", []),
            "device_traces": data.get("device_traces") or [],
        }
    elif isinstance(data, list):
        return {"segments": data, "device_traces": []}
    else:
        raise ValueError(f"无效的 snapshot 格式: 期望 list 或 dict, 得到 {type(data).__name__}")


def _stack_hash(frames: List[str]) -> str:
    """计算堆栈帧的 MD5 哈希"""
    content = json.dumps(frames, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode("utf-8")).hexdigest()


class _CallStackResolver:
    """堆栈解析器，通过内存缓存减少重复的 SQL 查询"""

    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor
        self._cache: Dict[str, int] = {}
        self._load_existing()

    def _load_existing(self):
        rows = self._cursor.execute("SELECT id, stack_hash FROM call_stacks").fetchall()
        self._cache = {row[1]: row[0] for row in rows}

    def resolve(self, frames: List[str]) -> Optional[int]:
        if not frames:
            return None

        hash_val = _stack_hash(frames)

        cached_id = self._cache.get(hash_val)
        if cached_id is not None:
            return cached_id

        frames_json = json.dumps(frames, ensure_ascii=False)
        self._cursor.execute(
            "INSERT OR IGNORE INTO call_stacks (stack_hash, frames_json) VALUES (?, ?)",
            (hash_val, frames_json),
        )
        self._cursor.execute("SELECT id FROM call_stacks WHERE stack_hash = ?", (hash_val,))
        row = self._cursor.fetchone()
        if row:
            self._cache[hash_val] = row[0]
            return row[0]
        return None

    def flush_all(self):
        """当前实现为即时写入 (resolve 内 INSERT OR IGNORE)，无需 flush，保留接口供未来扩展"""
        pass


def _resolve_device(cursor: sqlite3.Cursor, device_index: int) -> int:
    """解析设备，返回 devices 表中的 id"""
    cursor.execute(
        "INSERT OR IGNORE INTO devices (device_index) VALUES (?)",
        (device_index,),
    )
    cursor.execute("SELECT id FROM devices WHERE device_index = ?", (device_index,))
    return cursor.fetchone()[0]


def _create_tables(conn: sqlite3.Connection):
    """创建表结构（不含索引）"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_index INTEGER UNIQUE NOT NULL,
            device_type TEXT DEFAULT 'Ascend-NPU'
        );

        CREATE TABLE IF NOT EXISTS call_stacks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stack_hash TEXT UNIQUE,
            frames_json TEXT
        );

        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER NOT NULL,
            address INTEGER,
            total_size INTEGER,
            allocated_size INTEGER,
            active_size INTEGER,
            requested_size INTEGER,
            stream INTEGER,
            segment_type TEXT,
            pool_id_0 INTEGER,
            pool_id_1 INTEGER,
            is_expandable INTEGER,
            stack_id INTEGER,
            FOREIGN KEY (device_id) REFERENCES devices(id),
            FOREIGN KEY (stack_id) REFERENCES call_stacks(id)
        );

        CREATE TABLE IF NOT EXISTS blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id INTEGER NOT NULL,
            address INTEGER,
            size INTEGER,
            requested_size INTEGER,
            state TEXT,
            stack_id INTEGER,
            FOREIGN KEY (segment_id) REFERENCES segments(id) ON DELETE CASCADE,
            FOREIGN KEY (stack_id) REFERENCES call_stacks(id)
        );

        CREATE TABLE IF NOT EXISTS traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id INTEGER NOT NULL,
            trace_index INTEGER,
            action TEXT,
            addr INTEGER,
            device_free INTEGER,
            size INTEGER,
            stream INTEGER,
            stack_id INTEGER,
            FOREIGN KEY (device_id) REFERENCES devices(id),
            FOREIGN KEY (stack_id) REFERENCES call_stacks(id)
        );
    """)
    conn.commit()


def _create_indexes(conn: sqlite3.Connection):
    """创建索引（数据导入完成后调用）"""
    indexes = [
        ("idx_blocks_state", "blocks", "state"),
        ("idx_blocks_stack", "blocks", "stack_id"),
        ("idx_blocks_segment", "blocks", "segment_id"),
        ("idx_blocks_segment_state", "blocks", "segment_id, state"),
        ("idx_traces_stack", "traces", "stack_id"),
        ("idx_traces_device_action", "traces", "device_id, action"),
        ("idx_traces_device_index", "traces", "device_id, trace_index"),
        ("idx_segments_stack", "segments", "stack_id"),
        ("idx_segments_device", "segments", "device_id"),
        ("idx_segments_device_type", "segments", "device_id, segment_type"),
    ]
    for name, table, columns in indexes:
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({columns})")
    conn.commit()


def _insert_segments(
    cursor: sqlite3.Cursor,
    segments: List[Dict],
    stack_resolver: _CallStackResolver,
) -> Tuple[int, int]:
    """插入 segments 和 blocks 数据，返回 (segment_count, block_count)"""
    seg_count = 0
    block_count = 0
    blend = len(segments)
    progress_interval = max(1, blend // 10)

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue

        device_id = _resolve_device(cursor, seg.get("device", 0))
        stack_id = stack_resolver.resolve(seg.get("frames", []))

        pool_id = seg.get("segment_pool_id", ())
        pool_0 = pool_id[0] if isinstance(pool_id, (tuple, list)) and len(pool_id) > 0 else None
        pool_1 = pool_id[1] if isinstance(pool_id, (tuple, list)) and len(pool_id) > 1 else None

        cursor.execute(
            """INSERT INTO segments
               (device_id, address, total_size, allocated_size, active_size,
                requested_size, stream, segment_type, pool_id_0, pool_id_1,
                is_expandable, stack_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                device_id,
                seg.get("address", 0),
                seg.get("total_size", 0),
                seg.get("allocated_size", 0),
                seg.get("active_size", 0),
                seg.get("requested_size", 0),
                seg.get("stream", 0),
                seg.get("segment_type", "unknown"),
                pool_0,
                pool_1,
                1 if seg.get("is_expandable") else 0,
                stack_id,
            ),
        )
        segment_id = cursor.lastrowid

        blocks = seg.get("blocks", [])
        block_rows = []
        for blk in blocks:
            if not isinstance(blk, dict):
                continue
            blk_stack_id = stack_resolver.resolve(blk.get("frames", []))
            block_rows.append((
                segment_id,
                blk.get("address", 0),
                blk.get("size", 0),
                blk.get("requested_size", 0),
                blk.get("state", "unknown"),
                blk_stack_id,
            ))

        if block_rows:
            cursor.executemany(
                "INSERT INTO blocks (segment_id, address, size, requested_size, state, stack_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                block_rows,
            )
            block_count += len(block_rows)

        seg_count += 1

        if (i + 1) % progress_interval == 0 or i == blend - 1:
            pct = (i + 1) / blend * 100
            print(
                f"\r  [Segments] {i + 1}/{blend} ({pct:.0f}%) | blocks: {block_count}",
                end="",
                flush=True,
            )

    stack_resolver.flush_all()
    print("")
    return seg_count, block_count


def _insert_traces(
    cursor: sqlite3.Cursor,
    device_traces: List[List[Dict]],
    stack_resolver: _CallStackResolver,
) -> int:
    """插入 traces 数据，返回插入行数"""
    total_events = sum(len(tl) for tl in device_traces if tl)
    print(f"  [Traces] 共 {total_events} 条事件，开始导入...")

    count = 0
    batch_size = 5000
    progress_interval = max(1, total_events // 10)

    for device_index, trace_list in enumerate(device_traces):
        if not trace_list:
            continue

        device_id = _resolve_device(cursor, device_index)
        batch = []

        for idx, trace in enumerate(trace_list):
            if not isinstance(trace, dict):
                continue

            stack_id = stack_resolver.resolve(trace.get("frames", []))
            action = trace.get("action", "unknown")
            addr = trace.get("addr")
            device_free = trace.get("device_free")

            batch.append((
                device_id,
                idx,
                action,
                addr,
                device_free,
                trace.get("size", 0),
                trace.get("stream", 0),
                stack_id,
            ))
            count += 1

            if len(batch) >= batch_size:
                cursor.executemany(
                    "INSERT INTO traces (device_id, trace_index, action, addr, device_free, size, stream, stack_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                batch = []

            if count % progress_interval == 0:
                pct = count / total_events * 100
                print(
                    f"\r  [Traces] {count}/{total_events} ({pct:.0f}%)",
                    end="",
                    flush=True,
                )

        if batch:
            cursor.executemany(
                "INSERT INTO traces (device_id, trace_index, action, addr, device_free, size, stream, stack_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )

    stack_resolver.flush_all()
    print(f"\r  [Traces] {count}/{total_events} (100%)")
    return count


def convert_snapshot(
    filepath: str,
    output_path: Optional[str] = None,
    create_indexes: bool = True,
) -> Dict[str, Any]:
    """将 pickle 文件转换为 SQLite DB

    Args:
        filepath: pickle 文件路径
        output_path: 输出 DB 路径，默认与 pickle 同名（仅换后缀 .db）
        create_indexes: 是否创建索引

    Returns:
        包含导入统计信息的字典
    """
    t_start = time.time()

    if output_path is None:
        base = os.path.splitext(filepath)[0]
        output_path = base + ".db"

    raw = _pickle_load(filepath)
    segments = raw["segments"]
    device_traces = raw["device_traces"]

    print(f"加载完成: {len(segments)} segments, {sum(len(tl) for tl in device_traces if tl)} trace 事件")
    print(f"输出: {output_path}")

    conn = sqlite3.connect(output_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA mmap_size=268435456")
    cursor = conn.cursor()

    try:
        _create_tables(conn)
        print("表结构创建完成")

        stack_resolver = _CallStackResolver(cursor)

        seg_count, block_count = _insert_segments(cursor, segments, stack_resolver)
        conn.commit()
        print(f"Segments 导入完成: {seg_count} segments, {block_count} blocks")

        trace_count = _insert_traces(cursor, device_traces, stack_resolver)
        conn.commit()

        if create_indexes:
            t_idx_start = time.time()
            _create_indexes(conn)
            t_idx = time.time() - t_idx_start
        else:
            t_idx = 0

        cursor.execute("SELECT COUNT(*) FROM call_stacks")
        stack_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM devices")
        device_count = cursor.fetchone()[0]

    finally:
        conn.close()

    t_total = time.time() - t_start
    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    return {
        "output_path": output_path,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "segments": seg_count,
        "blocks": block_count,
        "traces": trace_count,
        "call_stacks": stack_count,
        "devices": device_count,
        "time_total_s": round(t_total, 2),
        "time_indexes_s": round(t_idx, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ascend NPU Memory Snapshot pickle → SQLite 转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("snapshot", help="snapshot pickle 文件路径")
    parser.add_argument("-o", "--output", help="输出 SQLite DB 路径（默认与 pickle 同名，后缀 .db）")
    parser.add_argument("--no-indexes", action="store_true", help="跳过索引创建（调试用）")

    args = parser.parse_args()

    try:
        stats = convert_snapshot(
            args.snapshot,
            output_path=args.output,
            create_indexes=not args.no_indexes,
        )

        print(f"转换完成: {stats['output_path']}")
        print(f"文件大小: {stats['file_size_mb']} MB")
        print(f"Segments: {stats['segments']}")
        print(f"Blocks:   {stats['blocks']}")
        print(f"Traces:   {stats['traces']}")
        print(f"堆栈条目: {stats['call_stacks']}")
        print(f"设备数:   {stats['devices']}")
        print(f"总耗时:   {stats['time_total_s']}s")
        if stats['time_indexes_s'] > 0:
            print(f"索引耗时: {stats['time_indexes_s']}s")

    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()