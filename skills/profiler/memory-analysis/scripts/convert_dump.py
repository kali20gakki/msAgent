# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
memscope_dump 数据格式转换工具（csv ↔ sqlite db 双向转换）。

用途：
- csv → db：采集时使用 --format=csv 的产物，转换后可导入 MindStudio Insight 可视化；
- db → csv：采集时使用 --format=db 的产物，转换后可供本技能分析脚本（aggregate_dump.py）使用。

db 为 sqlite 数据库，结构与官方输出文件一致，包含 4 张表：
- memscope_dump        主表：12 列与 csv 一致，Attr 列为 JSON 字符串
- memory_allocation    曲线表：每条 MALLOC/FREE 事件一行，totalSize 为事件后该 eventType 的活跃量
- memory_block         内存块生命周期表：每块一行，attr 仅保留 allocation_id
- status_info          解析状态标记（MEM_SCOPE_PARSE_STATUS）

大数据量：对流式处理，内存占用 O(活跃块数) 而非 O(总行数)（百万行级可用）；
csv → db 的曲线按输入行序推进（不再排序），转换前建议先对 csv 执行
`aggregate_dump.py --check/--sort` 预处理，保证输入按时间戳升序；未排序时打印警告。

用法：
    # csv 转 db（默认输出 memscope_dump_{时间戳}.db，可导入 MindStudio Insight）
    python3 convert_dump.py <dump.csv> --to-db [--output <out.db>] [--table <表名>]

    # db 转 csv（默认输出 memscope_dump_{时间戳}.csv）
    python3 convert_dump.py <dump.db> --to-csv [--output <out.csv>] [--table <表名>]

依赖：仅 Python 3 标准库。
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys

COL_EVENT = "Event"
COL_EVENT_TYPE = "Event Type"
COL_TIMESTAMP = "Timestamp(ns)"
COL_DEVICE = "Device Id"
COL_PTR = "Ptr"
COL_ATTR = "Attr"

EVENT_MALLOC = "MALLOC"
EVENT_FREE = "FREE"

# 批插入大小：攒够一批再 executemany，控制主表缓冲内存（每批 ~几 MB）
BATCH_SIZE = 10000

# 主表列定义（与官方 db schema 一致：ID/时间/进程/线程为 INTEGER，其余 TEXT）
MAIN_TABLE_COLUMNS = [
    ("ID", "INTEGER"),
    ("Event", "TEXT"),
    ("Event Type", "TEXT"),
    ("Name", "TEXT"),
    ("Timestamp(ns)", "INTEGER"),
    ("Process Id", "INTEGER"),
    ("Thread Id", "INTEGER"),
    ("Device Id", "TEXT"),
    ("Ptr", "TEXT"),
    ("Attr", "TEXT"),
    ("Call Stack(Python)", "TEXT"),
    ("Call Stack(C)", "TEXT"),
]

ALLOC_TABLE_DDL = (
    'CREATE TABLE IF NOT EXISTS "memory_allocation" ('
    "  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
    "  timestamp integer, totalSize integer, optimized integer,"
    "  deviceId text(255), eventType text(255))"
)
BLOCK_TABLE_DDL = (
    'CREATE TABLE IF NOT EXISTS "memory_block" ('
    "  id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
    "  deviceId text(255), addr TEXT(255), size integer,"
    "  startTimestamp integer, endTimestamp integer,"
    "  eventType text(255), owner text(255), attr TEXT(255),"
    "  processId integer, threadId integer,"
    "  firstAccessTimestamp integer, lastAccessTimestamp integer,"
    "  maxAccessInterval integer)"
)
STATUS_TABLE_DDL = (
    'CREATE TABLE IF NOT EXISTS "status_info" ('
    "  id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, value TEXT)"
)


def parse_attr(attr_str):
    """解析 Attr 列 '{k:v,k2:v2,...}' -> {'k': 'v', ...}；非 {k:v} 格式时原样返回。"""
    if not attr_str:
        return {}
    body = attr_str.strip()
    if body.startswith("{") and body.endswith("}"):
        result = {}
        for item in body[1:-1].split(","):
            if ":" in item:
                key, _, value = item.partition(":")
                result[key.strip()] = value.strip()
        return result
    return attr_str


def int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def attr_to_brace(attr):
    """dict -> '{k:v,k2:v2}' 无引号格式（db 转 csv 时还原 Attr 列）。"""
    if not attr:
        return ""
    parts = []
    for k, v in attr.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        parts.append(f"{k}:{v}")
    return "{" + ",".join(parts) + "}"


def default_output_path(input_path, ext):
    """由输入文件名生成默认输出名，输出到**输入文件所在目录**。

    多卡采集产物结构为 device_{id}/dump/memscope_dump_{时间戳}.db——
    各 device 时间戳文件名相同，输出必须保留输入目录，否则转换结果互相覆盖。
    显式 --output 可覆盖此行为。
    """
    out_name = os.path.basename(input_path)
    m = re.search(r"memscope_dump_(\d+)", out_name)
    if m:
        out_name = f"memscope_dump_{m.group(1)}{ext}"
    else:
        out_name = os.path.splitext(out_name)[0] + ext
    return os.path.join(os.path.dirname(input_path), out_name)


def iter_csv_rows(input_path):
    """流式读取 csv：校验表头后逐行 yield（col_index, row）；不整文件载入内存。"""
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            fieldnames = next(reader)
        except StopIteration:
            raise ValueError(f"文件为空或缺少表头: {input_path}")
        missing = [c for c, _ in MAIN_TABLE_COLUMNS if c not in fieldnames]
        if missing:
            raise ValueError(f"缺少关键列: {missing}（实际列: {fieldnames}）")
        col_index = {name: fieldnames.index(name) for name in fieldnames}
        for row in reader:
            yield col_index, row


def do_batch_insert(cur, sql, batch):
    """批量 INSERT 并清空缓冲；batch 为空时不变。"""
    if batch:
        cur.executemany(sql, batch)
        batch.clear()


def cmd_csv_to_db(input_path, output_path, table_name):
    """csv -> db：重建输出文件（全量转换语义），流式处理，内存 O(活跃块数)。

    生成主表 + 3 张辅助表；memory_allocation 曲线按输入行序推进（输入应已 --sort）。
    """
    if os.path.exists(output_path):
        os.remove(output_path)
    conn = sqlite3.connect(output_path)
    cur = conn.cursor()
    cur.execute(f'CREATE TABLE "{table_name}" ('
                + ", ".join(f'"{c}" {t}' for c, t in MAIN_TABLE_COLUMNS) + ")")
    cur.execute(ALLOC_TABLE_DDL)
    cur.execute(BLOCK_TABLE_DDL)
    cur.execute(STATUS_TABLE_DDL)

    main_sql = f'INSERT INTO "{table_name}" VALUES (' + ",".join("?" * len(MAIN_TABLE_COLUMNS)) + ")"
    alloc_sql = ("INSERT INTO memory_allocation (timestamp, totalSize, optimized,"
                 " deviceId, eventType) VALUES (?, ?, ?, ?, ?)")
    block_sql = ("INSERT INTO memory_block (deviceId, addr, size, startTimestamp,"
                 " endTimestamp, eventType, owner, attr, processId, threadId,"
                 " firstAccessTimestamp, lastAccessTimestamp, maxAccessInterval)"
                 " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")

    main_batch = []    # 主表行批缓冲
    block_batch = []   # 已结束块批缓冲
    alloc_batch = []   # memory_allocation 曲线点批缓冲（曲线点行数≈事件行数，必需批化）
    active = {}        # ptr -> 块信息（未释放）；内存仅随活跃块数增长
    per_type = {}      # eventType -> 曲线该类型活跃累计
    main_count = alloc_count = block_count = 0
    min_ts = None
    last_ts = None
    warned_unsorted = False

    for col_index, row in iter_csv_rows(input_path):
        def cell(name):
            idx = col_index.get(name)
            return row[idx] if idx is not None and idx < len(row) else ""

        event = cell(COL_EVENT)
        ts = int_or_none(cell(COL_TIMESTAMP))
        if ts is not None:
            if last_ts is not None and ts < last_ts and not warned_unsorted:
                print("[WARN] 输入时间戳非升序：曲线 per_type 按输入行序推进可能不准确，"
                      "建议先执行 aggregate_dump.py --sort", file=sys.stderr)
                warned_unsorted = True
            last_ts = ts
            if min_ts is None or ts < min_ts:
                min_ts = ts
        attr = parse_attr(cell(COL_ATTR))
        attr = attr if isinstance(attr, dict) else {}
        ptr = cell(COL_PTR)

        # 主表行（Attr 转 JSON 字符串，与官方 db 存储一致）
        main_batch.append([json.dumps(attr, ensure_ascii=False, separators=(",", ":"))
                           if name == COL_ATTR else cell(name)
                           for name, _ in MAIN_TABLE_COLUMNS])
        main_count += 1
        if len(main_batch) >= BATCH_SIZE:
            do_batch_insert(cur, main_sql, main_batch)

        # 块生命周期
        if event == EVENT_MALLOC:
            if ptr in active:  # 同地址未释放时被再次申请：先关闭旧块
                prev = active.pop(ptr)
                prev["end"] = ts
                block_batch.append(format_block(prev, None))
                block_count += 1
                if len(block_batch) >= BATCH_SIZE:
                    do_batch_insert(cur, block_sql, block_batch)
            active[ptr] = {
                "device_id": cell(COL_DEVICE),
                "addr": ptr,
                "size": int_or_none(attr.get("size")) or 0,
                "start": ts,
                "end": None,
                "event_type": cell(COL_EVENT_TYPE),
                "owner": attr.get("owner") or "",  # 无 owner 的 shadow 块官方记录为空串
                "attr": block_attr(attr),
                "pid": int_or_none(cell("Process Id")),
                "tid": int_or_none(cell("Thread Id")),
            }
        elif event == EVENT_FREE and ptr in active:
            block = active.pop(ptr)
            block["end"] = ts
            block["pid"] = int_or_none(cell("Process Id"))   # 官方记录释放时刻的进程/线程
            block["tid"] = int_or_none(cell("Thread Id"))
            block_batch.append(format_block(block, None))
            block_count += 1
            if len(block_batch) >= BATCH_SIZE:
                do_batch_insert(cur, block_sql, block_batch)

        # 曲线点（按输入行序推进）：含 used 直接用 used 同步；缺 used 按前值 ± size 推算
        if event in (EVENT_MALLOC, EVENT_FREE):
            etype = cell(COL_EVENT_TYPE)
            used = int_or_none(attr.get("used"))
            if used is not None:
                per_type[etype] = used
            else:
                size = int_or_none(attr.get("size")) or 0
                per_type[etype] = max(0, per_type.get(etype, 0)
                                      + (size if event == EVENT_MALLOC else -size))
            alloc_batch.append((ts or 0, per_type[etype], 0, cell(COL_DEVICE), etype))
            alloc_count += 1
            if len(alloc_batch) >= BATCH_SIZE:
                do_batch_insert(cur, alloc_sql, alloc_batch)

    do_batch_insert(cur, main_sql, main_batch)
    do_batch_insert(cur, alloc_sql, alloc_batch)
    do_batch_insert(cur, block_sql, block_batch)
    block_count += len(active)  # 采集结束时未释放的块（endTimestamp 置空）

    first_access = (min_ts - 1) if min_ts is not None else None  # 无 access 事件时的采集开始时刻近似
    if active:
        cur.executemany(block_sql,
                        (format_block(b, first_access) for b in active.values()))

    cur.execute("INSERT INTO status_info (key, value) VALUES ('MEM_SCOPE_PARSE_STATUS', 'FINISH')")
    conn.commit()
    conn.close()

    print(f"[OK] {input_path} → {output_path}")
    print(f"     主表 {table_name}: {main_count} 行")
    print(f"     memory_allocation: {alloc_count} 行（alloc/free 事件 used 曲线点）")
    print(f"     memory_block: {block_count} 行（含 {len(active)} 个未释放块）")


def block_attr(attr):
    """memory_block.attr 仅保留 allocation_id（值转 int，与官方格式一致）。"""
    aid = attr.get("allocation_id")
    if aid is None:
        return "{}"
    try:
        return json.dumps({"allocation_id": int(aid)}, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps({"allocation_id": aid}, separators=(",", ":"))


def format_block(block, first_access):
    return (
        block["device_id"], block["addr"], block["size"],
        block["start"], block["end"],
        block["event_type"], block["owner"], block["attr"],
        block["pid"], block["tid"],
        first_access, first_access, 0,
    )


def cmd_db_to_csv(input_path, output_path, table_name):
    """db -> csv：流式读取主表，Attr 列由 JSON 字符串还原为 {k:v} 格式。"""
    conn = sqlite3.connect(input_path)
    conn.text_factory = str
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table_name}")')
    cols = [r[1] for r in cur.fetchall()]
    if not cols:
        conn.close()
        raise ValueError(f"数据库中不存在表 {table_name}（可尝试 --table 指定其他表名，如 leaks_dump）")
    attr_idx = cols.index(COL_ATTR) if COL_ATTR in cols else None

    count = 0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        cur.execute(f'SELECT * FROM "{table_name}"')
        for row in cur:
            row = list(row)
            if attr_idx is not None and isinstance(row[attr_idx], str):
                try:
                    parsed = json.loads(row[attr_idx])
                except ValueError:
                    parsed = None
                if isinstance(parsed, dict):
                    row[attr_idx] = attr_to_brace(parsed)
            writer.writerow(row)
            count += 1
    conn.close()
    print(f"[OK] {input_path} → {output_path}（表 {table_name}，共 {count} 行）")


def main():
    # 输出稳定为 UTF-8：Windows 控制台/管道默认按 locale（GBK）编码，重定向时可能抛
    # UnicodeEncodeError；Linux 下无影响。reconfigure 需 Python 3.7+，旧版本静默跳过。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        description="memscope_dump 数据格式转换工具（csv <-> sqlite db 双向）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python3 convert_dump.py dump.csv --to-db\n"
               "  python3 convert_dump.py dump.csv --to-db --output out.db\n"
               "  python3 convert_dump.py dump.db --to-csv\n"
               "  python3 convert_dump.py dump.db --to-csv --table leaks_dump",
    )
    parser.add_argument("input", help="输入文件：memscope_dump_*.csv 或 *.db")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--to-db", action="store_true",
                       help="csv 转 sqlite db（可导入 MindStudio Insight 可视化）")
    group.add_argument("--to-csv", action="store_true",
                       help="db 转 csv（供 aggregate_dump.py 等脚本分析）")
    parser.add_argument("--output", help="输出文件路径（默认按输入文件名生成）")
    parser.add_argument("--table", default="memscope_dump",
                        help="主表名（默认 memscope_dump；旧版工具链可用 leaks_dump）")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[FAIL] 输入文件不存在: {args.input}", file=sys.stderr)
        sys.exit(1)
    try:
        if args.to_db:
            cmd_csv_to_db(args.input, args.output or default_output_path(args.input, ".db"), args.table)
        else:
            cmd_db_to_csv(args.input, args.output or default_output_path(args.input, ".csv"), args.table)
    except (ValueError, sqlite3.Error, OSError) as exc:
        print(f"[FAIL] 转换失败: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()