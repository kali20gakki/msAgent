#!/usr/bin/python3
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

"""db_query.py —— 昇腾 Profiling DB 通用查询工具

只读连接昇腾 Profiling DB（sqlite），提供三组查询能力：

1. STRING_IDS 字符串查询：
   - 是否存在某个字符串（精确匹配）；
   - 以某个字符串开头的字符串（字面前缀匹配）；
   - 给定字符串列表，返回其中存在于表中的子集。

2. MSTX_EVENTS 起止时间查询：JOIN STRING_IDS ON message 反查打点名，
   支持单个字符串 / 字符串序列 / 一个或多个前缀（前缀匹配）。

3. PYTORCH_API / CANN_API 起止时间查询：JOIN STRING_IDS ON name 反查 API 名，
   两者各有独立方法，另提供合并查询方法（query_api_combined）。

时间语义：start / end 均为 Linux epoch 纳秒（ns），duration = end - start（ns）。

字符串匹配两种模式：
- exact： 精确匹配（`value IN (...)`），可传单个字符串或列表；
- prefix：前缀匹配（`value LIKE 'x%'`，`%`/`_`/`\\` 已转义），可传单个或多个前缀。

用法（CLI 子命令，输出 JSON）：
  python db_query.py string  --db <db> --value <str>
  python db_query.py string  --db <db> --prefix <str>
  python db_query.py string  --db <db> --list <str> [<str> ...]
  python db_query.py mstx    --db <db> --name <str> [<str> ...] [--mode exact|prefix]
  python db_query.py pytorch --db <db> --name <str> [<str> ...] [--mode exact|prefix] [--type 50001 ...]
  python db_query.py cann    --db <db> --name <str> [<str> ...] [--mode exact|prefix] [--type ...]
  python db_query.py api     --db <db> --name <str> [<str> ...] [--mode exact|prefix] [--type ...]

也可作为模块 import，直接调用 string_* / query_mstx / query_pytorch_api /
query_cann_api / query_api_combined 等函数。
"""

import argparse
import json
import os
import sqlite3
from urllib.parse import quote

# ENUM_API_TYPE 关键取值（见 DB 格式文档 ENUM_API_TYPE 表）
OP_TYPE = 50001     # op（record_function scope / 算子）
QUEUE_TYPE = 50002  # queue（Enqueue@/Dequeue@ 宿主队列封装，通常需排除）
TRACE_TYPE = 50003  # trace
MSTX_TYPE = 50004   # mstx（低开销打点）


def open_ro(path):
    """只读打开 sqlite。profiling DB 常带未 checkpoint 的 -wal，immutable=1 会跳过
    WAL 恢复导致表缺失，故仅用 mode=ro（依赖已落盘的 -shm/-wal 正常读取）。
    路径先归一化为绝对路径并做 URI 百分号编码（空格 / ? / # / % 等特殊字符
    不会破坏 sqlite URI 的解析）。"""
    abs_path = os.path.abspath(path)
    uri = "file:%s?mode=ro" % quote(abs_path.replace("\\", "/"))
    return sqlite3.connect(uri, uri=True)


def _escape_like(s):
    """转义 LIKE 特殊字符，使前缀匹配语义等价于 str.startswith（字面匹配前缀）。"""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _as_list(names):
    """把单个字符串 / 字符串列表归一化为去空、去重（保持顺序）的非空列表。"""
    if isinstance(names, str):
        names = [names]
    uniq, seen = [], set()
    for n in names:
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    if not uniq:
        raise ValueError("names 不能为空")
    return uniq


def _name_cond(column, names, mode):
    """根据 names 与 mode 生成 (条件 SQL 片段, 参数列表)。

    column：参与匹配的列（例如 `S.value`）。
    mode='exact'  -> `column IN (?, ...)`；
    mode='prefix' -> `(column LIKE ? ESCAPE '\\' OR ...)`，前缀按字面转义后加 `%`。
    """
    names = _as_list(names)
    if mode == "exact":
        marks = ",".join("?" for _ in names)
        cond = "%s IN (%s)" % (column, marks)
        params = list(names)
    elif mode == "prefix":
        cond = " OR ".join("%s LIKE ? ESCAPE '\\'" % column for _ in names)
        params = [_escape_like(n) + "%" for n in names]
    else:
        raise ValueError("mode 必须为 'exact' 或 'prefix'，实际为 %r" % (mode,))
    return cond, params


def _rows(rows, source):
    """把 SQL 结果行 (start, end, name) 归一化为记录 dict 列表。"""
    out = []
    for start, end, name in rows:
        s, e = int(start), int(end)
        out.append({"source": source, "name": name, "start": s, "end": e,
                    "duration": e - s})
    return out


# ---------------------------------------------------------------------------
# 1. STRING_IDS 字符串查询
# ---------------------------------------------------------------------------

def string_exists(cur, value):
    """是否存在某个字符串（精确匹配）。返回 bool。"""
    cur.execute("SELECT 1 FROM STRING_IDS WHERE value = ? LIMIT 1", (value,))
    return cur.fetchone() is not None


def strings_like(cur, prefix):
    """返回以 prefix 开头的所有字符串（字面前缀，升序）。"""
    cur.execute("SELECT value FROM STRING_IDS WHERE value LIKE ? ESCAPE '\\' "
                "ORDER BY value", (_escape_like(prefix) + "%",))
    return [r[0] for r in cur.fetchall()]


def strings_present(cur, values):
    """给定字符串列表，返回其中存在于 STRING_IDS 的子集（保持入参顺序、去重）。"""
    values = _as_list(values)
    marks = ",".join("?" for _ in values)
    cur.execute("SELECT value FROM STRING_IDS WHERE value IN (%s)" % marks, values)
    present = {r[0] for r in cur.fetchall()}
    return [v for v in values if v in present]


# ---------------------------------------------------------------------------
# 2. MSTX_EVENTS 起止时间查询（JOIN STRING_IDS ON message）
# ---------------------------------------------------------------------------

def query_mstx(cur, names, mode="exact"):
    """查询 MSTX_EVENTS 中 message 匹配 names 的事件起止时间，按 start 升序。

    names：单个字符串或字符串列表；mode='exact' 精确 / 'prefix' 前缀。
    返回 [{source, name, start, end, duration}, ...]（时间单位 ns）。
    """
    cond, params = _name_cond("S.value", names, mode)
    sql = (
        "SELECT CAST(MX.startNs AS INTEGER) AS start, "
        "CAST(MX.endNs AS INTEGER) AS end, S.value AS name "
        "FROM MSTX_EVENTS MX "
        "LEFT JOIN STRING_IDS S ON S.id = MX.message "
        "WHERE (%s) AND MX.startNs IS NOT NULL AND MX.endNs IS NOT NULL "
        "ORDER BY MX.startNs" % cond
    )
    cur.execute(sql, params)
    return _rows(cur.fetchall(), "MSTX_EVENTS")


# ---------------------------------------------------------------------------
# 3. PYTORCH_API / CANN_API 起止时间查询（JOIN STRING_IDS ON name）
# ---------------------------------------------------------------------------

def _query_api(cur, table, names, mode, types):
    """PYTORCH_API / CANN_API 共用的查询模板。"""
    cond, params = _name_cond("S.value", names, mode)
    sql = (
        "SELECT CAST(A.startNs AS INTEGER) AS start, "
        "CAST(A.endNs AS INTEGER) AS end, S.value AS name "
        "FROM %s A LEFT JOIN STRING_IDS S ON S.id = A.name "
        "WHERE (%s) AND A.startNs IS NOT NULL AND A.endNs IS NOT NULL"
        % (table, cond)
    )
    if types:
        tmarks = ",".join("?" for _ in types)
        sql += " AND A.type IN (%s)" % tmarks
        params = params + [int(t) for t in types]
    sql += " ORDER BY A.startNs"
    cur.execute(sql, params)
    return _rows(cur.fetchall(), table)


def query_pytorch_api(cur, names, mode="exact", types=None):
    """查询 PYTORCH_API 中 name 匹配 names 的事件起止时间（JOIN name），按 start 升序。

    types：可选，按 ENUM_API_TYPE 过滤（如 (50001, 50004) 可排除 50002 queue 封装）。
    返回 [{source, name, start, end, duration}, ...]（时间单位 ns）。
    """
    return _query_api(cur, "PYTORCH_API", names, mode, types)


def query_cann_api(cur, names, mode="exact", types=None):
    """查询 CANN_API 中 name 匹配 names 的事件起止时间（JOIN name），按 start 升序。

    types：可选，按 ENUM_API_TYPE 过滤。
    返回 [{source, name, start, end, duration}, ...]（时间单位 ns）。
    """
    return _query_api(cur, "CANN_API", names, mode, types)


def query_api_combined(cur, names, mode="exact", types=None):
    """合并查询 PYTORCH_API 与 CANN_API，返回按 start 升序的记录，source 区分来源表。"""
    merged = (query_pytorch_api(cur, names, mode, types)
              + query_cann_api(cur, names, mode, types))
    merged.sort(key=lambda r: r["start"])
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_json(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="昇腾 Profiling DB 通用查询工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("string", help="STRING_IDS 字符串查询")
    sp.add_argument("--db", required=True, help="Profiling DB 路径")
    g = sp.add_mutually_exclusive_group(required=True)
    g.add_argument("--value", help="精确匹配的字符串，返回是否存在（bool）")
    g.add_argument("--prefix", help="前缀，返回所有以它开头的字符串")
    g.add_argument("--list", nargs="+", help="字符串列表，返回其中存在于表中的子集")

    for name, help_txt in (("mstx", "MSTX_EVENTS 起止时间查询"),
                           ("pytorch", "PYTORCH_API 起止时间查询"),
                           ("cann", "CANN_API 起止时间查询"),
                           ("api", "PYTORCH_API + CANN_API 合并查询")):
        p = sub.add_parser(name, help=help_txt)
        p.add_argument("--db", required=True, help="Profiling DB 路径")
        p.add_argument("--name", nargs="+", required=True,
                       help="单个字符串或字符串序列；搭配 --mode prefix 时表示前缀")
        p.add_argument("--mode", choices=["exact", "prefix"], default="exact",
                       help="匹配模式：exact 精确 / prefix 前缀（默认 exact）")
        if name != "mstx":
            p.add_argument("--type", type=int, action="append", default=None,
                           help="按 ENUM_API_TYPE 过滤，可多次指定（如 50001、50004）")

    args = parser.parse_args(argv)
    conn = open_ro(args.db)
    try:
        cur = conn.cursor()

        if args.cmd == "string":
            if args.value is not None:
                _print_json({"db": args.db, "query": "exists", "value": args.value,
                             "result": string_exists(cur, args.value)})
            elif args.prefix is not None:
                _print_json({"db": args.db, "query": "prefix", "prefix": args.prefix,
                             "result": strings_like(cur, args.prefix)})
            else:
                _print_json({"db": args.db, "query": "list", "values": args.list,
                             "result": strings_present(cur, args.list)})
        elif args.cmd == "mstx":
            records = query_mstx(cur, args.name, args.mode)
            _print_json({"db": args.db, "table": args.cmd, "mode": args.mode,
                         "count": len(records), "records": records})
        elif args.cmd == "pytorch":
            records = query_pytorch_api(cur, args.name, args.mode, args.type)
            _print_json({"db": args.db, "table": args.cmd, "mode": args.mode,
                         "count": len(records), "records": records})
        elif args.cmd == "cann":
            records = query_cann_api(cur, args.name, args.mode, args.type)
            _print_json({"db": args.db, "table": args.cmd, "mode": args.mode,
                         "count": len(records), "records": records})
        else:
            records = query_api_combined(cur, args.name, args.mode, args.type)
            _print_json({"db": args.db, "table": args.cmd, "mode": args.mode,
                         "count": len(records), "records": records})
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())