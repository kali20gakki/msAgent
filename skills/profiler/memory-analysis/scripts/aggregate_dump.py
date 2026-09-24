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
memscope_dump csv 聚合统计与本地可视化工具。

对 msMemScope 采集产物 memscope_dump_{timestamp}.csv 做确定性聚合统计，
供"显存大头排名、峰值时刻占比、显存曲线"等数值解读使用（禁止手工估算）。

大数据量支持：本脚本按流式（逐行解析、单遍扫描）处理 dump，不将全量事件
载入内存，可处理数十万至百万行级 dump（--ascii 曲线按列式数组分桶，
内存占用 O(事件数) 且为紧凑基础类型，远小于每行 dict 方式）。

用法：
    # 数据预处理（每个 dump 文件解读前执行一次，幂等；已处理过的文件会提示已就绪）
    python3 aggregate_dump.py <dump.csv> --check            # 完整性校验（表头/列数/最后一行/是否已有序）
    python3 aggregate_dump.py <dump.csv> --sort [--overwrite]  # 按时间戳排序；--overwrite 覆写原文件

    # 按 owner 聚合显存占用排名（需已开启 decompose；metric=peak 为峰值口径，total 为累计申请口径）
    python3 aggregate_dump.py <dump.csv> --group-by owner --metric peak

    # 按事件来源（HAL/PTA/ATB/MindSpore/HOST_PINNED）聚合（未开 decompose 时的降级分析）
    python3 aggregate_dump.py <dump.csv> --group-by event_type --metric peak

    # 统计指定时间点活跃块分布（要求 dump 已按时间戳有序，未排序时提示先 --sort）。
    # 拆解是单维度（内存池）操作且为**级联**结构：HAL 事件是驱动层全集（含 PTA/ATB/
    # MindSpore 框架池向驱动申请的大段，owner=CANN@APP），池事件是同一物理内存的
    # 池内子视图——两类数值嵌套、不可相加，故默认从驱动层全集开始拆，池内再下钻。
    # HOST 暂不分析。
    python3 aggregate_dump.py <dump.csv> --at-timestamp <ns>                 # 默认拆 HAL（驱动层全集）
    python3 aggregate_dump.py <dump.csv> --at-timestamp <ns> --pool HAL      # 显式驱动层拆解
    python3 aggregate_dump.py <dump.csv> --at-timestamp <ns> --pool PTA      # HAL 之下第二层：池内用途

    # 泄漏候选扫描：筛选"采集窗口内申请、无真实释放（FREE 全部 shadow:true）"的块，
    # 按 owner/调用栈聚合（--detail N 控制明细条数，默认 10，0=不输出；跨池时建议 --pool）
    python3 aggregate_dump.py <dump.csv> --leak-candidates [--detail 10] [--pool PTA]

    # 输出整卡/进程显存曲线数据点（ts, used, device_used, process_used）
    python3 aggregate_dump.py <dump.csv> --curve [--limit 10000]

    # 输出文本化曲线（ASCII，无外部依赖）
    python3 aggregate_dump.py <dump.csv> --curve --ascii [--bins 40]

    # 峰值定位（--key 选择 used/device_used/process_used），输出可直接衔接 --at-timestamp
    python3 aggregate_dump.py <dump.csv> --peak [--key used] [--pool HAL]

    # OOM 诊断汇总（TRIGGER/TOP_ALLOC/RECENT_ALLOC + 量化推断提示；--detail 0=不输出明细）
    python3 aggregate_dump.py <dump.csv> --oom [--detail 10]

    # 池碎片率与扩容画像（扩容清单/碎片率统计/有效使用率；--pool 指定池，默认全部池事件）
    python3 aggregate_dump.py <dump.csv> --fragmentation [--pool PTA]

    # HAL 池大段清单（核对 HCCL CCL Buffer ~401MB、池扩容段；--min-size 过滤，支持 KB/MB/GB 后缀）
    python3 aggregate_dump.py <dump.csv> --hal-segments [--min-size 100MB]

    # 跨周期趋势（等宽桶 used/total 末值 + 算法 A/B 判定；默认拆 HAL + PTA 双视图（PTA 存在时），
    # --pool 指定单一池）
    python3 aggregate_dump.py <dump.csv> --trend [--buckets 20] [--pool PTA]

    # 数据画像（事件分布/时间范围/解析能力提示，分析前体检）
    python3 aggregate_dump.py <dump.csv> --stats

    # 时间窗切片（起点/终点活跃块、窗口内申请/释放/净变化、TOP 归口）
    python3 aggregate_dump.py <dump.csv> --window <START> <END> [--pool HAL]

说明：
    - --group-by 与文件是否已排序无关（典型 dump 即按时间顺序写入，峰值统计
      只需 malloc/free 配对）；--curve 按文件记录顺序描点，若文件时间戳未
      排序会在 stderr 提示（曲线图形/数值准确性以先执行 --check/--sort 为佳）；
      --at-timestamp 需要文件按时间戳有序（流式检查，未排序会提示先 --sort）。
      --peak/--stats 与排序无关；--trend/--window 按时间序流式处理，未排序时
      stderr 提示（准确性以先 --check/--sort 为佳）。
    - --limit N：--curve 数据点输出最多 N 条（默认 0=全部；ASCII 模式下
      默认不输出数据点表，如需前 N 条随图打印可加 --limit N）。

依赖：仅 Python 3 标准库。
字段格式与 msMemScope 官方《输出文件说明》一致（Attr 为 {键:值,...} 格式）。
"""

import argparse
import array
import csv
import sys
from collections import Counter, defaultdict, deque

# memscope_dump csv 列名（与官方输出文件说明一致）
COL_ID = "ID"
COL_EVENT = "Event"
COL_EVENT_TYPE = "Event Type"
COL_TIMESTAMP = "Timestamp(ns)"
COL_DEVICE = "Device Id"
COL_PTR = "Ptr"
COL_ATTR = "Attr"
COL_STACK_PY = "Call Stack(Python)"

EVENT_MALLOC = "MALLOC"
EVENT_FREE = "FREE"
EVENT_SNAPSHOT = "SNAPSHOT"
EVENT_OOM = "OOM_DETAIL"

# 各命令需要从 Attr 中提取的键（减少解析量）
KEYS_CURVE = ("used", "device_used", "process_used", "total")
KEYS_OWNER = ("size", "owner")
KEYS_SNAPSHOT = ("size", "owner")
KEYS_LEAK = ("size", "owner", "shadow")
KEYS_PEAK = ("used", "device_used", "process_used")
KEYS_POOL = ("size", "owner", "total", "used")
KEYS_HALSEG = ("size", "owner", "alloc_type", "page_type")
KEYS_OOM = ("func", "req_size", "flag", "ret", "pool", "ptr", "size",
            "timestamp", "step", "kernel", "client")
KEYS_SNAPSHOT_STATS = ("total_mem", "free_mem", "reserved", "allocated",
                       "peak_reserved", "peak_allocated", "device_utilization",
                       "pt_utilization")
KEYS_STATS = ("owner", "shadow")
KEYS_TREND = ("used", "total")


def extract_attr(attr_str, keys):
    """从 Attr 字符串 '{k:v,k:v,...}' 中仅提取指定键，返回轻量 dict。

    用子串定位（C 实现的 str.find）替代全量 split，避免为每行构建完整
    attr dict ——百万行级 dump 的主要内存/CPU 开销来源。
    """
    result = {}
    if not attr_str or not keys:
        return result
    body = attr_str.strip()
    if body.startswith("{") and body.endswith("}"):
        body = body[1:-1]
    if not body.strip():
        return result
    for key in keys:
        marker = "," + key + ":"
        idx = body.find(marker)
        if idx == -1:
            # 键为首项（无前导逗号）时匹配 'key:' 开头
            if not body.startswith(key + ":"):
                continue
            start = len(key) + 1
        else:
            start = idx + len(marker)
        end = body.find(",", start)
        if end == -1:
            end = len(body)
        result[key] = body[start:end].strip()
    return result


def int_or_none(value):
    """Attr 值转 int，无法转换返回 None。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def iter_events(path, attr_keys=(), with_stack=False):
    """流式读取 dump csv，逐行产出轻量事件 dict。

    不排序、不保存全量事件；事件按文件记录顺序产出（msmemscope dump 为
    追加写入，文件顺序即事件顺序）。dict 仅含少量列 + attr_keys 中命中的键：
        {"Event": ..., "Event Type": ..., "Timestamp(ns)": ..., "Ptr": ...,
         "Attr": {"size": ..., "owner": ...}}
    with_stack=True 时额外携带 "Call Stack(Python)" 全列（供 --leak-candidates/
    --oom/--hal-segments/--stats 调用栈归因；未被采集时列不存在、dict 不含该键）。
    表头缺列或行短于表头时逐列安全跳过，不抛 KeyError（Event 键保持存在、值为
    空串；Attr 键保持存在、值为空 dict）。
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"文件为空或缺少表头: {path}")
        idx = {name: header.index(name) for name in header}
        for row in reader:
            e = {"Event": ""}
            if COL_EVENT in idx and idx[COL_EVENT] < len(row):
                e[COL_EVENT] = row[idx[COL_EVENT]]
            if COL_EVENT_TYPE in idx and idx[COL_EVENT_TYPE] < len(row):
                e[COL_EVENT_TYPE] = row[idx[COL_EVENT_TYPE]]
            if COL_TIMESTAMP in idx and idx[COL_TIMESTAMP] < len(row):
                e[COL_TIMESTAMP] = row[idx[COL_TIMESTAMP]]
            if COL_PTR in idx and idx[COL_PTR] < len(row):
                e[COL_PTR] = row[idx[COL_PTR]]
            if COL_ATTR in idx and idx[COL_ATTR] < len(row):
                e[COL_ATTR] = extract_attr(row[idx[COL_ATTR]], attr_keys)
            else:
                e[COL_ATTR] = {}
            if with_stack and COL_STACK_PY in idx and idx[COL_STACK_PY] < len(row):
                e[COL_STACK_PY] = row[idx[COL_STACK_PY]]
            yield e


def read_raw_rows(path):
    """读取 csv 原始行（不解析 Attr、不排序），返回 (fieldnames, rows)。"""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        fieldnames = next(reader, None)
        rows = [row for row in reader]
    if fieldnames is None:
        raise ValueError(f"文件为空或缺少表头: {path}")
    return fieldnames, rows


def timestamp_value(row, ts_index):
    """取原始行的时间戳数值；缺失/非法返回 None。"""
    if ts_index is None or len(row) <= ts_index:
        return None
    try:
        return int(row[ts_index].strip())
    except (TypeError, ValueError):
        return None


def rows_sorted(rows, ts_index):
    """检查有效时间戳序列是否非递减（缺失时间戳忽略）。"""
    prev = None
    for row in rows:
        ts = timestamp_value(row, ts_index)
        if ts is None:
            continue
        if prev is not None and ts < prev:
            return False
        prev = ts
    return True


def cmd_check(path):
    """完整性校验（流式，O(1) 内存）：表头、行数、列数一致、行关键字段完整、时间戳是否已有序。"""
    expected = [COL_ID, COL_EVENT, COL_EVENT_TYPE, COL_TIMESTAMP,
                COL_DEVICE, COL_PTR, COL_ATTR, COL_STACK_PY]
    problems = []
    ordered = True
    nrows = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            fieldnames = next(reader)
        except StopIteration:
            fieldnames = None
        if fieldnames is None or not fieldnames:
            print(f"[FAIL] {path}: 文件为空或缺少表头")
            return False
        missing = [c for c in expected if c not in fieldnames]
        if missing:
            problems.append(f"缺少关键列: {missing}（实际列: {fieldnames}）")
        ts_index = fieldnames.index(COL_TIMESTAMP) if COL_TIMESTAMP in fieldnames else None
        prev_ts = None
        for row in reader:
            nrows += 1
            if len(row) != len(fieldnames):
                problems.append(f"第 {nrows + 2} 行列数不一致: {len(row)} != {len(fieldnames)}")
            elif not (row[0].strip() and row[1].strip()):
                problems.append(f"第 {nrows + 2} 行关键字段（ID/Event）为空")
            if ordered and ts_index is not None and len(row) > ts_index:
                ts = timestamp_value(row, ts_index)
                if ts is not None:
                    if prev_ts is not None and ts < prev_ts:
                        ordered = False
                    prev_ts = ts
    if not nrows:
        problems.append("无数据行")
    if problems:
        print(f"[FAIL] {path} 完整性校验未通过:")
        for p in problems:
            print(f"  - {p}")
        return False
    print(f"[OK] {path}: 共 {nrows} 行，表头与列数一致，行关键字段完整")
    if ordered:
        print(f"     时间戳已按升序排列，无需排序")
    else:
        print(f"     时间戳未排序，请执行 --sort 排序（排序后重新校验）")
    return True


def cmd_sort(path, fieldnames, rows, overwrite):
    """按 Timestamp(ns) 升序排序。默认输出到 stdout；--overwrite 覆写原文件。"""
    ts_index = fieldnames.index(COL_TIMESTAMP) if COL_TIMESTAMP in fieldnames else None
    if rows_sorted(rows, ts_index):
        print(f"[SKIP] {path}: 时间戳已有序，无需重复排序（不修改文件）")
        return
    if ts_index is None:
        print(f"[FAIL] 缺少 {COL_TIMESTAMP} 列，无法按时间戳排序")
        return

    def sort_key(row):
        ts = timestamp_value(row, ts_index)
        return (0, ts) if ts is not None else (1, 0)  # 缺失时间戳的行排最后

    sorted_rows = sorted(rows, key=sort_key)
    first_ts = timestamp_value(sorted_rows[0], ts_index)
    last_ts = timestamp_value(sorted_rows[-1], ts_index)

    if overwrite:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            writer.writerows(sorted_rows)
        print(f"[OK] {path}: 已按时间戳排序并覆写原文件"
              f"（首行 ts={first_ts}，末行 ts={last_ts}，共 {len(sorted_rows)} 行）")
    else:
        # Windows 文本模式 stdout 会把 '\n' 翻译为 '\r\n'，故这里用 '\n' 终止符可
        # 在两端得到正确行尾（Linux 无翻译、Windows 一次翻译）；若用默认 '\r\n'
        # 在 Windows 重定向时会写出 '\r\r\n'，造成 csv 奇偶行错位。
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(fieldnames)
        writer.writerows(sorted_rows)
        print(f"[OK] 排序结果已输出到标准输出；使用 --overwrite 可覆写原文件"
              f"（首行 ts={first_ts}，末行 ts={last_ts}）", file=sys.stderr)


def owner_stats(events, metric="peak", key_field=None):
    """按 owner（或指定字段）聚合显存占用统计（活跃块口径，流式）。

    metric: 'peak' 为活跃占用峰值（该 owner 同时活跃的块大小之和的最大值）；
            'total' 为累计申请量（该 owner 所有 MALLOC size 之和）。
    key_field='event_type' 时按事件来源（内存池维度）聚合；否则按 owner
    （decode 到 owner 时按 owner，未 decode 时 fallback 到事件来源）。
    返回 key -> 统计值字典。
    同一 ptr 的 malloc -> free 为一个生命周期；FREE 通过 ptr 找到原申请并扣除。
    事件按文件顺序给出即可（不依赖时间戳排序），与顺序相关的仅 ptr 生命周期匹配。
    """
    active = {}                       # ptr -> (size, owner)
    owner_active = defaultdict(int)   # owner -> 当前活跃总量
    owner_stat = defaultdict(int)     # owner -> peak / total

    for e in events:
        event = e.get(COL_EVENT)
        attr = e.get(COL_ATTR, {})
        if event == EVENT_MALLOC:
            ptr = e.get(COL_PTR)
            size = int_or_none(attr.get("size")) or 0
            if key_field == "event_type":
                owner = e.get(COL_EVENT_TYPE) or "UNKNOWN"
            else:
                owner = attr.get("owner") or e.get(COL_EVENT_TYPE) or "UNKNOWN"
            if metric == "total":
                owner_stat[owner] += size
            if ptr in active:  # 同地址在未释放时被再次申请（数据异常或复用），先扣除旧块
                prev_size, prev_owner = active[ptr]
                owner_active[prev_owner] -= prev_size
            active[ptr] = (size, owner)
            owner_active[owner] += size
            owner_stat[owner] = max(owner_stat[owner], owner_active[owner])
        elif event == EVENT_FREE:
            ptr = e.get(COL_PTR)
            if ptr in active:
                prev_size, prev_owner = active[ptr]
                del active[ptr]
                owner_active[prev_owner] -= prev_size
                if owner_active[prev_owner] < 0:
                    owner_active[prev_owner] = 0
                owner_stat[prev_owner] = max(owner_stat[prev_owner], owner_active[prev_owner])
    return dict(owner_stat)


def snapshot_at(path, ts, pool=None):
    """统计指定时间点（<=ts 的最近事件时刻）的活跃块（流式）。

    要求文件按时间戳有序：流式检查逆序（回退时间戳且 <=ts 时）立即报错，
    提示先执行 --sort。pool 指定时只保留该内存池（Event Type）维度的块。

    返回 dict {by_owner(owner→size), by_pool(池→size), total, total_count,
               owner_count(owner→块数), pool_count(池→块数)}。
    """
    active = {}
    prev_ts = None
    for e in iter_events(path, KEYS_SNAPSHOT):
        event = e.get(COL_EVENT)
        if event not in (EVENT_MALLOC, EVENT_FREE):
            continue
        ts_event = int_or_none(e.get(COL_TIMESTAMP))
        if ts_event is not None:
            if prev_ts is not None and ts_event < prev_ts:
                raise ValueError(
                    f"文件未按时间戳升序排列（回退发生在 ts={ts_event} < {prev_ts}），"
                    f"--at-timestamp 要求有序 dump，请先执行 --sort [--overwrite]")
            prev_ts = ts_event
            if ts_event > ts:
                break
        if event == EVENT_MALLOC:
            ptr = e.get(COL_PTR)
            size = int_or_none(e.get(COL_ATTR, {}).get("size")) or 0
            owner = e.get(COL_ATTR, {}).get("owner") or e.get(COL_EVENT_TYPE) or "UNKNOWN"
            etype = e.get(COL_EVENT_TYPE) or "UNKNOWN"
            if pool and etype != pool:
                continue
            active[ptr] = (size, owner, etype)
        elif event == EVENT_FREE:
            active.pop(e.get(COL_PTR), None)
    by_owner = defaultdict(int)
    by_pool = defaultdict(int)
    owner_count = defaultdict(int)
    pool_count = defaultdict(int)
    for size, owner, etype in active.values():
        by_owner[owner] += size
        by_pool[etype] += size
        owner_count[owner] += 1
        pool_count[etype] += 1
    return {"by_owner": dict(by_owner), "by_pool": dict(by_pool),
            "total": sum(by_owner.values()), "total_count": len(active),
            "owner_count": dict(owner_count), "pool_count": dict(pool_count)}


def stream_curve(path, on_unsorted=None):
    """流式产出曲线数据点 (ts, event, used, device_used, process_used, total)。

    只产出 MALLOC/FREE 且 attr 含 used 的事件（与 dump 字段语义一致），
    每点 6 元组逐行 yield，不保存全量列表。
    on_unsorted：可选回调（无参），检测到时间戳回退（文件未排序）时调用，
    dump 仍按原顺序产出。
    """
    prev_ts = None
    for e in iter_events(path, KEYS_CURVE):
        if e.get(COL_EVENT) not in (EVENT_MALLOC, EVENT_FREE):
            continue
        attr = e.get(COL_ATTR, {})
        if "used" not in attr:
            continue
        ts = int_or_none(e.get(COL_TIMESTAMP))
        if ts is not None:
            if prev_ts is not None and ts < prev_ts and on_unsorted is not None:
                on_unsorted()
            prev_ts = ts
        yield (ts,
               e.get(COL_EVENT),
               int_or_none(attr.get("used")),
               int_or_none(attr.get("device_used")),
               int_or_none(attr.get("process_used")),
               int_or_none(attr.get("total")))


def format_bytes(value):
    if value is None:
        return "N/A"
    if value < 0:
        return str(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:.2f}{unit}" if unit != "B" else f"{value}B"
        value /= 1024
    return f"{value}B"


def ascii_chart(ts_arr, val_arr, label, bins=40, height=20):
    """按列式数组（array('q')）渲染 ASCII 曲线（分桶取最大值，保持峰值语义）。

    输入为与 ts_arr 等长的有效值数组（均 >=0 已过滤），单遍分桶：
    先扫一遍求整体 vmax 与 ts 范围，再按桶写入（桶累加最大值）。
    """
    n = len(ts_arr)
    if n == 0:
        return f"{label} 曲线: (无数据点)"
    ts_min = min(ts_arr)
    ts_max = max(ts_arr)
    span = max(ts_max - ts_min, 1)
    bucket_max = [0] * bins
    for i in range(n):
        idx = min(int((ts_arr[i] - ts_min) * bins / span), bins - 1)
        v = val_arr[i]
        if v > bucket_max[idx]:
            bucket_max[idx] = v
    vmax = max(bucket_max) or 1
    lines = [f"{label} 曲线（分桶 {bins}，桶内取峰值，纵轴单位字节）:"]
    for level in range(height, 0, -1):
        bar = "".join("#" if b >= vmax * level / height else " " for b in bucket_max)
        label_str = format_bytes(vmax * level / height)
        lines.append(f"{label_str:>12} |{bar}")
    lines.append(f"{'':>12} +" + "-" * bins)
    lines.append(f"{'ts 起点':>12} |{ts_min}    ts 终点: {ts_max}  （1 格 ≈ {span / bins:.0f} ns，共 {n} 个数据点）")
    return "\n".join(lines)


def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)))


def cmd_group_by(path, group_key, metric, pool=None):
    events = iter_events(path, KEYS_OWNER)
    if pool:
        events = (e for e in events if e.get(COL_EVENT_TYPE) == pool)
    stats = owner_stats(events, metric=metric,
                        key_field="event_type" if group_key == "event_type" else None)
    total = sum(stats.values()) or 1
    rows = []
    for owner, value in sorted(stats.items(), key=lambda kv: -kv[1]):
        rows.append((owner, value, format_bytes(value), f"{value / total * 100:.1f}%"))
    header = "owner" if group_key == "owner" else "event_type"
    print_table(rows, [header, f"{metric}(字节)", f"{metric}", "占比"])
    if not pool:
        print(f"\n注：未指定维度，聚合结果跨内存池混聚（同一物理内存同时有 HAL 事件与池事件两套"
              f"记录，数值不可相加）；显存拆解优先从驱动层全集开始 --pool HAL，"
              f"池内用途再 --pool PTA（HOST 暂不分析）。")
    print(f"\n注：占比分母为全部 {group_key} 的{metric}之和（{format_bytes(total)}）；"
          f"owner 未全覆盖时与 used 曲线存在差值属正常。")


# 内存池（维度）枚举：Event Type 即池维度；HCCL/APP/GE/RUNTIME 等组件内存
# 均落在 HAL 池（owner 形如 CANN@HCCL，见 decompose_analyzer.cpp InitOwner）
POOLS = ("PTA", "PTA_WORKSPACE", "ATB", "MINDSPORE", "HAL", "HOST", "HOST_PINNED")
FRAMEWORK_FIRST = set(POOLS)


def owner_segments(owner):
    """owner 拆段并丢弃框架段（FRAMEWORK 级=分配器来源名）。

    owner 多级结构 = 框架@组件@流程@细化@...（空段跳过，深度天然不一致）：
    - PTA 池  → "PTA@fsdp2@all_gather_output@ops" → [fsdp2, all_gather_output, ops]
    - HAL 池  → "CANN@HCCL@comm_0" → [HCCL, comm_0]（框架段为 CANN@xxx，非 HAL@）
    - 无框架段直标块（weight@ops）→ [weight, ops]（首段非框架名则不丢弃）
    丢弃后为空（块未打任何标签）→ []，由调用方归为"(未标注)"。
    """
    parts = owner.split("@")
    if parts and (parts[0] in FRAMEWORK_FIRST or parts[0] == "CANN"):
        parts = parts[1:]
    return parts


def owner_group_key(owner):
    """维度内一级归口：丢弃框架段后取首段（HAL 池 → HCCL/APP/GE，PTA 池 → weight/optimizer/fsdp2）。"""
    segs = owner_segments(owner)
    return segs[0] if segs else "(未标注)"


def owner_tree_lines(by_owner, owner_count, total):
    """把完整 owner 链（丢弃框架段后）挂成嵌套树，输出缩进树行列表。

    owner 链按级别槽位拆段（深度天然不一致），树体按语义层级下钻，
    天然容纳深度错位；同链块聚合 size/块数。节点按 size 降序。
    """
    root = {"children": {}}
    for owner, size in by_owner.items():
        segs = owner_segments(owner) or ["(未标注)"]
        node = root
        for seg in segs:
            child = node["children"].setdefault(seg, {"size": 0, "count": 0, "children": {}})
            child["size"] += size
            child["count"] += owner_count.get(owner, 0)
            node = child
    lines = []

    def walk(node, prefix):
        children = sorted(node["children"].items(), key=lambda kv: -kv[1]["size"])
        for i, (name, child) in enumerate(children):
            last = i == len(children) - 1
            branch = "└── " if last else "├── "
            lines.append(f"{prefix}{branch}{name} {format_bytes(child['size'])} "
                         f"{child['size'] / total * 100:.1f}% ({child['count']} 块)")
            walk(child, prefix + ("    " if last else "│   "))

    walk(root, "")
    return lines


def cmd_at_timestamp(path, ts, pool=None):
    # 默认从驱动层全集（HAL 事件）拆解：HAL 为驱动层打包全集，包含 PTA/ATB/
    # MindSpore 等框架池从驱动申请的大段（owner=CANN@APP）；同一物理内存另有
    # 池事件子视图，两者数值嵌套不可相加，故不存在"池并列汇总"。HOST 暂不分析。
    if pool is None:
        pool = "HAL"
    try:
        snap = snapshot_at(path, ts, pool=pool)
    except ValueError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)
    total = snap["total"]
    if not snap["by_owner"]:
        print(f"时间点 {ts} 无活跃内存块（可能早于首条事件或晚于末条事件）"
              + (f"，或 {pool} 池在该时刻无活跃块" if pool else ""))
        return
    print(f"时间点 {ts} 活跃总量: {format_bytes(total)}（共 {snap['total_count']} 块，"
          f"{pool} 池事件，筛选条件=生命周期包含该时间点的内存块）")
    # 维度（内存池）内拆解：一级归口表 + 层级树
    l1_size = defaultdict(int)
    l1_count = defaultdict(int)
    for owner, size in snap["by_owner"].items():
        l1_size[owner_group_key(owner)] += size
        l1_count[owner_group_key(owner)] += snap["owner_count"].get(owner, 0)
    if pool == "HAL":
        view_label = "驱动层全集（默认拆解入口；含框架池段）"
    elif pool in ("HOST", "HOST_PINNED"):
        view_label = "主机侧锁页内存（当前暂不纳入显存拆解分析）"
    else:
        view_label = "HAL 之下的第二层池内视图（物理段在 HAL 事件的 CANN@APP 中）"
    print(f"\n[{pool} 池拆解]（{view_label}；维度内一级归口，含各子级）：")
    rows = [(k, v, format_bytes(v), f"{v / total * 100:.1f}%", l1_count[k])
            for k, v in sorted(l1_size.items(), key=lambda kv: -kv[1])]
    print_table(rows, [f"{pool} 一级归口", "占用(字节)", "占用", "占比", "块数"])
    print(f"\n[{pool} 池层级分配树]（丢弃框架段，按 组件/流程/细化 下钻）：")
    for line in owner_tree_lines(snap["by_owner"], snap["owner_count"], total):
        print(line)
    if pool == "HAL":
        print("\n口径：HAL 事件为驱动层全集（≈ 进程驱动显存，与 process_used 同源量级），"
              "其中 CANN@APP 即 PTA/ATB/MindSpore 框架池向驱动申请的大段；"
              "池内用途拆解 → --pool PTA；HOST（锁页/主机侧）暂不分析。"
              "同一物理内存同时有池事件子视图，两类数值不可相加。")
    else:
        tail = "；当前暂不纳入显存拆解分析" if pool in ("HOST", "HOST_PINNED") else \
            "；池内总量 ≈ HAL 事件中 CANN@APP 段合计，与 HAL 数值不可相加；泄漏/增长分析建议回到驱动层（不指定 --pool）视角"
        print(f"\n口径：池事件是同一物理内存的池内视图{tail}。")


def leak_candidates(path, pool=None):
    """流式扫描"无真实释放"的内存块（泄漏候选），O(活跃块数) 内存。

    语义：采集窗口内申请、至采集结束**无真实 FREE** 的块——同一 Ptr 出现的
    FREE 若全部为 attr.shadow=true（幽灵/补齐释放，无真实释放动作），该块不
    存在真正的释放行为，即泄漏候选（对应 leak_diagnosis_guide 算法 C 的块级
    筛选）。幽灵机制下 MALLOC/FREE 必然配对，切勿用"配对"否认候选。

    shadow FREE 有两类来源（非采集期正常释放 / 进程退出补齐），判别见
    leak_diagnosis_guide §4——本命令输出原始信号（shadow FREE 时间戳、
    申请时间跨度），由分析侧结合采集状态判别。

    pool 指定时只扫描该内存池（Event Type）维度的事件。

    返回 (candidates, malloc_total)；candidates 为 list of dict：
    {ptr, size, owner, alloc_ts, shadow_free_count, last_shadow_ts,
     stack_first}。地址复用（未释放即再次 MALLOC）时旧块不列为候选
    （与 owner_stats 的保守口径一致），避免 HAL 段复用误报。
    """
    active = {}        # ptr -> 块信息
    candidates = []
    malloc_total = 0
    for e in iter_events(path, KEYS_LEAK, with_stack=True):
        event = e.get(COL_EVENT)
        if event not in (EVENT_MALLOC, EVENT_FREE):
            continue
        if pool and e.get(COL_EVENT_TYPE) != pool:
            continue
        attr = e.get(COL_ATTR, {})
        ptr = e.get(COL_PTR)
        if event == EVENT_MALLOC:
            if ptr in active:      # 地址复用：旧块按不候选处理（同 owner_stats 口径）
                del active[ptr]
            size = int_or_none(attr.get("size")) or 0
            stack = (e.get(COL_STACK_PY) or "").strip().splitlines()
            active[ptr] = {
                "ptr": ptr, "size": size, "owner": attr.get("owner") or e.get(COL_EVENT_TYPE) or "UNKNOWN",
                "alloc_ts": int_or_none(e.get(COL_TIMESTAMP)),
                "shadow_free_count": 0, "last_shadow_ts": None,
                "stack_first": stack[0].strip() if stack else "",
            }
            malloc_total += size
        elif event == EVENT_FREE and ptr in active:
            if attr.get("shadow") == "true":
                blk = active[ptr]
                blk["shadow_free_count"] += 1
                blk["last_shadow_ts"] = int_or_none(e.get(COL_TIMESTAMP))
            else:                  # 真实释放
                del active[ptr]
    for blk in active.values():
        candidates.append(blk)
    return candidates, malloc_total


def cmd_leak_candidates(path, detail, pool=None):
    candidates, malloc_total = leak_candidates(path, pool=pool)
    if not candidates:
        print("[OK] 未发现“无真实释放”的块（采集窗口内申请均有真实 FREE）——"
              "注意：退出补齐的 shadow FREE 也会配对成功，本结论仅基于块级释放真伪，"
              "常驻内存（总量稳定）不属于泄漏，见 leak_diagnosis_guide §4")
        return
    total = sum(c["size"] for c in candidates)
    with_shadow = [c for c in candidates if c["shadow_free_count"] > 0]
    print(f"[泄漏候选扫描] 共 {len(candidates)} 块 / {format_bytes(total)}，"
          f"占全部申请量 {format_bytes(malloc_total)} 的 {total / malloc_total * 100:.1f}%"
          + (f"（{pool} 池维度）" if pool else ""))
    if not pool:
        print("  注意：未指定维度，聚合跨内存池混聚（同一物理内存有 HAL 事件与池事件两套记录，"
              "数值不可相加）；泄漏扫描建议从驱动层全集 --pool HAL 开始，池内 --pool PTA")
    print(f"  其中含 shadow FREE（工具补齐释放，无真实释放动作）: {len(with_shadow)} 块；"
          f"无任何 FREE 事件: {len(candidates) - len(with_shadow)} 块")
    # 按完整 owner 聚合
    by_owner = defaultdict(lambda: [0, 0])
    for c in candidates:
        by_owner[c["owner"]][0] += c["size"]
        by_owner[c["owner"]][1] += 1
    print("\n按 owner（完整层级）:")
    rows = [(k, v[1], v[0], format_bytes(v[0]), f"{v[0] / total * 100:.1f}%")
            for k, v in sorted(by_owner.items(), key=lambda kv: -kv[1][0])]
    print_table(rows, ["owner", "块数", "占用(字节)", "占用", "占比"])
    # 维度内一级归口（丢弃框架段，含层级包含关系）
    l1 = defaultdict(lambda: [0, 0])
    for c in candidates:
        l1[owner_group_key(c["owner"])][0] += c["size"]
        l1[owner_group_key(c["owner"])][1] += 1
    print("\n按维度内一级归口（丢弃框架段后首段: PTA→weight/fsdp2…，HAL→HCCL/APP/GE…）:")
    rows = [(k, v[1], v[0], format_bytes(v[0]), f"{v[0] / total * 100:.1f}%")
            for k, v in sorted(l1.items(), key=lambda kv: -kv[1][0])]
    print_table(rows, ["一级归口", "块数", "占用(字节)", "占用", "占比"])
    # 申请时间跨度（早-常驻 vs 晚-新增）
    ts_list = [c["alloc_ts"] for c in candidates if c["alloc_ts"] is not None]
    if ts_list:
        print(f"\n候选块申请时间跨度: {min(ts_list)} ~ {max(ts_list)} ns"
              f"（整体申请较晚 → 后期新增未释放；申请早且总量稳定 → 常驻内存，见 "
              f"leak_diagnosis_guide §4）")
    # 调用栈归因（首帧聚合）
    by_stack = defaultdict(lambda: [0, 0])
    for c in candidates:
        key = c["stack_first"] or "(无 Python 调用栈)"
        by_stack[key][0] += c["size"]
        by_stack[key][1] += 1
    print("\n按申请调用栈首帧归因（TOP）:")
    rows = [(k, v[1], v[0], format_bytes(v[0]), f"{v[0] / total * 100:.1f}%")
            for k, v in sorted(by_stack.items(), key=lambda kv: -kv[1][0])[:20]]
    print_table(rows, ["Call Stack(Python) 首帧", "块数", "占用(字节)", "占用", "占比"])
    # 明细采样
    if detail and detail > 0:
        shown = candidates[:detail]
        print(f"\n候选块明细（前 {len(shown)} 块，--detail 0 不输出明细）:")
        rows = []
        for c in shown:
            rows.append((c["ptr"], c["size"], format_bytes(c["size"]), c["owner"],
                         c["alloc_ts"] or "-",
                         c["last_shadow_ts"] if c["shadow_free_count"] else "-",
                         c["shadow_free_count"], c["stack_first"] or "-"))
        print_table(rows, ["Ptr", "size", "占用", "owner", "申请时间(ns)",
                           "最后shadow FREE(ns)", "shadow FREE数", "调用栈首帧"])
    print("\n判别提示：shadow FREE 有两类来源（非采集期正常释放 / 进程退出补齐），"
          "结合采集状态判别后再下泄漏结论；常驻内存≠泄漏，见 leak_diagnosis_guide §4。")


def cmd_curve(path, ascii_mode, bins, limit):
    """绘制显存曲线：流式单遍扫描，不将全量事件载入内存。

    ASCII 模式：按列式数组（array('q')）收集各曲线键的数据点（O(事件数)
    紧凑存储），渲染分桶曲线；数据点表默认不再全量打印（旧行为在百万行
    下会刷屏），如需随图打印前 N 条可加 --limit N。
    非 ASCII 模式：逐点流式打印（O(1) 内存），--limit N 限制打印条数。
    文件未按时间戳排序时在 stderr 给出提示（曲线按文件顺序描点）。
    """
    unsorted_hint = ["警告: 文件时间戳未排序，曲线按文件记录顺序描点，"
                     "建议先执行 --sort [--overwrite] 后重新绘图（准确性校验用 --check）"]
    warned = [False]

    def note_unsorted():
        if not warned[0]:
            print(unsorted_hint[0], file=sys.stderr)
            warned[0] = True

    if ascii_mode:
        # 键 -> (ts 数组, 值数组)；device_used/process_used/used 均可能缺失（-1），
        # ascii_chart 只画 >=0 的有效值，故每个键收集一份有效数组
        ts_lists = {"used": array.array("q"), "device_used": array.array("q")}
        val_lists = {"used": array.array("q"), "device_used": array.array("q")}
        total_points = 0
        for ts, event, used, device_used, process_used, total in stream_curve(path, note_unsorted):
            total_points += 1
            for key, val in (("used", used), ("device_used", device_used)):
                if val is not None and val >= 0 and ts is not None:
                    ts_lists[key].append(ts)
                    val_lists[key].append(val)
        if total_points == 0:
            print("无曲线数据点（dump 中缺少含 used 统计的 MALLOC/FREE 事件，确认采集配置）")
            return
        print(ascii_chart(ts_lists["used"], val_lists["used"], label="used", bins=bins))
        print()
        dev_pts = len(ts_lists["device_used"])
        if dev_pts == 0:
            print("device_used 曲线: (无有效数据：环境可能不支持该统计)")
        else:
            print(ascii_chart(ts_lists["device_used"], val_lists["device_used"],
                              label="device_used", bins=bins))
        print()
        if limit and limit > 0:
            print(f"数据点（ts, event, used, device_used, process_used, total，前 {limit} 条）：")
            shown = 0
            for ts, event, used, device_used, process_used, total in stream_curve(path, note_unsorted):
                print(f"{ts}\t{event}\t{used}\t{device_used}\t{process_used}\t{total}")
                shown += 1
                if shown >= limit:
                    break
        else:
            print(f"共 {total_points} 个数据点；数据点全量输出请使用不带 --ascii 的 --curve"
                  f"（或加 --limit N 随图打印前 N 条）")
    else:
        print("ts\tEvent\tused\tdevice_used\tprocess_used\ttotal")
        count = 0
        all_count = 0
        for ts, event, used, device_used, process_used, total in stream_curve(path, note_unsorted):
            all_count += 1
            if limit and limit > 0 and count >= limit:
                continue
            count += 1
            print(f"{ts}\t{event}\t{used}\t{device_used}\t{process_used}\t{total}")
        if limit and limit > 0:
            print(f"\n共 {all_count} 个数据点（--limit 限制显示前 {count} 条）", file=sys.stderr)
        else:
            print(f"\n共 {all_count} 个数据点（事件驱动曲线，仅 MALLOC/FREE 时刻有点）")


def positive_int(value):
    """CLI 正整数参数校验（--bins/--buckets 下限 1，0/负值直接报参数错误，避免空桶数组越界）。"""
    try:
        v = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"须为正整数（收到 {value!r}）")
    if v < 1:
        raise argparse.ArgumentTypeError(f"须 >= 1（收到 {value}）")
    return v


def parse_size(value):
    """解析带单位的大小（'100MB'/'1KB'/'1048576'/'2GB'），单位不区分大小写；无法解析抛 ValueError。"""
    text = str(value).strip().upper()
    units = {"KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4, "B": 1}
    for suffix, mul in units.items():
        if text.endswith(suffix):
            body = text[:-len(suffix)].strip()
            try:
                return int(float(body) * mul)
            except ValueError:
                break
    try:
        return int(text)
    except ValueError:
        raise ValueError(f"无法解析大小: {value!r}（支持裸字节或 KB/MB/GB 后缀，如 100MB）")


def cmd_peak(path, key="used", pool=None):
    """定位曲线峰值点（流式单遍，O(1) 内存）：输出该键最大值、时间戳与所在事件。

    峰值时间戳可直接衔接 --at-timestamp <ts> 做时刻拆解（--curve 的人眼替代）。
    device_used 等键可能为 -1（环境不支持），负值自动过滤。
    """
    best = None
    total_points = 0
    for e in iter_events(path, KEYS_PEAK):
        if e.get(COL_EVENT) not in (EVENT_MALLOC, EVENT_FREE):
            continue
        if pool and e.get(COL_EVENT_TYPE) != pool:
            continue
        attr = e.get(COL_ATTR, {})
        if key not in attr:
            continue
        val = int_or_none(attr.get(key))
        if val is None or val < 0:
            continue
        total_points += 1
        if best is None or val > best[0]:
            best = (val, int_or_none(e.get(COL_TIMESTAMP)),
                    e.get(COL_EVENT_TYPE), e.get(COL_EVENT))
    if best is None:
        print(f"[FAIL] 无有效 {key} 曲线数据点（dump 中缺少含 {key} 的 MALLOC/FREE 事件"
              + (f"，或 {pool} 池维度无事件" if pool else "")
              + "；device_used 可能为环境不支持）")
        return
    val, ts, etype, event = best
    print(f"[峰值定位] {key} 峰值: {format_bytes(val)}（共 {total_points} 个有效数据点，"
          f"事件 {event}/{etype} 时刻后的统计值）")
    print(f"  峰值时间戳: {ts} ns")
    print(f"  衔接命令: python3 scripts/aggregate_dump.py <dump.csv> --at-timestamp {ts}"
          + (f" --pool {etype}" if pool else ""))
    if not pool:
        print("  注：未指定维度，峰值跨内存池混聚（HAL 与池事件统计键层级嵌套、数值不可相加）；"
              "按维度定位加 --pool（如 --pool HAL 驱动层全集 / --pool PTA 池内）")


def cmd_oom(path, detail):
    """汇总 OOM_DETAIL 诊断（流式，内存 O(OOM 记录数)）。

    三类子事件分表输出：OOM_TRIGGER（触发操作）、OOM_TOP_ALLOC（OOM 时最大占用）、
    OOM_RECENT_ALLOC（最近分配）。剩余空间取 OOM 时刻前最近的 SNAPSHOT free_mem，
    用于量化推断提示（启发式，非结论）：req_size ≥80% 剩余 → 单次巨量；RECENT 前5
    条同栈 ≥3 次 → 同栈重复分配；前5条 size 累计 >3×剩余 → 接力耗尽。
    --detail 控制各表条数（默认 10；0=不输出明细，与 --leak-candidates 一致）。
    """
    triggers, recents, tops = [], [], []
    last_snap = None
    for e in iter_events(path, KEYS_OOM + KEYS_SNAPSHOT_STATS, with_stack=True):
        event = e.get(COL_EVENT)
        attr = e.get(COL_ATTR, {})
        if event == EVENT_SNAPSHOT:
            ts = int_or_none(e.get(COL_TIMESTAMP))
            if ts is not None and "free_mem" in attr:
                last_snap = {"ts": ts,
                             "total_mem": int_or_none(attr.get("total_mem")),
                             "free_mem": int_or_none(attr.get("free_mem")),
                             "device_utilization": int_or_none(attr.get("device_utilization")),
                             "pt_utilization": int_or_none(attr.get("pt_utilization"))}
            continue
        if event != EVENT_OOM:
            continue
        stack = (e.get(COL_STACK_PY) or "").strip().splitlines()
        row = {"ts": int_or_none(e.get(COL_TIMESTAMP)),
               "etype": e.get(COL_EVENT_TYPE) or "UNKNOWN",
               "stack_first": stack[0].strip() if stack else "",
               "snap": dict(last_snap) if last_snap else None,
               "attr": attr}
        if row["etype"] == "OOM_TRIGGER":
            triggers.append(row)
        elif row["etype"] == "OOM_TOP_ALLOC":
            tops.append(row)
        elif row["etype"] == "OOM_RECENT_ALLOC":
            recents.append(row)
    if not (triggers or recents or tops):
        print("[OK] dump 中无 OOM_DETAIL 记录（采集时未配置 --analysis=oom[:K]，或 OOM 未发生在采集区间内）")
        return
    print(f"[OOM 诊断汇总] OOM_TRIGGER {len(triggers)} 条 / OOM_TOP_ALLOC {len(tops)} 条 / "
          f"OOM_RECENT_ALLOC {len(recents)} 条"
          + ("；剩余空间取最近 SNAPSHOT free_mem" if last_snap else "；⚠️ 无 SNAPSHOT，剩余空间不可量化"))
    if triggers:
        print("\n[1] OOM_TRIGGER（触发操作）:")
        rows = []
        for r in triggers:
            a = r["attr"]
            req = int_or_none(a.get("req_size"))
            free_step = ""
            if (r["snap"] and r["snap"]["free_mem"] is not None and req is not None
                    and r["snap"]["free_mem"] > 0):
                free_step = f"req/剩余={req / r['snap']['free_mem'] * 100:.0f}%"
            rows.append((r["ts"], a.get("func") or "-",
                         req if req is not None else "-",
                         format_bytes(req) if req is not None else "-",
                         a.get("ret") or "-", free_step))
        print_table(rows, ["时间戳(ns)", "func", "req_size(字节)", "req_size", "ret", "量化提示"])
    limit = detail if detail and detail > 0 else 0
    if tops and limit:
        print(f"\n[2] OOM_TOP_ALLOC（OOM 时刻最大占用，按 size 降序"
              + ("；--detail 控制条数" if detail else "") + "）:")
        rows = []
        for r in sorted(tops, key=lambda kv: -(int_or_none(kv["attr"].get("size")) or 0))[:limit]:
            a = r["attr"]
            size = int_or_none(a.get("size"))
            rows.append((r["ts"], a.get("ptr") or "-", size if size is not None else "-",
                         format_bytes(size) if size is not None else "-",
                         a.get("pool") or "-", a.get("step") or "-", a.get("kernel") or "-",
                         a.get("timestamp") or "-", r["stack_first"] or "-"))
        print_table(rows, ["事件时间(ns)", "Ptr", "size", "占用", "池", "step", "kernel",
                           "申请时间", "栈首帧"])
    if recents and limit:
        print(f"\n[3] OOM_RECENT_ALLOC（最近分配，按事件时间倒序"
              + ("；--detail 控制条数" if detail else "") + "）:")
        rows = []
        for r in list(reversed(recents))[:limit]:
            a = r["attr"]
            size = int_or_none(a.get("size"))
            rows.append((r["ts"], a.get("ptr") or "-", size if size is not None else "-",
                         format_bytes(size) if size is not None else "-",
                         a.get("pool") or "-", a.get("step") or "-", a.get("kernel") or "-",
                         a.get("timestamp") or "-", r["stack_first"] or "-"))
        print_table(rows, ["事件时间(ns)", "Ptr", "size", "占用", "池", "step", "kernel",
                           "申请时间", "栈首帧"])
    print("\n[4] 量化推断提示（启发式，供归因参考，非结论）:")
    remaining = last_snap["free_mem"] if (last_snap and last_snap["free_mem"] is not None) else None
    if remaining is None:
        print("  - 无可用 SNAPSHOT free_mem，跳过 req_size/累计型判定（补采 take_snapshot 后可量化）")
    for r in triggers:
        req = int_or_none(r["attr"].get("req_size"))
        if req is not None and remaining is not None and remaining > 0 and req >= 0.8 * remaining:
            print(f"  - OOM_TRIGGER(t={r['ts']}): req_size {format_bytes(req)} ≥ "
                  f"80%×剩余空间({format_bytes(remaining)}) → 单次巨量申请")
    if recents:
        first5 = recents[:5]
        cnt = Counter(r["stack_first"] for r in first5 if r["stack_first"])
        for s, c in cnt.items():
            if c >= 3:
                print(f"  - OOM_RECENT_ALLOC 前 5 条同栈「{s}」×{c} ≥3 → 同栈重复分配（循环内累积申请嫌疑）")
        s5 = sum(int_or_none(r["attr"].get("size")) or 0 for r in first5)
        if remaining is not None and remaining > 0 and s5 > 3 * remaining:
            print(f"  - OOM_RECENT_ALLOC 前 5 条 size 累计 {format_bytes(s5)} > "
                  f"3×剩余空间({format_bytes(remaining)}) → 一连串大分配接力耗尽")
    print("  - 归因顺序: 单次巨量（[1] req_size）→ 大头归属（[2] TOP）→ 分配模式（[3] RECENT）→"
          " 累积型（转泄漏诊断）；func 指向池扩容路径时按 SKILL B3⑤ 扩容失败 OOM 处理")


def cmd_fragmentation(path, pool=None):
    """池碎片率与扩容画像（流式单遍，内存 O(活跃块 + 扩容记录限定)）。

    依据池事件统计键：同池连续两条池事件间 total 变大 = 扩容（期间一般伴随 HAL 段申请，
    以缓冲的近邻 HAL 段作规模核对参考）；碎片率 = (total-used)/total（同一条事件，
    used>0 时刻）；扩容前 used/total = 池有效使用率。HAL 池无 total/used 统计键自动跳过。
    """
    stats = {}                                    # pool -> 统计 dict
    last = {}                                     # pool -> (ts, total, used)
    pending = {p: deque(maxlen=4096) for p in POOLS}   # pool -> 上次池事件以来的 HAL 段 (ts, size)
    for e in iter_events(path, KEYS_POOL):
        event = e.get(COL_EVENT)
        if event not in (EVENT_MALLOC, EVENT_FREE):
            continue
        etype = e.get(COL_EVENT_TYPE) or "UNKNOWN"
        attr = e.get(COL_ATTR, {})
        ts = int_or_none(e.get(COL_TIMESTAMP))
        if etype == "HAL" and event == EVENT_MALLOC:
            size = int_or_none(attr.get("size")) or 0
            for p in pending.values():
                p.append((ts, size))
            continue
        if "total" not in attr:
            continue
        if pool and etype != pool:
            continue
        total = int_or_none(attr.get("total"))
        used = int_or_none(attr.get("used"))
        if total is None:
            continue
        st = stats.setdefault(etype, {"expansions": [], "exp_trunc": 0, "events": 0,
                                      "frag_max": None, "frag_worst": None, "frag_n": 0,
                                      "frag_sum": 0.0, "last_frag": None, "low": []})
        st["events"] += 1
        prev = last.get(etype)
        if prev is not None and total > prev[1]:
            eff = (prev[2] / prev[1] * 100) if prev[1] > 0 and prev[2] is not None else None
            segs = pending.get(etype) or ()
            if len(st["expansions"]) < 500:
                st["expansions"].append({"ts": ts, "prev": prev[1], "new": total,
                                         "delta": total - prev[1], "eff": eff,
                                         "seg_n": len(segs),
                                         "seg_sum": sum(s for _, s in segs),
                                         "prev_used": prev[2], "prev_ts": prev[0]})
            else:
                st["exp_trunc"] += 1
        if used is not None and used > 0 and total > 0:
            frag = (total - used) / total * 100
            st["frag_n"] += 1
            st["frag_sum"] += frag
            if st["frag_max"] is None or frag > st["frag_max"]:
                st["frag_max"] = frag
            if st["frag_worst"] is None or frag > st["frag_worst"][0]:
                st["frag_worst"] = (frag, ts)
            st["last_frag"] = frag
            usage = used / total * 100
            lows = st["low"]
            if len(lows) < 5:
                lows.append((usage, ts, total, used))
                lows.sort(key=lambda x: x[0])
            elif usage < lows[-1][0]:
                lows[-1] = (usage, ts, total, used)
                lows.sort(key=lambda x: x[0])
        last[etype] = (ts, total, used)
        if etype in pending:      # 非 POOLS 枚举的池事件无 HAL 段缓冲（HAL 段只进枚举内 deque）
            pending[etype].clear()
    if not stats:
        print("[FAIL] 无池事件（dump 中无 total/used 统计键的池事件；确认采集配置 events 含 alloc/free）")
        return
    for p, st in sorted(stats.items()):
        exps = st["expansions"]
        print(f"\n[{p} 池碎片与扩容画像]（共 {st['events']} 条池事件"
              + (f"，{p} 池维度" if pool else "") + "）")
        print("  扩容次数: " + str(len(exps))
              + (f"（另有 {st['exp_trunc']} 条截断未显示）" if st["exp_trunc"] else "")
              + "（total 变大 = 池向驱动申请新段，正常扩容行为）")
        if exps:
            rows = []
            for x in exps:
                seg = f"{x['seg_n']}段/{format_bytes(x['seg_sum'])}" if x["seg_n"] else "-"
                rows.append((x["ts"],
                             f"{x['prev']}（{format_bytes(x['prev'])}）",
                             f"{x['new']}（{format_bytes(x['new'])}）",
                             format_bytes(x["delta"]),
                             f"{x['eff']:.1f}%" if x["eff"] is not None else "-",
                             seg))
            print("  扩容明细（时间戳 | 扩容前 total | 扩容后 total | 增量 | 扩容前 used/total | 期间 HAL 段）:")
            print_table(rows, ["时间戳(ns)", "扩容前 total", "扩容后 total", "增量",
                               "有效使用率", "伴随 HAL 段"])
        if st["frag_n"]:
            mean = st["frag_sum"] / st["frag_n"]
            print(f"  碎片率 (total-used)/total（used>0 时刻，{st['frag_n']} 个采样）: "
                  f"当前 {st['last_frag']:.1f}% / 峰值 {st['frag_max']:.1f}% / 均值 {mean:.1f}%"
                  + (f"（最差时刻 {st['frag_worst'][1]} ns）" if st["frag_worst"] else ""))
        else:
            print("  无 used>0 的池事件采样（池尚未实际使用）")
        if st["low"]:
            print("  有效使用率最低（used/total 最小，used>0）TOP5:")
            print_table([(t, f"{u:.1f}%", total, format_bytes(total), used, format_bytes(used))
                         for u, t, total, used in st["low"]],
                        ["时间戳(ns)", "used/total", "total(字节)", "total", "used(字节)", "used"])
        print("  阈值: <5% 正常 / 5~15% 偏高 / >15% 严重（Ascend NPU Snapshot Analyzer 口径，见 msmemscope_data §9）")
    print("\n注：池 total 只增不减是释放≠归还的正常行为（见 pta_memory_management §2.2）；"
          "伴随 HAL 段为期间近邻段（无字段归因，可能含其他池段，供核对扩容规模）；"
          "多流场景碎片率可能虚高（真碎片/流池残留/跨流延迟假性碎片无法区分），"
          "深层核对需 ascend-npu-snapshot-analyzer 补采 snapshot（msmemscope_data §9.2）。")


def cmd_hal_segments(path, min_size):
    """HAL 池大段清单（流式，内存 O(大段数)）：HAL MALLOC 且 size>=min_size 事件列表 + 汇总。

    用途：核对 CCL Buffer（~401MB/域，hccl_memory_detail §6.2 应然值）、MC2/AIV buffer
    （16MB/40MB）、框架池扩容段（如 20MB）；alloc_type=create 出现即 expandable_segments
    生效（halMemCreate 物理页，见 pta_memory_management §7.2）。
    """
    rows = []
    alloc_cnt = create_cnt = other_cnt = 0
    alloc_sum = create_sum = other_sum = 0
    big_cnt = big_sum = 0
    for e in iter_events(path, KEYS_HALSEG, with_stack=True):
        if e.get(COL_EVENT) != EVENT_MALLOC or e.get(COL_EVENT_TYPE) != "HAL":
            continue
        attr = e.get(COL_ATTR, {})
        size = int_or_none(attr.get("size")) or 0
        at = attr.get("alloc_type") or "UNKNOWN"
        if at == "alloc":
            alloc_cnt += 1
            alloc_sum += size
        elif at == "create":
            create_cnt += 1
            create_sum += size
        else:
            other_cnt += 1
            other_sum += size
        if size >= min_size:
            stack = (e.get(COL_STACK_PY) or "").strip().splitlines()
            rows.append((int_or_none(e.get(COL_TIMESTAMP)), size, format_bytes(size),
                         attr.get("owner") or "-",
                         at, attr.get("page_type") or "-", stack[0].strip() if stack else "-"))
            if size >= 100 * 1024 * 1024:
                big_cnt += 1
                big_sum += size
    if not rows:
        print(f"[OK] HAL 池无 size ≥ {format_bytes(min_size)} 的申请段"
              "（可调小 --min-size 查看更小段）")
        return
    total = sum(r[1] for r in rows)
    print(f"[HAL 池大段清单] size ≥ {format_bytes(min_size)}: {len(rows)} 段 / {format_bytes(total)}"
          f"（≥100MB 大段 {big_cnt} 段 / {format_bytes(big_sum)}；"
          "CCL Buffer 应然 ~401MB/域，MC2 +16MB，AIV +40MB，核对见 hccl_memory_detail §6.2）")
    print_table(sorted(rows, key=lambda kv: -kv[1])[:200],
                ["时间戳(ns)", "size", "占用", "owner", "alloc_type", "page_type",
                 "Call Stack(Python) 首帧"])
    if len(rows) > 200:
        print(f"  （仅展示 size 最大的 200 段，共 {len(rows)} 段；如需全量调大阈值 --min-size）")
    elif len(rows) < 10:
        print(f"  （仅 {len(rows)} 段，阈值偏大；可调小 --min-size 查看更小段）")
    print(f"\n分配类型汇总: alloc(普通段申请) {alloc_cnt} 段 / {format_bytes(alloc_sum)}；"
          f"create(expandable 物理页) {create_cnt} 段 / {format_bytes(create_sum)}"
          + (f"；其他 {other_cnt} 段 / {format_bytes(other_sum)}" if other_cnt else "")
          + (" → expandable_segments 生效（halMemCreate）" if create_cnt
             else " → 未见 create（expandable 未启用或非 expandable 配置）"))


def pool_exists(path, etype):
    """流式检查 dump 中是否存在指定 Event Type（供 --trend 默认双视图判定）。"""
    for e in iter_events(path, ()):
        if e.get(COL_EVENT_TYPE) == etype:
            return True
    return False


def cmd_trend(path, buckets, pool=None):
    """跨周期趋势（桶级，泄漏算法 A/B 自动化）：等宽时间桶输出 used/total 末值 + 判定。

    默认拆 HAL + PTA 双视图（PTA 在该 dump 存在时）：HAL=驱动层段级（段级 used 单调上升
    多为池缓存正常行为、仅作参考），PTA=池内块级（算法 A 判据 = 跨周期不回基线）；
    --pool 指定单一池。算法 B（total 预留增长，弱信号）仅池事件有 total，须与 A 联动。
    未排序文件仅提示（与 --curve 同口径），建议先 --check/--sort。
    """
    if pool is None:
        targets = ["HAL"]
        if pool_exists(path, "PTA"):
            targets.append("PTA")
        for i, p in enumerate(targets, 1):
            print(f"\n{'=' * 6} 视图 {i}/{len(targets)}: {p} 池 {'=' * 6}")
            _trend_one(path, buckets, p, dual=len(targets) > 1)
    else:
        _trend_one(path, buckets, pool, dual=False)


def _trend_one(path, buckets, pool, dual):
    used_ts = array.array("q")
    used_val = array.array("q")
    tot_ts = array.array("q")
    tot_val = array.array("q")
    n = 0
    prev_ts = None
    warned = False
    for e in iter_events(path, KEYS_TREND):
        if e.get(COL_EVENT) not in (EVENT_MALLOC, EVENT_FREE):
            continue
        if e.get(COL_EVENT_TYPE) != pool:
            continue
        attr = e.get(COL_ATTR, {})
        ts = int_or_none(e.get(COL_TIMESTAMP))
        if ts is not None:
            if prev_ts is not None and ts < prev_ts and not warned:
                print("警告: 文件时间戳未排序，趋势判定建议先执行 --sort [--overwrite]",
                      file=sys.stderr)
                warned = True
            prev_ts = ts
        used = int_or_none(attr.get("used"))
        if used is not None and used >= 0 and ts is not None:
            used_ts.append(ts)
            used_val.append(used)
            n += 1
        if "total" in attr:
            tot = int_or_none(attr.get("total"))
            if tot is not None and tot >= 0 and ts is not None:
                tot_ts.append(ts)
                tot_val.append(tot)
    if n == 0:
        print(f"[FAIL] 无 {pool} 池有效曲线数据点（确认采集 events 含 alloc/free，或该池无事件）")
        return
    ts_min, ts_max = min(used_ts), max(used_ts)
    span = max(ts_max - ts_min, 1)
    b_last, b_min, b_max = [None] * buckets, [None] * buckets, [0] * buckets
    b_tot = [None] * buckets
    for i in range(n):
        # used 桶：基于 used 首点，不会越下界；max(0,..) 双保险（防止后续首点口径变化引入负索引回绕）
        idx = max(0, min(int((used_ts[i] - ts_min) * buckets / span), buckets - 1))
        v = used_val[i]
        b_last[idx] = v
        b_min[idx] = v if b_min[idx] is None else min(b_min[idx], v)
        b_max[idx] = max(b_max[idx], v)
    for i in range(len(tot_ts)):
        # total 桶：used=-1 被过滤而 total 保留时，total 首点可早于 used 首点 →
        # (tot_ts - ts_min) 为负，缺下界会回绕写入 b_tot[-1]（桶尾被覆盖）
        idx = max(0, min(int((tot_ts[i] - ts_min) * buckets / span), buckets - 1))
        b_tot[idx] = tot_val[i]
    rows = []
    for i in range(buckets):
        lo = ts_min + span * i // buckets
        hi = ts_min + span * (i + 1) // buckets if i < buckets - 1 else ts_max
        fmt = lambda x: format_bytes(x) if x is not None else "-"
        rows.append((i, lo, hi, fmt(b_last[i]), fmt(b_min[i]), fmt(b_max[i]), fmt(b_tot[i])))
    print(f"\n[{pool} 池跨桶趋势]（等宽 {buckets} 桶，时间范围 {ts_min} ~ {ts_max} ns；"
          "事件驱动，桶内取末值/最小/最大）:")
    print_table(rows, ["桶", "起点(ns)", "终点(ns)", "used 末值", "used 最小", "used 最大", "total 末值"])
    lasts = [b_last[i] for i in range(1, buckets) if b_last[i] is not None]
    if len(lasts) < 2:
        print("\n（第 1 桶之后无数据点，无法做趋势判定）")
        return
    base = lasts[0]
    above = sum(1 for v in lasts if v > base)
    mono = all(lasts[i] >= lasts[i - 1] for i in range(1, len(lasts)))
    net = lasts[-1] - base
    caveat = ("HAL=驱动层段级视图，段级 used 单调上升是池缓存正常行为（释放≠归还），"
              + ("块级算法 A 判定见下方 PTA 视图" if dual
                 else "须 --pool PTA 看块级 used 才算算法 A")) if pool == "HAL" else \
             f"{pool} 池块级 used（算法 A 判据 = 跨周期不回基线）"
    print(f"\n[趋势判定]（剔除第 1 桶基线 {format_bytes(base)} 后的 {len(lasts)} 个桶）:")
    print(f"  末桶末值: {format_bytes(lasts[-1])}（净 {format_bytes(net)}，"
          f"{net / max(base, 1) * 100:+.1f}%）")
    print(f"  高于基线的桶: {above}/{len(lasts)}；桶末序列单调" + ("非降" if mono else "有回落"))
    if above == len(lasts) and mono:
        print(f"  ⚠️ 全部桶末高于基线且单调非降 → 增长强信号（{caveat}），"
              "进入泄漏诊断（leak_diagnosis_guide §3 算法 A/B 联动）")
    elif above == 0:
        print("  ✓ 无桶超过基线 → 未见跨周期增长（常驻/稳定）")
    else:
        print(f"  ⚠️ {above}/{len(lasts)} 桶高于基线（增长不连续，核对对应时间窗事件；{caveat}）")
    t_lasts = [b_tot[i] for i in range(buckets) if b_tot[i] is not None]
    if len(t_lasts) >= 2:
        t_mono = all(t_lasts[i] >= t_lasts[i - 1] for i in range(1, len(t_lasts)))
        if t_mono:
            print(f"  算法 B（弱信号）: total 桶末单调非减 {format_bytes(t_lasts[0])} → "
                  f"{format_bytes(t_lasts[-1])}（池预留增长；不单独作泄漏证据，须与算法 A 联动）")


def cmd_stats(path):
    """数据画像（流式单遍，O(1) 内存）：事件分布、时间范围、解析能力提示。

    分析前体检——判断"有什么数据、能走哪些分析路径"（owner/SNAPSHOT/OOM/Call Stack/shadow），
    对应 SKILL B1 数据产物确认与降级判断。
    """
    events = defaultdict(int)
    etypes = defaultdict(int)
    mallocs = frees = shadows = snapshots = oom = owned = stacks = 0
    ts_first = ts_last = None
    for e in iter_events(path, KEYS_STATS, with_stack=True):
        ev = e.get(COL_EVENT)
        if not ev:
            continue
        events[ev] += 1
        et = e.get(COL_EVENT_TYPE)
        if et:
            etypes[et] += 1
        ts = int_or_none(e.get(COL_TIMESTAMP))
        if ts is not None:
            ts_first = ts if ts_first is None else min(ts_first, ts)
            ts_last = ts if ts_last is None else max(ts_last, ts)
        attr = e.get(COL_ATTR, {})
        has_stack = (e.get(COL_STACK_PY) or "").strip()
        if ev == EVENT_MALLOC:
            mallocs += 1
            if attr.get("owner"):
                owned += 1
            if has_stack:
                stacks += 1
        elif ev == EVENT_FREE:
            frees += 1
            if attr.get("shadow") == "true":
                shadows += 1
        elif ev == EVENT_SNAPSHOT:
            snapshots += 1
        elif ev == EVENT_OOM:
            oom += 1
            if has_stack:
                stacks += 1
    total = sum(events.values())
    if total == 0:
        print("[FAIL] 文件无数据行")
        return
    print(f"[数据画像] 共 {total} 行，时间范围 {ts_first} ~ {ts_last} ns")
    print("\n按 Event:")
    print_table([(k, v, f"{v / total * 100:.1f}%")
                 for k, v in sorted(events.items(), key=lambda kv: -kv[1])],
                ["Event", "行数", "占比"])
    print("\n按 Event Type:")
    print_table([(k, v) for k, v in sorted(etypes.items(), key=lambda kv: -kv[1])],
                ["Event Type", "行数"])
    print(f"\n解析能力提示: MALLOC {mallocs}（含 owner {owned}，"
          "开启 decompose 才可按 owner 拆解）")
    print(f"  FREE {frees}（shadow:true {shadows} 为幽灵释放）; "
          f"SNAPSHOT {snapshots} 条（可量化 OOM 剩余空间/整卡水位）; "
          f"OOM_DETAIL {oom} 条（可走 --oom 汇总）; Call Stack(Python) 非空 {stacks} 条")
    print("  未开启 decompose → 降级按 Event Type 分析（--group-by event_type）；"
          "无调用栈 → 只能定位分配行为无法给代码位置（见 SKILL B1）")


def cmd_window(path, start, end, pool=None):
    """时间窗切片（流式单遍，内存 O(活跃块)）：起点/终点活跃块、窗口内申请/释放/净变化、TOP 归口。

    start/end 为纳秒时间戳（含端点）。要求 dump 按时间戳有序（未排序时窗口边界判定不可靠，
    建议先 --check/--sort）。--pool 指定维度（默认全池混聚，HAL 与池事件嵌套不可相加）。
    """
    active = {}                       # ptr -> (size, owner, etype)
    in_window = False
    start_total = start_count = 0
    start_by_owner = None
    alloc_w = free_w = shadow_w = 0
    alloc_size = free_size = shadow_size = 0
    alloc_by = defaultdict(int)
    peak_used = None
    prev_ts = None
    for e in iter_events(path, KEYS_OWNER + ("used",)):   # used 供窗口内峰值分支；attr 只多提取一个键
        event = e.get(COL_EVENT)
        if event not in (EVENT_MALLOC, EVENT_FREE):
            continue
        ts = int_or_none(e.get(COL_TIMESTAMP))
        if ts is None:
            continue
        if prev_ts is not None and ts < prev_ts:
            print("警告: 文件时间戳未排序，窗口切片边界判定不可靠，建议先执行 --sort [--overwrite]",
                  file=sys.stderr)
        prev_ts = ts
        if ts > end:
            break
        etype = e.get(COL_EVENT_TYPE) or "UNKNOWN"
        if pool and etype != pool:
            continue
        attr = e.get(COL_ATTR, {})
        if ts >= start and not in_window:
            in_window = True
            start_active = [b for b in active.values()]
            start_total = sum(b[0] for b in start_active)
            start_count = len(start_active)
            start_by_owner = defaultdict(int)
            for b in start_active:
                key = owner_group_key(b[1]) if pool else b[1]
                start_by_owner[key] += b[0]
        if in_window:
            used = int_or_none(attr.get("used"))
            if used is not None and used >= 0 and (peak_used is None or used > peak_used[0]):
                peak_used = (used, ts)
        ptr = e.get(COL_PTR)
        if event == EVENT_MALLOC:
            size = int_or_none(attr.get("size")) or 0
            owner = attr.get("owner") or etype or "UNKNOWN"
            if ptr in active:
                del active[ptr]
            active[ptr] = (size, owner, etype)
            if in_window:
                alloc_w += 1
                alloc_size += size
                key = owner_group_key(owner) if pool else owner
                alloc_by[key] += size
        else:
            blk = active.pop(ptr, None)
            if blk and in_window:
                if attr.get("shadow") == "true":
                    shadow_w += 1
                    shadow_size += blk[0]
                else:
                    free_w += 1
                    free_size += blk[0]
    if start_by_owner is None:
        print(f"[FAIL] 无事件落在时间窗 [{start}, {end}] ns（起点晚于全部事件或早于数据时序）")
        return
    end_by = defaultdict(int)
    for b in active.values():
        key = owner_group_key(b[1]) if pool else b[1]
        end_by[key] += b[0]
    end_total = sum(end_by.values())
    end_count = len(active)
    dim = f"{pool} 池维度" if pool else "全池（HAL 与池事件嵌套不可相加，仅作数量参考）"
    print(f"[时间窗切片] [{start}, {end}] ns（{dim}，含端点）:")
    print(f"  窗口起点活跃: {format_bytes(start_total)}（{start_count} 块）")
    print(f"  窗口内: 申请 {alloc_w} 块 / {format_bytes(alloc_size)}；真实释放 {free_w} 块 / "
          f"{format_bytes(free_size)}；shadow 释放 {shadow_w} 块 / {format_bytes(shadow_size)}")
    print(f"  窗口终点活跃: {format_bytes(end_total)}（{end_count} 块）→ "
          f"净变化 {format_bytes(end_total - start_total)}（{end_count - start_count} 块）")
    if peak_used is not None:
        print(f"  窗口内 used 峰值: {format_bytes(peak_used[0])} @ {peak_used[1]} ns")
    label = "一级归口" if pool else "owner"
    if alloc_by:
        print(f"\n窗口内申请量 TOP（按{label}）:")
        rows = [(k, v, format_bytes(v), f"{v / alloc_size * 100:.1f}%")
                for k, v in sorted(alloc_by.items(), key=lambda kv: -kv[1])[:10]]
        print_table(rows, [label, "申请量(字节)", "申请量", "占窗口申请比"])
    print(f"\n窗口终点活跃 TOP（按{label}）:")
    rows = [(k, v, format_bytes(v), f"{v / end_total * 100:.1f}%")
            for k, v in sorted(end_by.items(), key=lambda kv: -kv[1])[:10]]
    print_table(rows, [label, "占用(字节)", "占用", "占比"])
    if start_by_owner:
        print(f"\n窗口起点活跃 TOP（按{label}）:")
        rows = [(k, v, format_bytes(v)) for k, v in
                sorted(start_by_owner.items(), key=lambda kv: -kv[1])[:10]]
        print_table(rows, [label, "占用(字节)", "占用"])


def main():
    # 输出稳定为 UTF-8：Windows 控制台/管道默认按 locale（GBK）编码，重定向时可能抛
    # UnicodeEncodeError；Linux 下无影响。reconfigure 需 Python 3.7+，旧版本静默跳过。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(
        description="memscope_dump csv 聚合统计与本地可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python3 aggregate_dump.py dump.csv --check\n"
               "  python3 aggregate_dump.py dump.csv --sort --overwrite\n"
               "  python3 aggregate_dump.py dump.csv --group-by owner --metric peak\n"
               "  python3 aggregate_dump.py dump.csv --at-timestamp 1786526882887795927 --ratio\n"
               "  python3 aggregate_dump.py dump.csv --curve --ascii\n"
               "  python3 aggregate_dump.py dump.csv --peak --key used\n"
               "  python3 aggregate_dump.py dump.csv --oom\n"
               "  python3 aggregate_dump.py dump.csv --fragmentation --pool PTA\n"
               "  python3 aggregate_dump.py dump.csv --hal-segments --min-size 100MB\n"
               "  python3 aggregate_dump.py dump.csv --trend --pool PTA\n"
               "  python3 aggregate_dump.py dump.csv --stats\n"
               "  python3 aggregate_dump.py dump.csv --window 1786526882000000000 1786526883000000000",
    )
    parser.add_argument("dump", help="memscope_dump_*.csv 文件路径")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true",
                       help="完整性校验（表头/列数/最后一行/时间戳是否已有序）")
    group.add_argument("--sort", action="store_true",
                       help="按时间戳升序排序；已有序时跳过（幂等）")
    group.add_argument("--group-by", choices=["owner", "event_type"],
                       help="按 owner（模块/组件）或 event_type（HAL/PTA/...）聚合")
    group.add_argument("--at-timestamp", type=int,
                       help="统计指定时间点（ns）的活跃块分布（默认拆 HAL 驱动层全集：一级归口表 + "
                            "层级分配树，--pool <池> 下钻池内；要求有序 dump）")
    group.add_argument("--leak-candidates", action="store_true",
                       help="扫描无真实释放的内存块（泄漏候选：FREE 全部为 shadow:true），按 owner/调用栈聚合")
    group.add_argument("--curve", action="store_true", help="输出显存曲线数据点")
    group.add_argument("--peak", action="store_true",
                       help="定位曲线峰值点（--key 选择曲线键，默认 used；输出峰值时间戳，可直接衔接 --at-timestamp）")
    group.add_argument("--oom", action="store_true",
                       help="汇总 OOM_DETAIL 诊断（TRIGGER/TOP_ALLOC/RECENT_ALLOC 分表 + 量化推断提示；--detail 控制条数，0=不输出明细）")
    group.add_argument("--fragmentation", action="store_true",
                       help="池碎片率与扩容画像（扩容清单/碎片率统计/有效使用率最低 TOP；--pool 指定池，默认全部池事件）")
    group.add_argument("--hal-segments", action="store_true",
                       help="HAL 池大段清单（--min-size 过滤，默认 100MB，段数过少可调小；含 alloc_type/page_type，expandable 生效判定）")
    group.add_argument("--trend", action="store_true",
                       help="跨周期趋势（等宽桶 used/total 末值 + 算法 A/B 判定；默认拆 HAL + PTA 双视图（PTA 存在时，HAL 段级仅参考/PTA 块级=算法 A），--pool 指定单一池）")
    group.add_argument("--stats", action="store_true",
                       help="数据画像：事件分布/时间范围/解析能力提示（分析前体检）")
    group.add_argument("--window", nargs=2, type=int, metavar=("START", "END"),
                       help="时间窗切片（纳秒，含端点）：起点/终点活跃块、窗口内申请/释放/净变化、TOP 归口")
    parser.add_argument("--overwrite", action="store_true",
                        help="--sort 时将排序结果覆写回原文件（默认仅输出到标准输出）")
    parser.add_argument("--metric", choices=["peak", "total"], default="peak",
                        help="聚合口径：peak 活跃峰值（默认）/ total 累计申请量")
    parser.add_argument("--ratio", action="store_true",
                        help="（兼容保留：占比已为 --at-timestamp 默认输出，本参数不改变行为）")
    parser.add_argument("--ascii", action="store_true", help="--curve 时输出 ASCII 文本曲线")
    parser.add_argument("--bins", type=positive_int, default=40, help="--ascii 分桶数（默认 40，须 >=1）")
    parser.add_argument("--limit", type=int, default=0,
                        help="--curve 数据点输出上限（0=全部；ASCII 模式默认不输出数据点表，加 --limit N 打印前 N 条）")
    parser.add_argument("--pool", choices=list(POOLS),
                        help="指定拆解维度（内存池=Event Type）。缺省时 --at-timestamp 默认拆 HAL"
                             "（驱动层全集，含框架池段）；HAL 事件与池事件是同一物理内存的嵌套"
                             "视图，数值不可相加；HOST 暂不分析")
    parser.add_argument("--detail", type=int, default=10,
                        help="--leak-candidates 候选块明细条数 / --oom 各表条数（默认 10；0=不输出明细）")
    parser.add_argument("--key", choices=["used", "device_used", "process_used"], default="used",
                        help="--peak 统计的曲线键（默认 used）")
    parser.add_argument("--min-size", default="100MB",
                        help="--hal-segments 的最小段大小（支持裸字节或 KB/MB/GB 后缀，默认 100MB；段数过少可调小）")
    parser.add_argument("--buckets", type=positive_int, default=20,
                        help="--trend 等宽时间桶数（默认 20）")
    args = parser.parse_args()

    # --check 流式校验（O(1) 内存，大数据量可用）；--sort 需要全量行列表
    # （排序本质，内存约 O(文件行数)；msmemscope 产物通常已有序，--check 通过即可跳过 --sort）。
    if args.check:
        try:
            if not cmd_check(args.dump):
                sys.exit(1)
        except OSError as exc:
            print(f"读取文件失败: {exc}", file=sys.stderr)
            sys.exit(1)
    elif args.sort:
        try:
            fieldnames, rows = read_raw_rows(args.dump)
        except OSError as exc:
            print(f"读取文件失败: {exc}", file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(f"解析失败: {exc}", file=sys.stderr)
            sys.exit(1)
        cmd_sort(args.dump, fieldnames, rows, args.overwrite)
    else:
        try:
            if args.group_by:
                cmd_group_by(args.dump, args.group_by, args.metric, args.pool)
            elif args.at_timestamp is not None:
                cmd_at_timestamp(args.dump, args.at_timestamp, args.pool)
            elif args.leak_candidates:
                cmd_leak_candidates(args.dump, args.detail, args.pool)
            elif args.curve:
                cmd_curve(args.dump, args.ascii, args.bins, args.limit)
            elif args.peak:
                cmd_peak(args.dump, args.key, args.pool)
            elif args.oom:
                cmd_oom(args.dump, args.detail)
            elif args.fragmentation:
                cmd_fragmentation(args.dump, args.pool)
            elif args.hal_segments:
                cmd_hal_segments(args.dump, parse_size(args.min_size))
            elif args.trend:
                cmd_trend(args.dump, args.buckets, args.pool)
            elif args.stats:
                cmd_stats(args.dump)
            elif args.window:
                cmd_window(args.dump, args.window[0], args.window[1], args.pool)
        except OSError as exc:
            print(f"读取文件失败: {exc}", file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(f"解析失败: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()