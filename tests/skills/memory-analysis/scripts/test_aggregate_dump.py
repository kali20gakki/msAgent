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
"""
aggregate_dump.py 单元测试（2026-08-19：检视缺陷修复的回归）

选择性覆盖两个函数：
- iter_events   —— with_stack 携带 Call Stack(Python)（P1）+ 缺列安全（P3-2）
- _trend_one    —— total 数据点早于 used 首点时桶索引下界截断（P3-1）
"""

import csv
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "skills",
        "profiler",
        "memory-analysis",
        "scripts",
    ),
)

import aggregate_dump as agg


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


FULL_HEADER = [
    "ID",
    "Event",
    "Event Type",
    "Name",
    "Timestamp(ns)",
    "Process Id",
    "Thread Id",
    "Device Id",
    "Ptr",
    "Attr",
    "Call Stack(Python)",
    "Call Stack(C)",
]


class TestIterEvents(unittest.TestCase):
    """P1：with_stack 应携带调用栈列；P3-2：缺列不抛 KeyError。"""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ut_iter_")

    def test_with_stack_carries_stack_column(self):
        path = os.path.join(self.dir, "full.csv")
        write_csv(
            path,
            FULL_HEADER,
            [
                [1, "MALLOC", "HAL", "", 100, 1, 1, 0, "0x1", "{size:64,owner:PTA@weight}", "Module.load:<module>", ""],
            ],
        )
        # with_stack=True 时携带该列
        ev = next(iter(agg.iter_events(path, ("size",), with_stack=True)))
        self.assertEqual(ev[agg.COL_STACK_PY], "Module.load:<module>")
        # with_stack=False 时键不存在（不读大字段）
        ev2 = next(iter(agg.iter_events(path, ("size",))))
        self.assertNotIn(agg.COL_STACK_PY, ev2)

    def test_missing_columns_safe(self):
        # 精简表头：无 Event Type/Ptr/Call Stack(Python)/Call Stack(C)
        path = os.path.join(self.dir, "min.csv")
        write_csv(
            path,
            ["ID", "Event", "Timestamp(ns)", "Attr"],
            [
                [1, "MALLOC", 100, "{size:1024}"],
            ],
        )
        ev = next(iter(agg.iter_events(path, ("size",), with_stack=True)))
        # 键保持存在、值为安全默认（不抛 KeyError）
        self.assertEqual(ev[agg.COL_EVENT], "MALLOC")
        self.assertEqual(ev[agg.COL_ATTR], {"size": "1024"})
        self.assertNotIn(agg.COL_STACK_PY, ev)  # 缺列时不携带
        self.assertEqual(next(iter(agg.iter_events(path, ("size",))))[agg.COL_EVENT], "MALLOC")

    def test_extract_attr_selective_keys(self):
        body = "{size:64,owner:PTA@weight,used:32,shadow:true}"
        got = agg.extract_attr(body, ("size", "owner"))
        self.assertEqual(got, {"size": "64", "owner": "PTA@weight"})
        # 键为首项（无前导逗号）时也能命中
        self.assertEqual(agg.extract_attr("size:64,owner:x", ("size",)), {"size": "64"})


class TestTrendBucketClamp(unittest.TestCase):
    """P3-1：total 数据点早于 used 首点（used=-1 被滤而 total 保留）时，
    桶索引缺下界会回绕写 b_tot[-1]；修复后应归入桶 0。
    """

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ut_trend_")

    def test_total_earlier_than_used_no_wrap(self):
        path = os.path.join(self.dir, "trend.csv")
        write_csv(
            path,
            FULL_HEADER,
            [
                # t=1000: used=-1 被过滤（仅 total 保留）→ total 首点早于 used 首点
                [1, "MALLOC", "HAL", "", 1000, 1, 1, 0, "0xa", "{size:1048576,used:-1,total:999}"],
                [2, "MALLOC", "HAL", "", 2000, 1, 1, 0, "0xb", "{size:1048576,used:500,total:999}"],
                [3, "FREE", "HAL", "", 3000, 1, 1, 0, "0xa", "{used:500,total:999}"],
            ],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            agg._trend_one(path, buckets=4, pool="HAL", dual=False)
        out = buf.getvalue()
        # 桶 0 应有 total 999B（修复前回绕：桶 0 为 "-"、值被写进末桶）
        bucket0 = next(line for line in out.splitlines() if line.startswith("0 |"))
        self.assertIn("999B", bucket0, out)
        self.assertIn("[HAL 池跨桶趋势]", out)

    def test_rejects_nodata(self):
        path = os.path.join(self.dir, "nodata.csv")
        write_csv(
            path,
            FULL_HEADER,
            [
                [1, "MALLOC", "OTHER", "", 1000, 1, 1, 0, "0x1", "{size:64}"],
            ],
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            agg._trend_one(path, buckets=4, pool="HAL", dual=False)
        self.assertIn("[FAIL] 无 HAL 池有效曲线数据点", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
