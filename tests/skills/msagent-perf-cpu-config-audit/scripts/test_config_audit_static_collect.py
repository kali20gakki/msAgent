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
static_collect.py 单元测试：NPU 事实采集只执行一次 npu-smi info。
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "skills",
        "host-cpu",
        "msagent-perf-cpu-config-audit",
        "scripts",
    ),
)

import static_collect


class TestNpuFactsSingleRun(unittest.TestCase):
    def test_snapshot_used_once_for_overview_and_processes(self):
        calls = []

        def fake_snapshot():
            calls.append(1)
            return ({"available": True, "npus": [{"npu": 0, "name": "910B2"}]}, [{"pid": 1}, {"pid": 2}])

        facts = []
        with (
            patch.object(static_collect.npu_query, "query_snapshot", fake_snapshot),
            patch.object(static_collect.npu_query, "query_info_l", lambda: None),
            patch.object(static_collect.npu_query, "query_die_map", lambda: None),
            patch.object(static_collect.npu_query, "query_topo", lambda: {"form": "npu", "fabric": "pcie", "rows": 1}),
            patch.object(static_collect, "_count_davinci_devices", lambda: None),
        ):
            static_collect._collect_npu_facts(facts)

        self.assertEqual(len(calls), 1)
        values = {item["item"]: item["value"] for item in facts}
        self.assertTrue(values["npu_available"])
        self.assertEqual(values["npu_count"], 1)
        self.assertEqual(values["npu_process_count"], 2)

    def test_unavailable_skips_package_level_queries(self):
        facts = []
        with (
            patch.object(static_collect.npu_query, "query_snapshot", lambda: ({"available": False}, [])),
            patch.object(
                static_collect.npu_query, "query_info_l", side_effect=AssertionError("npu-smi 不可用时不应再查询")
            ),
            patch.object(
                static_collect.npu_query, "query_topo", lambda: {"form": "unknown", "fabric": "unknown", "rows": 0}
            ),
        ):
            static_collect._collect_npu_facts(facts)

        values = {item["item"]: item["value"] for item in facts}
        self.assertFalse(values["npu_available"])
        self.assertEqual(values["npu_process_count"], 0)


if __name__ == "__main__":
    unittest.main()
