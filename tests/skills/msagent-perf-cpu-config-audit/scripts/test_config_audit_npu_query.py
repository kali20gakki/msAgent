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
npu_query.py 单元测试：npu-smi 输出解析（-l / -m / -t topo）与 fabric 判定。
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

import npu_query

INFO_L = """
        NPU ID                         : 0
        Product Name                   : Ascend910
        Chip Count                     : 2
        NPU ID                         : 1
        Product Name                   : Ascend910
        Chip Count                     : 2
        Total Count                    : 2
"""

INFO_M_5COL = """
NPU ID    Chip ID    Chip Logic ID    Chip Phy-ID    Chip Name
0         0          0                0              Ascend 910C
0         1          1                1              Ascend 910C
0         2          -                -              Mcu
1         0          2                2              Ascend 910C
1         1          3                3              Ascend 910C
"""

INFO_M_4COL = """
NPU ID    Chip ID    Chip Logic ID    Chip Name
0         0          0                Ascend 910B2
0         1          -                Mcu
1         0          1                Ascend 910B2
"""

TOPO_DIE = """
             Phy-ID0  Phy-ID1  Phy-ID2  Phy-ID3
Phy-ID0      X        SIO      HCCS_SW  HCCS_SW
Phy-ID1      SIO      X        HCCS_SW  HCCS_SW
Phy-ID2      HCCS_SW  HCCS_SW  X        SIO
Phy-ID3      HCCS_SW  HCCS_SW  SIO      X
"""

TOPO_NPU = """
           NPU0       NPU1       CPU Affinity
NPU0       X          HCCS       144-167
NPU1       HCCS       X          168-191

Legend:

  X    = Self
  SYS  = Path traversing PCIe and NUMA nodes.
  PHB  = Path traversing PCIe and the PCIe host bridge of a CPU.
"""

TOPO_DIE_GAP = """
             Phy-ID0  Phy-ID1  Phy-ID2  Phy-ID3
Phy-ID0      X        -        SIO      HCCS_SW
Phy-ID1      -        X        HCCS_SW  HCCS_SW
Phy-ID2      HCCS_SW  HCCS_SW  X        HCCS_SW
Phy-ID3      HCCS_SW  HCCS_SW  HCCS_SW  X
"""


class TestParseInfoL(unittest.TestCase):
    def test_total_and_chips(self):
        data = npu_query.parse_info_l(INFO_L)
        self.assertEqual(data["total_count"], 2)
        self.assertEqual(len(data["npus"]), 2)
        self.assertEqual(data["npus"][0]["npu"], 0)
        self.assertEqual(data["npus"][0]["chips"], 2)
        self.assertEqual(data["npus"][0]["name"], "Ascend910")


class TestParseInfoM(unittest.TestCase):
    def test_5col_double_die(self):
        data = npu_query.parse_info_m(INFO_M_5COL)
        self.assertEqual(data["logic_ids"], [0, 1, 2, 3])
        self.assertEqual(data["groups"], [[0, 1], [2, 3]])
        self.assertEqual(data["dies"][0]["phy_id"], 0)

    def test_5col_filters_mcu(self):
        data = npu_query.parse_info_m(INFO_M_5COL)
        self.assertNotIn("Mcu", [d["name"] for d in data["dies"]])

    def test_4col_no_phy_id(self):
        data = npu_query.parse_info_m(INFO_M_4COL)
        self.assertEqual(data["logic_ids"], [0, 1])
        self.assertEqual(data["groups"], [[0], [1]])
        # 无 Phy-ID 列时退回 Logic ID，名称含空格也能完整取到
        self.assertEqual(data["dies"][0]["phy_id"], 0)
        self.assertEqual(data["dies"][0]["name"], "Ascend 910B2")


class TestParseProcesses(unittest.TestCase):
    def test_merged_head_cell(self):
        text = """
| NPU ID | Chip ID | Process id | Process name | Process memory(MB) | Process id in container |
| 0       0 | 293350 | VLLMWorker | 117 | NA |
"""
        procs = npu_query.parse_processes(text)
        self.assertEqual(procs, [{"npu": 0, "pid": 293350, "name": "VLLMWorker", "mem_mb": 117}])

    def test_no_process_section(self):
        self.assertEqual(npu_query.parse_processes("no table here"), [])


class TestIsRelation(unittest.TestCase):
    def test_header_tokens_not_relation(self):
        self.assertFalse(npu_query._is_relation("NPU0"))
        self.assertFalse(npu_query._is_relation("Phy-ID1"))
        self.assertFalse(npu_query._is_relation("CPU"))
        self.assertFalse(npu_query._is_relation("Affinity"))
        self.assertFalse(npu_query._is_relation("X"))

    def test_relation_tokens(self):
        for token in ("SIO", "HCCS", "HCCS_SW", "UB", "PHB", "PIX", "PXB", "SYS"):
            self.assertTrue(npu_query._is_relation(token), token)


class TestFabric(unittest.TestCase):
    def test_enum(self):
        self.assertEqual(npu_query._fabric_from_relations({"UB"}), "ub")
        self.assertEqual(npu_query._fabric_from_relations({"HCCS_SW", "SIO"}), "hccs_sw")
        self.assertEqual(npu_query._fabric_from_relations({"HCCS"}), "hccs")
        self.assertEqual(npu_query._fabric_from_relations({"PHB"}), "pcie")
        self.assertEqual(npu_query._fabric_from_relations({"SYS"}), "pcie")
        self.assertEqual(npu_query._fabric_from_relations(set()), "unknown")


class TestParseTopo(unittest.TestCase):
    def test_die_form_phy_id_label(self):
        data = npu_query.parse_topo(TOPO_DIE)
        self.assertEqual(data["form"], "die")
        self.assertEqual(data["rows"], 4)
        self.assertEqual(data["fabric"], "hccs_sw")
        self.assertEqual(data["sio_pairs"], [[0, 1], [2, 3]])

    def test_npu_form_header_skipped(self):
        data = npu_query.parse_topo(TOPO_NPU)
        self.assertEqual(data["form"], "npu")
        # 表头以 NPU0 开头但无关系符号，不得计入行数
        self.assertEqual(data["rows"], 2)
        self.assertEqual(data["fabric"], "hccs")
        self.assertIn({"npu": 0, "cpus": "144-167"}, data["affinity"])

    def test_legend_not_relation_source(self):
        data = npu_query.parse_topo(TOPO_NPU)
        # Legend 段的 SYS/PHB 不得把 HCCS 直连误判为 pcie
        self.assertNotEqual(data["fabric"], "pcie")
        self.assertNotIn("SYS", data["relations"])

    def test_empty(self):
        data = npu_query.parse_topo("")
        self.assertEqual(data["form"], "unknown")
        self.assertEqual(data["rows"], 0)
        self.assertEqual(data["fabric"], "unknown")

    def test_sio_pair_uses_column_not_filtered_index(self):
        """格内 "-" 被过滤后，SIO 仍按原始列号（Phy-ID）配对。"""
        data = npu_query.parse_topo(TOPO_DIE_GAP)
        self.assertEqual(data["form"], "die")
        self.assertEqual(data["sio_pairs"], [[0, 2]])


class TestQuerySnapshot(unittest.TestCase):
    """query_snapshot：一次 npu-smi info 同时喂给总览与进程解析。"""

    def test_single_run_feeds_both_parsers(self):
        calls = []

        def fake_run(args):
            calls.append(args)
            return "RAW"

        with (
            patch.object(npu_query, "_run", fake_run),
            patch.object(npu_query, "parse_info", lambda text: {"npus": [{"npu": 0}], "parse_ok": True}),
            patch.object(npu_query, "parse_processes", lambda text: [{"pid": 1}]),
        ):
            info, procs = npu_query.query_snapshot()
        self.assertEqual(calls, [["info"]])
        self.assertTrue(info["available"])
        self.assertEqual(info["npus"], [{"npu": 0}])
        self.assertEqual(procs, [{"pid": 1}])

    def test_unavailable_returns_empty(self):
        with patch.object(npu_query, "_run", lambda args: None):
            self.assertEqual(npu_query.query_snapshot(), ({"available": False}, []))


if __name__ == "__main__":
    unittest.main()
