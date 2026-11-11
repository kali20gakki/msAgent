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
# MERCHANTABILITY OR FIT FOR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""npu_query.py：npu-smi 输出解析（info / -l / -m / -t topo）与 fabric 判定。"""

import unittest

import testutils  # noqa: F401 - 负责把 skill scripts 加入 sys.path
import npu_query

PROC_6COL = """| NPU   Name           | Health | Power(W) Temp(C) Hugepages-Usage(page) |
| 0     910B2          | OK     | 100.0    45      0    / 0             |
| 0     | 0000:9D:00.0 | 0.0  0  / 0  0  / 0 |
+----------------------+--------+----------------------------------------+
| NPU     Chip         | Process id | Process name | Process memory(MB) | Process id in container |
| 0       0            | 293354     | VLLMWorker   | 117                | NA                      |
"""

PROC_5COL = """| NPU     Chip         | Process id | Process name | Process memory(MB) |
| 0       0            | 380691     | vec_add      | 117                |
"""

INFO_L = """
        NPU ID                         : 0
        Product Name                   : 910C
        Chip Count                     : 2

        NPU ID                         : 1
        Product Name                   : 910C
        Chip Count                     : 2

        Total Count                    : 8
"""

INFO_M_5COL = """
NPU ID    Chip ID    Chip Logic ID    Chip Phy-ID    Chip Name
0         0          0                0              Ascend910
0         1          1                1              Ascend910
0         2          -                -              Mcu
1         0          2                2              Ascend910
1         1          3                3              Ascend910
1         2          -                -              Mcu
"""

INFO_M_4COL = """
NPU ID                 Chip ID                Chip Logic ID          Chip Name
0                      0                      0                      Ascend 910B2
0                      1                      -                      Mcu
1                      0                      1                      Ascend 910B2
"""

PHY_TOPO = """           Phy-ID0    Phy-ID1    Phy-ID2    Phy-ID3
Phy-ID0    X          SIO        HCCS_SW    HCCS_SW
Phy-ID1    SIO        X          HCCS_SW    HCCS_SW
Phy-ID2    HCCS_SW    HCCS_SW    X          SIO
Phy-ID3    HCCS_SW    HCCS_SW    SIO        X
"""

AFFINITY_TOPO = """           NPU0       NPU1       CPU Affinity
NPU0       X          SYS        144-167
NPU1       SYS        X          96-119
"""

HCCS_WITH_LEGEND = """           NPU0       NPU1       CPU Affinity
NPU0       X          HCCS       144-167
NPU1       HCCS       X          144-167

Legend:

  X    = Self
  SYS  = Path traversing PCIe and NUMA nodes.
  PHB  = Path traversing PCIe and the PCIe host bridge of a CPU.
"""


class TestParseProcesses(unittest.TestCase):
    def test_six_columns_with_container(self):
        info = npu_query.parse_info(PROC_6COL)
        self.assertEqual(len(info["processes"]), 1)
        proc = info["processes"][0]
        self.assertEqual((proc["npu"], proc["chip"], proc["pid"]), (0, 0, 293354))
        self.assertEqual(proc["name"], "VLLMWorker")
        self.assertEqual(proc["mem_mb"], 117)
        # npu-smi 用字面量 NA 表示不在容器内，需归一成 None
        self.assertIsNone(proc["container_id"])

    def test_five_columns_without_container(self):
        """旧版 npu-smi 无 container 列，进程不能被整行丢弃。"""
        info = npu_query.parse_info(PROC_5COL)
        self.assertEqual(len(info["processes"]), 1)
        self.assertEqual(info["processes"][0]["pid"], 380691)
        self.assertIsNone(info["processes"][0]["container_id"])

    def test_a2_fixture_keeps_all_processes(self):
        info, _ = testutils.load_a2_npu_raw()
        self.assertEqual(len(info["devices"]), 8)
        self.assertEqual(len(info["processes"]), 5)


class TestPrimaryProcesses(unittest.TestCase):
    def test_primary_npu_by_memory(self):
        """同一 pid 占多卡时，取显存最大的卡作为 primary_npu。"""
        procs = [
            {"pid": 1, "npu": 0, "name": "vllm", "mem_mb": 48 * 1024},
            {"pid": 1, "npu": 1, "name": "vllm", "mem_mb": 117},
            {"pid": 2, "npu": 2, "name": "vllm", "mem_mb": 10},
        ]
        primaries = npu_query.primary_processes(procs)
        by_pid = {p["pid"]: p["primary_npu"] for p in primaries}
        self.assertEqual(by_pid, {1: 0, 2: 2})

    def test_a2_fixture_primaries(self):
        info, _ = testutils.load_a2_npu_raw()
        primaries = npu_query.primary_processes(info["processes"])
        self.assertEqual([(p["pid"], p["primary_npu"]) for p in primaries], [(293354, 0), (293350, 1), (293351, 6)])


class TestUnknownHbm(unittest.TestCase):
    def test_occupancy_without_process(self):
        """有 HBM 占用但无可见进程的卡，由唯一共享实现识别。"""
        info = {
            "devices": [{"id": 0, "hbm_used_mb": 3584}, {"id": 1, "hbm_used_mb": 3584}, {"id": 2, "hbm_used_mb": 0}],
            "processes": [{"pid": 1, "npu": 0, "mem_mb": 100}],
        }
        self.assertEqual(npu_query.unknown_hbm_npus(info), [{"npu": 1, "hbm_used_mb": 3584}])


class TestInfoLM(unittest.TestCase):
    def test_info_l_package_counts(self):
        parsed = npu_query.parse_info_l(INFO_L)
        self.assertEqual(parsed["total_count"], 8)
        self.assertEqual(len(parsed["npus"]), 2)
        self.assertEqual(parsed["npus"][0]["chips"], 2)

    def test_info_m_five_columns_filters_mcu(self):
        parsed = npu_query.parse_info_m(INFO_M_5COL)
        self.assertEqual(len(parsed["dies"]), 4)
        self.assertEqual(parsed["groups"], [[0, 1], [2, 3]])
        self.assertEqual(parsed["logic_ids"], [0, 1, 2, 3])

    def test_info_m_four_columns(self):
        """25.x 无 Chip Phy-ID 列且 Chip Name 含空格，仍需按列自适应解析。"""
        parsed = npu_query.parse_info_m(INFO_M_4COL)
        self.assertEqual(len(parsed["dies"]), 2)
        self.assertEqual(parsed["groups"], [[0], [1]])
        self.assertEqual(parsed["dies"][0]["name"], "Ascend 910B2")


class TestTopo(unittest.TestCase):
    def test_phy_matrix_is_supernode_without_affinity(self):
        parsed = npu_query.parse_topo(PHY_TOPO)
        self.assertEqual(sorted(parsed), [0, 1, 2, 3])
        self.assertTrue(all(not v["affinity_cpus"] for v in parsed.values()))
        meta = npu_query.topo_meta(PHY_TOPO)
        self.assertEqual(meta["format"], "phy_matrix")
        self.assertFalse(meta["has_affinity"])
        self.assertEqual(meta["fabric_hint"], "hccs_supernode")

    def test_affinity_matrix_is_pcie(self):
        parsed = npu_query.parse_topo(AFFINITY_TOPO)
        self.assertEqual(parsed[0]["affinity_cpus"], list(range(144, 168)))
        self.assertEqual(parsed[1]["affinity"], "96-119")
        meta = npu_query.topo_meta(AFFINITY_TOPO)
        self.assertEqual(meta["format"], "affinity")
        self.assertTrue(meta["has_affinity"])
        self.assertEqual(meta["fabric_hint"], "pcie")

    def test_hccs_not_polluted_by_legend(self):
        """Legend 段枚举的 SYS/PHB 不能当作关系符号（A2 真机教训）。"""
        meta = npu_query.topo_meta(HCCS_WITH_LEGEND)
        self.assertEqual(meta["fabric_hint"], "hccs")
        self.assertEqual(npu_query.parse_topo(HCCS_WITH_LEGEND)[0]["affinity_cpus"], list(range(144, 168)))


if __name__ == "__main__":
    unittest.main()
