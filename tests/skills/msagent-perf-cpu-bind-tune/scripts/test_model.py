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

"""model.py：两种求解模式的 node 归属与全局约束。"""

import unittest

import testutils
import model


class TestTopoAffinity(unittest.TestCase):
    def setUp(self):
        self.topo = testutils.a2_topo()
        all_cpus = list(range(192))
        self.units = [testutils.make_unit(i, i, all_cpus, affinity="144-165" if i < 2 else "0-23") for i in range(8)]

    def test_target_node_follows_npu(self):
        modeled = model.build(self.topo, self.units)
        self.assertEqual(modeled["locality_mode"], "topo_affinity")
        by_npu = {n["id"]: n["numa_node"] for n in self.topo["npu"]}
        for unit in modeled["units"]:
            self.assertEqual(unit["target_node"], by_npu[unit["primary_npu"]])
            self.assertEqual(unit["target_node_source"], "topo_affinity")

    def test_cross_node_majority_and_deviation(self):
        unit = testutils.make_unit(0, 6, list(range(192)))
        unit["npus_significant"] = [6, 6, 4]  # node2 ×2 vs node4 ×1
        unit["npus"] = [6, 4]
        model.build(self.topo, [unit])
        self.assertEqual(unit["target_node"], 2)
        self.assertEqual(unit["deviations"][0]["kind"], "multi_npu_cross_node")

    def test_tie_break_prefers_lower_node(self):
        unit = testutils.make_unit(0, 6, list(range(192)))
        unit["npus_significant"] = [6, 4]  # NPU6→node2 与 NPU4→node0 各 1
        unit["npus"] = [6, 4]
        model.build(self.topo, [unit])
        self.assertEqual(unit["target_node"], 0)


class TestUnknownNodeFindings(unittest.TestCase):
    def test_none_primary_npu_does_not_crash(self):
        findings = model._unknown_node_findings([{"unit_id": 0, "target_node": None, "primary_npu": None}])
        self.assertIn("NPU-", findings[0]["evidence"])


class TestGlobalSlice(unittest.TestCase):
    def _build(self, npus=16):
        pairs = [[i, i + 1] for i in range(0, npus, 2)]
        topo = testutils.a3_topo(nodes=8, cpus_per_node=80, npus=npus, die_pairs=pairs)
        all_cpus = [c for n in topo["nodes"] for c in n["cpus"]]
        units = [testutils.make_unit(i, i, all_cpus) for i in range(npus)]
        return topo, units

    def test_die_pair_same_node(self):
        topo, units = self._build()
        modeled = model.build(topo, units)
        self.assertEqual(modeled["locality_mode"], "global_slice")
        self.assertIn("die_pair_same_node", [f["kind"] for f in modeled["findings"]])
        by_package = {}
        for unit in units:
            by_package.setdefault(unit["primary_npu"] // 2, []).append(unit)
        for members in by_package.values():
            self.assertEqual(len({u["target_node"] for u in members}), 1)
            self.assertEqual(members[0]["package_shared_with"], [members[1]["unit_id"]])
        self.assertEqual(sorted({u["target_node"] for u in units}), list(range(8)))

    def test_single_die_degrades_to_per_unit(self):
        topo = testutils.a3_topo(nodes=8, cpus_per_node=40, npus=8)  # 无 die_groups
        all_cpus = [c for n in topo["nodes"] for c in n["cpus"]]
        units = [testutils.make_unit(i, i, all_cpus) for i in range(8)]
        modeled = model.build(topo, units)
        self.assertEqual(sorted(u["target_node"] for u in units), list(range(8)))
        self.assertTrue(all(u["target_node_source"] == "global_slice" for u in units))
        # 缺 die 信息时按 NPU 逐个分组，必须显式标注，不能静默当真实封装关系
        self.assertIn("die_info_unavailable", [f["kind"] for f in modeled["findings"]])

    def test_die_groups_present_has_no_missing_die_finding(self):
        topo, units = self._build()
        modeled = model.build(topo, units)
        self.assertNotIn("die_info_unavailable", [f["kind"] for f in modeled["findings"]])

    def test_legacy_locality_mode_alias(self):
        """legacy 别名统一由 topology.LOCALITY_ALIAS 映射。"""
        topo = testutils.a2_topo()
        topo["meta"]["locality_mode"] = "npu_local"
        modeled = model.build(topo, [testutils.make_unit(0, 0, list(range(192)))])
        self.assertEqual(modeled["locality_mode"], "topo_affinity")

    def test_no_nodes_falls_back_to_none(self):
        topo = testutils.a3_topo(nodes=0, cpus_per_node=0, npus=2)
        units = [testutils.make_unit(0, 0, list(range(8)))]
        model.build(topo, units)
        self.assertIsNone(units[0]["target_node"])
        self.assertEqual(units[0]["target_cpus"], [])


if __name__ == "__main__":
    unittest.main()
