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

"""feasibility.py：C1-C5 规划可行性检查。"""

import unittest

import testutils  # noqa: F401
import feasibility


def _unit(uid, cpus, current=None):
    return {
        "unit_id": uid,
        "cpuset_cpus": list(cpus),
        "current_affinity_cpus": list(current if current is not None else cpus),
    }


class TestCheck(unittest.TestCase):
    def setUp(self):
        self.topo = testutils.a2_topo()

    def test_c1_min_cores(self):
        result = feasibility.check(self.topo, [_unit(0, [0, 1])])
        self.assertEqual([u["check"] for u in result["unresolved"]], ["C1"])
        self.assertTrue(result["flags"]["oversubscribed"])

    def test_flags_informative_when_not_triggered(self):
        result = feasibility.check(self.topo, [_unit(0, range(0, 10))])
        self.assertFalse(result["flags"]["oversubscribed"])
        self.assertFalse(result["flags"]["cross_numa"])

    def test_cross_numa_from_current_vs_target_node(self):
        unit = _unit(0, range(0, 10))
        unit.update({"current_node": 1, "target_node": 6})
        self.assertTrue(feasibility.check(self.topo, [unit])["flags"]["cross_numa"])

    def test_capacity_gap_cleared_when_not_positive(self):
        """复用已加载 unit 时，gap<=0 不能残留历史 capacity_gap。"""
        unit = dict(_unit(0, range(0, 10)), capacity_gap=99)
        feasibility.check(self.topo, [unit])
        self.assertNotIn("capacity_gap", unit)

    def test_c3_overlap(self):
        result = feasibility.check(self.topo, [_unit(0, range(0, 4)), _unit(1, range(3, 8))])
        self.assertEqual([u["check"] for u in result["unresolved"]], ["C3"])
        self.assertEqual(result["unresolved"][0]["unit_id"], [0, 1])

    def test_c5_capacity_gap(self):
        unit = _unit(0, [144, 145, 146, 147], current=range(144, 156))
        result = feasibility.check(self.topo, [unit])
        self.assertEqual(unit["capacity_gap"], 8)
        self.assertEqual([w["check"] for w in result["warnings"]], ["C5"])
        self.assertIn("local-first", result["warnings"][0]["reason"])

    def test_c5_global_slice_tail(self):
        topo = testutils.a3_topo()
        unit = _unit(0, topo["nodes"][0]["cpus"][:10], current=range(640))
        result = feasibility.check(topo, [unit])
        self.assertIn("按 node 均分后少于全机亲和", result["warnings"][0]["reason"])

    def test_c4_smt_flag(self):
        topo = testutils.a3_topo()
        result = feasibility.check(topo, [_unit(0, topo["nodes"][0]["cpus"][:10])])
        self.assertTrue(result["flags"]["smt_shared"])
        self.assertIn("C4", [w["check"] for w in result["warnings"]])

    def test_view_limited_flag_from_caller(self):
        """采集侧受限时必须透传到 flags，供 plan / 报告机器可读消费。"""
        unit = _unit(0, self.topo["nodes"][0]["cpus"][:10])
        self.assertTrue(feasibility.check(self.topo, [unit], view_limited=True)["flags"]["view_limited"])
        self.assertFalse(feasibility.check(self.topo, [unit])["flags"]["view_limited"])

    def test_c4_inferred_from_thread_siblings(self):
        """topo 缺 smt 字段时按线程兄弟核推断，避免 C4 永不触发。"""
        topo = testutils.a2_topo()
        topo.pop("smt", None)
        topo["cpus"][0]["thread_siblings"] = [0, 96]
        result = feasibility.check(topo, [_unit(0, topo["nodes"][0]["cpus"][:10])])
        self.assertTrue(result["flags"]["smt_shared"])
        self.assertIn("C4", [w["check"] for w in result["warnings"]])


if __name__ == "__main__":
    unittest.main()
