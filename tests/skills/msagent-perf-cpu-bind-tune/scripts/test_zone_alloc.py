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

"""zone_alloc.py：node 内三区切分与多 unit 均分。"""

import unittest

import testutils  # noqa: F401
import topology
import zone_alloc


def _unit(uid, node, cpus):
    return {"unit_id": uid, "target_node": node, "target_cpus": list(cpus)}


class TestZoneAlloc(unittest.TestCase):
    def setUp(self):
        self.topo = testutils.a2_topo()

    def test_reserved_zone_tail(self):
        zones = zone_alloc.allocate(self.topo, [], {"irq_reserve_per_node": 2, "interference_reserve_per_node": 1})
        node0 = zones["zones"]["0"]
        self.assertEqual(node0["cpulist"], "0-23")
        self.assertNotIn("cpus", node0)
        self.assertEqual(node0["business"], "0-20")
        self.assertEqual(node0["system_irq"], "21-22")
        self.assertEqual(node0["interference"], "23")

    def test_two_units_split_without_overlap(self):
        units = [_unit(0, 6, range(144, 168)), _unit(1, 6, range(144, 168))]
        zones = zone_alloc.allocate(self.topo, units, {"irq_reserve_per_node": 2, "interference_reserve_per_node": 1})
        a = set(zones["alloc"][0])
        b = set(zones["alloc"][1])
        self.assertTrue(a and b)
        self.assertFalse(a & b)
        self.assertTrue((a | b) <= set(range(144, 165)))  # 业务区 144-164

    def test_no_reserve_when_zone_too_small(self):
        """预留超过 node 半数时放弃预留，全部给业务区。"""
        topo = testutils.a3_topo(nodes=1, cpus_per_node=2, npus=1)
        zones = zone_alloc.allocate(topo, [], {"irq_reserve_per_node": 2, "interference_reserve_per_node": 1})
        node0 = zones["zones"]["0"]
        self.assertEqual(node0["business"], "0-1")
        self.assertEqual(node0["system_irq"], "")
        self.assertEqual(node0["interference"], "")

    def test_target_cpus_preferred_over_node_cpus(self):
        units = [_unit(0, 0, topology.parse_cpulist("2-9"))]
        zones = zone_alloc.allocate(self.topo, units, {"irq_reserve_per_node": 2, "interference_reserve_per_node": 1})
        self.assertEqual(set(zones["alloc"][0]), set(range(2, 7)))


if __name__ == "__main__":
    unittest.main()
