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

"""topology.py：CPU 列表解析、node 归属、亲和汇总与降级。"""

import unittest
from unittest import mock

import testutils  # noqa: F401
import topology


class TestCpuList(unittest.TestCase):
    def test_parse_cpulist(self):
        self.assertEqual(topology.parse_cpulist("0-3,8,10-11"), [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(topology.parse_cpulist(""), [])
        self.assertEqual(topology.parse_cpulist("5"), [5])

    def test_cpus_to_str_compact(self):
        self.assertEqual(topology.cpus_to_str([0, 1, 2, 3, 8]), "0-3,8")
        self.assertEqual(topology.cpus_to_str([]), "")


class TestNodeResolve(unittest.TestCase):
    def setUp(self):
        self.topo = testutils.a2_topo()

    def test_node_of_cpus_by_majority(self):
        # node6 = 144-167，含少量越界核时仍判 node6
        self.assertEqual(topology.node_of_cpus([144, 145, 0], self.topo["nodes"]), 6)
        self.assertIsNone(topology.node_of_cpus([], self.topo["nodes"]))

    def test_distance(self):
        self.assertEqual(topology.distance(self.topo, 0, 1), 11)
        self.assertEqual(topology.distance(self.topo, 0, 3), 25)
        self.assertIsNone(topology.distance(self.topo, 0, 99))

    def test_node_devices(self):
        self.assertEqual([n["id"] for n in topology.npu_of_node(self.topo, 6)], [0, 1])
        self.assertEqual([n["name"] for n in topology.nics_of_node(self.topo, 2)], ["enp67s0f0"])
        self.assertEqual([s["name"] for s in topology.storage_of_node(self.topo, 0)], ["nvme0n1"])


class TestAffinitySummary(unittest.TestCase):
    def test_group_shared(self):
        npus = [{"id": i, "affinity": aff} for i, aff in ((0, "288-383"), (1, "288-383"), (2, "96-191"), (3, "96-191"))]
        granularity, groups = topology.affinity_summary(npus)
        self.assertEqual(granularity, "group")
        self.assertEqual(groups["288-383"], [0, 1])
        self.assertEqual(groups["96-191"], [2, 3])

    def test_per_npu(self):
        granularity, _ = topology.affinity_summary([{"id": 0, "affinity": "0-23"}, {"id": 1, "affinity": "24-47"}])
        self.assertEqual(granularity, "per_npu")

    def test_no_affinity(self):
        granularity, groups = topology.affinity_summary([{"id": 0, "affinity": None}])
        self.assertEqual(granularity, "none")
        self.assertEqual(groups, {})


class TestCollectMeta(unittest.TestCase):
    """meta 的 die 信息与纳秒时间戳：缺失与单 die 机型不能混淆。"""

    def _meta(self, npu_info):
        with (
            mock.patch.object(topology, "_collect_numa_nodes", return_value=([], {}, "sysfs")),
            mock.patch.object(topology, "_collect_cpu_topology", return_value=([], [], False)),
            mock.patch.object(topology, "_collect_npu_topology", return_value=[{"id": 0, "affinity_cpus": []}]),
            mock.patch.object(topology, "_collect_nics", return_value=[]),
            mock.patch.object(topology, "_collect_storage", return_value=[]),
            mock.patch.object(topology, "_ub_present", return_value=False),
        ):
            return topology.collect(npu_info)["meta"]

    def test_die_unavailable_is_explicit(self):
        meta = self._meta({})
        self.assertFalse(meta["die_available"])
        self.assertIsNone(meta["die_count"])
        self.assertEqual(meta["die_groups"], [])
        self.assertEqual(meta["die_source"], "unavailable")

    def test_die_count_matches_groups(self):
        meta = self._meta({"die_map": {"logic_ids": [0, 1], "groups": [[0, 1]]}})
        self.assertTrue(meta["die_available"])
        self.assertEqual(meta["die_count"], 2)
        self.assertEqual(meta["die_groups"], [[0, 1]])
        self.assertEqual(meta["die_source"], "npu_smi_info_m")

    def test_ts_ns_is_nanosecond_precision(self):
        meta = self._meta({})
        self.assertIsInstance(meta["ts_ns"], int)
        self.assertGreater(meta["ts_ns"], 10**18)


class TestSanitizeAffinity(unittest.TestCase):
    def test_invalid_affinity_degrades(self):
        """亲和区间与可见 CPU 无交集时丢弃并降级，而不是产出空核集。"""
        npus = [
            {
                "id": 0,
                "affinity": "9000-9095",
                "affinity_cpus": list(range(9000, 9096)),
                "affinity_source": "npu_smi_topo",
                "numa_node": 0,
            }
        ]
        has_affinity, degraded = topology.sanitize_affinity(npus, list(range(640)))
        self.assertFalse(has_affinity)
        self.assertTrue(degraded)
        self.assertEqual(npus[0]["affinity_cpus"], [])
        self.assertEqual(npus[0]["affinity_note"], "affinity_invalid")
        self.assertEqual(npus[0]["affinity_source"], "invalid")

    def test_valid_affinity_kept(self):
        npus = [{"id": 0, "affinity": "0-95", "affinity_cpus": list(range(96)), "affinity_source": "npu_smi_topo"}]
        has_affinity, degraded = topology.sanitize_affinity(npus, list(range(640)))
        self.assertTrue(has_affinity)
        self.assertFalse(degraded)
        self.assertEqual(len(npus[0]["affinity_cpus"]), 96)


if __name__ == "__main__":
    unittest.main()
