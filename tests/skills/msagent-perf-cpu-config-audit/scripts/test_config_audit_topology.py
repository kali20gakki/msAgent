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
topology.py 单元测试：CPU 核集字符串与列表互转（不依赖 /sys 的纯函数）。
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

import topology


class TestParseCpuList(unittest.TestCase):
    def test_ranges_and_single(self):
        self.assertEqual(topology.parse_cpu_list("0-3,8"), [0, 1, 2, 3, 8])

    def test_single(self):
        self.assertEqual(topology.parse_cpu_list("5"), [5])

    def test_empty(self):
        self.assertEqual(topology.parse_cpu_list(""), [])

    def test_dirty_text_ignored(self):
        self.assertEqual(topology.parse_cpu_list("0-1,abc,-3,7-"), [0, 1])


class TestFormatCpuList(unittest.TestCase):
    def test_compress_ranges(self):
        self.assertEqual(topology.format_cpu_list([0, 1, 2, 3, 8]), "0-3,8")

    def test_unordered_and_duplicated(self):
        self.assertEqual(topology.format_cpu_list([8, 0, 1, 0]), "0-1,8")

    def test_empty(self):
        self.assertEqual(topology.format_cpu_list([]), "")

    def test_round_trip(self):
        cpus = [0, 1, 2, 5, 6, 9]
        self.assertEqual(topology.parse_cpu_list(topology.format_cpu_list(cpus)), cpus)


class TestL3Domain(unittest.TestCase):
    def test_pick_level3_cache(self):
        base = "/sys/devices/system/cpu/cpu0/cache"
        values = {
            os.path.join(base, "index2", "level"): 2,
            os.path.join(base, "index2", "shared_cpu_list"): "0-3",
            os.path.join(base, "index3", "level"): 3,
            os.path.join(base, "index3", "shared_cpu_list"): "0-15",
        }
        with (
            patch.object(topology.os.path, "isdir", lambda path: True),
            patch.object(topology.os, "listdir", lambda path: ["index2", "index3"]),
            patch.object(topology, "_read_int", lambda path: values.get(path)),
            patch.object(topology, "_read_text", lambda path, *args: values.get(path)),
        ):
            self.assertEqual(topology.l3_domain(0), list(range(16)))

    def test_missing_cache_dir(self):
        with patch.object(topology.os.path, "isdir", lambda path: False):
            self.assertEqual(topology.l3_domain(0), [])


class TestNicNumaMap(unittest.TestCase):
    def test_skip_virtual_and_unknown_node(self):
        def fake_read_int(path):
            if path.endswith("/enp1s0f0/device/numa_node"):
                return 2
            if path.endswith("/enp2s0f0/device/numa_node"):
                return -1
            return None

        names = ["docker0", "enp1s0f0", "enp2s0f0", "lo"]
        with (
            patch.object(topology.os.path, "isdir", lambda path: True),
            patch.object(topology.os, "listdir", lambda path: names),
            patch.object(topology.os.path, "exists", lambda path: "/device" in path),
            patch.object(topology, "_read_int", fake_read_int),
        ):
            self.assertEqual(topology.nic_numa_map(), [{"dev": "enp1s0f0", "node": 2}])


class TestNodeIds(unittest.TestCase):
    """node id 不保证连续（如 0-1,4）：按实际 id 采集，不用 range(num_nodes())。"""

    def test_node_ids_from_online(self):
        with patch.object(
            topology, "_read_text", lambda path, *args: "0-1,4" if path.endswith("/node/online") else None
        ):
            self.assertEqual(topology.node_ids(), [0, 1, 4])
            self.assertEqual(topology.num_nodes(), 3)

    def test_online_unreadable_falls_back_to_possible(self):
        def fake(path, *args):
            if path.endswith("/node/possible"):
                return "0-1,4"
            return None

        with patch.object(topology, "_read_text", fake):
            self.assertEqual(topology.node_ids(), [0, 1, 4])

    def test_numa_topology_visits_real_ids(self):
        reads = {
            "/sys/devices/system/node/online": "0-1,4",
            "/sys/devices/system/node/node0/cpulist": "0-23",
            "/sys/devices/system/node/node1/cpulist": "24-47",
            "/sys/devices/system/node/node4/cpulist": "96-119",
        }
        with (
            patch.object(topology, "_read_text", lambda path, *args: reads.get(path)),
            patch.object(topology, "detect_snc", lambda: {"enabled": False, "sub_clusters": 1}),
        ):
            nodes = topology.numa_topology()["nodes"]
        self.assertEqual([item["node"] for item in nodes], [0, 1, 4])
        self.assertEqual(nodes[2]["cpus"], "96-119")


if __name__ == "__main__":
    unittest.main()
