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

"""procfs.py 单元测试：numa_maps / interrupts 解析（mock 文本，不依赖真机）。"""

import io
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

import procfs

NUMA_MAPS = """55d0f000 default anon=2 dirty=2 N0=1 kernelpagesize_kB=4
55d10000 default anon=5 N0=2 N1=3 kernelpagesize_kB=4
55d20000 default anon=1 N=1 kernelpagesize_kB=4
"""

INTERRUPTS = """           CPU0       CPU1
  0:         10         20   IO-APIC   2-edge      timer
 24:          5          0   PCI-MSI 524288-edge  enp1s0f0-TxRx-0
"""


class TestNumaMapsSummary(unittest.TestCase):
    def test_parse_node_page_tokens(self):
        """只累计 N<node>=<pages>；无 node 归属的 N=1 不计入。"""
        with patch("builtins.open", lambda *args, **kwargs: io.StringIO(NUMA_MAPS)):
            summary = procfs.read_pid_numa_maps_summary(123)
        self.assertEqual(summary["total_pages"], 6)
        self.assertIn("N=1", NUMA_MAPS.split())  # 夹具确实含无 node 归属的页
        self.assertEqual(set(summary), {"total_pages"})  # 不输出未采集的 dirty_pages

    def test_node_less_token_ignored(self):
        text = "55d0f000 default anon=1 N=5 kernelpagesize_kB=4\n"
        with patch("builtins.open", lambda *args, **kwargs: io.StringIO(text)):
            self.assertEqual(procfs.read_pid_numa_maps_summary(123)["total_pages"], 0)

    def test_unreadable_returns_zero(self):
        def raise_oserror(*args, **kwargs):
            raise OSError("denied")

        with patch("builtins.open", raise_oserror):
            summary = procfs.read_pid_numa_maps_summary(123)
        self.assertEqual(summary["total_pages"], 0)


class TestReadInterrupts(unittest.TestCase):
    def test_desc_keeps_first_field(self):
        with patch.object(procfs, "_read_text", lambda path, *args: INTERRUPTS):
            data = procfs.read_interrupts()
        self.assertEqual(data["0"]["counts"], [10, 20])
        self.assertEqual(data["0"]["desc"], "IO-APIC 2-edge timer")
        self.assertEqual(data["24"]["desc"], "PCI-MSI 524288-edge enp1s0f0-TxRx-0")


if __name__ == "__main__":
    unittest.main()
