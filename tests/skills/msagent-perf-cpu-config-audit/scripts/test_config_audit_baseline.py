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
baseline.py 单元测试：基线构建、落盘/读取与差异方向判定。
"""

import os
import sys
import tempfile
import unittest

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

import baseline


def _facts(**items):
    return [{"item": key, "value": value, "source": "/proc/x"} for key, value in items.items()]


class TestBuild(unittest.TestCase):
    def test_only_baseline_items_kept(self):
        data = baseline.build_baseline(_facts(scaling_governor="performance", swappiness=60, npu_count=8))
        self.assertEqual(data["type"], "hostcpu-baseline")
        self.assertEqual(data["values"]["scaling_governor"], "performance")
        self.assertEqual(data["values"]["swappiness"], 60)
        self.assertNotIn("npu_count", data["values"])


class TestRoundTrip(unittest.TestCase):
    def test_save_and_load(self):
        data = baseline.build_baseline(_facts(scaling_governor="powersave"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "hostcpu-baseline.json")
            baseline.save_baseline(data, path)
            loaded = baseline.load_baseline(path)
        self.assertEqual(loaded["values"]["scaling_governor"], "powersave")

    def test_load_missing_or_invalid(self):
        self.assertEqual(baseline.load_baseline(""), {})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("not json")
            self.assertEqual(baseline.load_baseline(path), {})


class TestDiff(unittest.TestCase):
    def _diff(self, old, new):
        previous = {"values": old}
        current = {"values": new}
        return {item["item"]: item for item in baseline.diff_baseline(current, previous)}

    def test_improved(self):
        diff = self._diff({"scaling_governor": "powersave"}, {"scaling_governor": "performance"})
        self.assertEqual(diff["scaling_governor"]["direction"], "improved")

    def test_degraded(self):
        diff = self._diff({"thp_enabled": "madvise"}, {"thp_enabled": "never"})
        self.assertEqual(diff["thp_enabled"]["direction"], "degraded")

    def test_changed_for_plain_value(self):
        diff = self._diff({"swappiness": 60}, {"swappiness": 10})
        self.assertEqual(diff["swappiness"]["direction"], "changed")

    def test_same_value_skipped(self):
        self.assertEqual(self._diff({"swappiness": 60}, {"swappiness": 60}), {})

    def test_numa_balancing_zero_is_good(self):
        diff = self._diff({"numa_balancing": "1"}, {"numa_balancing": "0"})
        self.assertEqual(diff["numa_balancing"]["direction"], "improved")
        diff = self._diff({"numa_balancing": "0"}, {"numa_balancing": "1"})
        self.assertEqual(diff["numa_balancing"]["direction"], "degraded")

    def test_zone_reclaim_zero_is_good(self):
        diff = self._diff({"zone_reclaim_mode": "0"}, {"zone_reclaim_mode": "1"})
        self.assertEqual(diff["zone_reclaim_mode"]["direction"], "degraded")

    def test_governor_uses_dominant_value(self):
        diff = self._diff(
            {"scaling_governor": {"powersave": 80, "performance": 4}}, {"scaling_governor": {"performance": 80}}
        )
        self.assertEqual(diff["scaling_governor"]["direction"], "improved")


if __name__ == "__main__":
    unittest.main()
