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

"""budget.py：事件集规模、时长矩阵、预估与四步收窄预算。"""

import os
import sys
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
        "msagent-perf-cpu-tracing",
        "scripts",
    ),
)

import budget


class TestEventsForTier(unittest.TestCase):
    def test_tier_sizes(self):
        self.assertEqual(len(budget.events_for_tier("min")), 4)
        self.assertEqual(len(budget.events_for_tier("std")), 8)
        self.assertEqual(len(budget.events_for_tier("ext")), 10)

    def test_unknown_tier_falls_back_to_std(self):
        self.assertEqual(budget.events_for_tier("unknown"), budget.TIERS["std"])
        self.assertEqual(budget.events_for_tier(None), budget.TIERS["std"])

    def test_returns_copy(self):
        """返回副本，外部改动不应污染全局事件集定义。"""
        events = budget.events_for_tier("min")
        events.append("bogus")
        self.assertNotIn("bogus", budget.TIERS["min"])


class TestDurationCap(unittest.TestCase):
    def test_std_boundaries(self):
        self.assertEqual(budget.duration_cap(8, "std"), 120)
        self.assertEqual(budget.duration_cap(9, "std"), 60)
        self.assertEqual(budget.duration_cap(16, "std"), 60)
        self.assertEqual(budget.duration_cap(24, "std"), 45)
        self.assertEqual(budget.duration_cap(32, "std"), 30)

    def test_above_matrix_uses_last_row(self):
        self.assertEqual(budget.duration_cap(100, "std"), 30)

    def test_min_and_ext(self):
        self.assertEqual(budget.duration_cap(8, "min"), 180)
        self.assertEqual(budget.duration_cap(8, "ext"), 60)

    def test_unknown_tier_falls_back_to_std(self):
        self.assertEqual(budget.duration_cap(9, "unknown"), 60)


class TestEstimate(unittest.TestCase):
    def test_events_and_bytes(self):
        result = budget.estimate(8, "std", 30, rate_per_core=300.0, cards=2)
        self.assertEqual(result["events"], 300 * 8 * 30 * 2)
        self.assertEqual(result["db_bytes"], result["events"] * budget.BYTES_PER_EVENT_DB)
        self.assertEqual(result["text_bytes"], result["events"] * budget.BYTES_PER_EVENT_TEXT)
        self.assertEqual(result["event_types"], 8)
        self.assertEqual(result["cards"], 2)
        self.assertEqual(result["duration_s"], 30)

    def test_cards_zero_treated_as_one(self):
        result = budget.estimate(8, "min", 10, rate_per_core=100.0, cards=0)
        self.assertEqual(result["events"], 100 * 8 * 10)


class TestPlan(unittest.TestCase):
    def test_ok_without_adjustments(self):
        ok, result, adjustments = budget.plan(8, "std", 30)
        self.assertTrue(ok)
        self.assertEqual(adjustments, [])
        self.assertEqual(result["segments"], 1)
        self.assertEqual(result["session_seconds"], 30)
        self.assertEqual(result["events"], 300 * 8 * 30)

    def test_duration_over_cap_is_clamped(self):
        ok, result, adjustments = budget.plan(8, "std", 200)
        self.assertTrue(ok)
        self.assertEqual(adjustments[0], {"step": "duration", "from": 200, "to": 120})
        self.assertEqual(result["duration_s"], 120)
        self.assertEqual(result["session_seconds"], 120)

    def test_single_card_over_32_cpus_rejected(self):
        ok, result, adjustments = budget.plan(33, "std", 30)
        self.assertFalse(ok)
        self.assertEqual(adjustments[0]["step"], "reject")
        self.assertEqual(result["cpus"], 33)
        self.assertEqual(len(adjustments), 1)

    def test_unknown_tier_keeps_custom_label(self):
        """非内置档位保留 custom 语义，不再改写成 std。"""
        ok, result, _ = budget.plan(8, "bogus", 30)
        self.assertTrue(ok)
        self.assertEqual(result["tier"], "custom")

    def test_custom_tier_skips_event_downgrade(self):
        """自定义事件集不参与降档：只收时长与核集，tier 始终保持 custom。"""
        ok, result, adjustments = budget.plan(16, "custom", 30, rate_per_core=3000.0)
        self.assertTrue(ok)
        self.assertEqual(result["tier"], "custom")
        self.assertNotIn("events", [item["step"] for item in adjustments])
        self.assertEqual([item["step"] for item in adjustments], ["duration", "cpus"])
        self.assertEqual(result["duration_s"], 15)
        self.assertEqual(result["cpus"], 8)

    def test_narrowing_order_events_duration_cpus(self):
        """超限时按「事件集 -> 时长 -> 核集」收窄，且顺序即调整列表顺序。"""
        ok, result, adjustments = budget.plan(32, "ext", 15, rate_per_core=2000.0)
        self.assertTrue(ok)
        steps = [(item["step"], item.get("from"), item.get("to")) for item in adjustments]
        self.assertEqual(
            steps,
            [
                ("events", "ext", "std"),
                ("events", "std", "min"),
                ("duration", 15, 10),
                ("cpus", 32, 16),
            ],
        )
        self.assertEqual(result["tier"], "min")
        self.assertEqual(result["duration_s"], 10)
        self.assertEqual(result["cpus"], 16)
        self.assertFalse(result.get("rejected", False))

    def test_verify_mode_does_not_reduce_cpus(self):
        """verify 模式减核会失去验证意义，应直接拒绝而不是缩核。"""
        ok, result, adjustments = budget.plan(32, "min", 30, rate_per_core=2000.0, verify=True)
        self.assertFalse(ok)
        self.assertTrue(result["rejected"])
        self.assertNotIn("cpus", [item["step"] for item in adjustments])
        self.assertEqual(result["cpus"], 32)
        self.assertEqual(adjustments[-1]["step"], "reject")

    def test_non_verify_reduces_cpus_until_fit(self):
        ok, result, adjustments = budget.plan(32, "min", 30, rate_per_core=2000.0)
        self.assertTrue(ok)
        steps = [item["step"] for item in adjustments]
        self.assertIn("cpus", steps)
        self.assertEqual(result["cpus"], 16)
        self.assertEqual(result["duration_s"], 15)

    def test_multi_card_serial_segments(self):
        ok, result, adjustments = budget.plan(24, "std", 30, cards=2)
        self.assertTrue(ok)
        self.assertEqual(adjustments[0]["step"], "per-card")
        self.assertEqual(result["segments"], 2)
        self.assertEqual(result["session_seconds"], 60)

    def test_session_over_600s_not_ok(self):
        ok, result, adjustments = budget.plan(8, "std", 30, cards=25)
        self.assertFalse(ok)
        self.assertEqual(result["segments"], 25)
        self.assertEqual(result["session_seconds"], 750)
        self.assertEqual(adjustments[-1]["step"], "reject")


class TestFormatBytes(unittest.TestCase):
    def test_units_and_rounding(self):
        self.assertEqual(budget.format_bytes(None), "0.0B")
        self.assertEqual(budget.format_bytes(0), "0.0B")
        self.assertEqual(budget.format_bytes(512), "512.0B")
        self.assertEqual(budget.format_bytes(1024), "1.0KB")
        self.assertEqual(budget.format_bytes(1536), "1.5KB")
        self.assertEqual(budget.format_bytes(1024**3), "1.0GB")

    def test_beyond_gb_stays_gb(self):
        self.assertEqual(budget.format_bytes(1024**4), "1024.0GB")


if __name__ == "__main__":
    unittest.main()
