#!/usr/bin/python3
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

"""
quantify.py 单元测试
"""

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

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
        "msagent-perf-cpu-perf-triage",
        "scripts",
    ),
)

import quantify


def _wait_ledger(interval_count=3, by_class=None):
    if by_class is None:
        by_class = {"net": {"share": 0.6, "ns": 300}, "io": {"share": 0.4, "ns": 200}}
    return {"interval_count": interval_count, "total_ns": 500, "by_class": by_class}


def _card(card=0, disturb=0.2, wait=0.5, verdict="FAIL", ledger=None, accounting=None):
    if accounting is None:
        accounting = {"window_ns": 1000, "disturb_ratio": disturb, "wait_ratio": wait}
    return {
        "card": card,
        "status": "OBSERVED",
        "verdict": verdict,
        "coverage": 0.9,
        "accounting": accounting,
        "summary": {"wait": _wait_ledger() if ledger is None else ledger},
    }


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


class TestConfigTyping(unittest.TestCase):
    def test_config_typing_passthrough(self):
        config = {
            "impact_level": "高",
            "config_primary_candidate": True,
            "status": "REUSED",
            "findings_count": 2,
            "action_count": 1,
            "top_findings": [{"finding_id": "R001"}],
            "degradation": ["cfg-note"],
        }
        result = quantify.build_attribution(config, {"cards": []})
        typing_info = result["config_typing"]
        self.assertEqual(typing_info["level"], "高")
        self.assertTrue(typing_info["candidate"])
        self.assertEqual(typing_info["status"], "REUSED")
        self.assertEqual(typing_info["findings_count"], 2)
        self.assertEqual(typing_info["top"][0]["finding_id"], "R001")
        self.assertIn("cfg-note", result["degradation"])

    def test_config_defaults(self):
        result = quantify.build_attribution({}, {})
        typing_info = result["config_typing"]
        self.assertEqual(typing_info["level"], "无")
        self.assertFalse(typing_info["candidate"])
        self.assertEqual(typing_info["status"], "UNAVAILABLE")


class TestNoObservation(unittest.TestCase):
    def test_unobserved_cards(self):
        trace = {"cards": [{"card": 0, "status": "UNOBSERVED"}]}
        result = quantify.build_attribution({}, trace)
        self.assertIsNone(result["disturb_pct"])
        self.assertIsNone(result["wait_pct"])
        self.assertIsNone(result["residual_pct"])
        self.assertFalse(result["converged"])
        self.assertEqual(result["basis"], "none")
        self.assertEqual(result["efficiency"], "not_observable")
        self.assertTrue(any("未取得可观测数据" in item for item in result["degradation"]))

    def test_observed_without_accounting(self):
        trace = {"cards": [{"card": 0, "status": "OBSERVED", "summary": {}}]}
        result = quantify.build_attribution({}, trace)
        self.assertIsNone(result["disturb_pct"])
        self.assertIsNone(result["residual_pct"])
        self.assertEqual(result["basis"], "none")
        self.assertTrue(any("未产出 accounting" in item for item in result["degradation"]))


class TestSingleCard(unittest.TestCase):
    def test_converged(self):
        result = quantify.build_attribution({}, {"cards": [_card()]})
        self.assertEqual(result["disturb_pct"], 20.0)
        self.assertEqual(result["wait_pct"], 50.0)
        self.assertEqual(result["residual_pct"], 30.0)
        self.assertTrue(result["converged"])
        self.assertEqual(result["wait_breakdown"], {"网络": 60.0, "IO 等待": 40.0})
        self.assertEqual(result["basis"], "card0")
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["window_ns"], 1000)
        self.assertEqual(result["degradation"], [])

    def test_not_converged(self):
        result = quantify.build_attribution({}, {"cards": [_card(disturb=0.05, wait=0.1)]})
        self.assertEqual(result["residual_pct"], 85.0)
        self.assertFalse(result["converged"])

    def test_empty_wait_ledger(self):
        result = quantify.build_attribution({}, {"cards": [_card(ledger=_wait_ledger(interval_count=0, by_class={}))]})
        self.assertEqual(result["wait_breakdown"], {})
        self.assertTrue(any("等待账为空" in item for item in result["degradation"]))

    def test_accounting_from_summary(self):
        card = {
            "card": 2,
            "status": "OBSERVED",
            "verdict": "PASS",
            "summary": {
                "accounting": {"window_ns": 1000, "disturb_ratio": 0.1, "wait_ratio": 0.2},
                "wait": _wait_ledger(),
            },
        }
        result = quantify.build_attribution({}, {"cards": [card]})
        self.assertEqual(result["basis"], "card2")
        self.assertEqual(result["disturb_pct"], 10.0)
        self.assertEqual(result["wait_pct"], 20.0)
        self.assertEqual(result["verdict"], "PASS")


class TestAccountingSafety(unittest.TestCase):
    def test_numeric_strings_are_accepted(self):
        card = _card(accounting={"window_ns": "1000", "disturb_ratio": "0.25", "wait_ratio": "0.5"})
        result = quantify.build_attribution({}, {"cards": [card]})
        self.assertEqual(result["window_ns"], 1000)
        self.assertEqual(result["disturb_pct"], 25.0)
        self.assertEqual(result["wait_pct"], 50.0)

    def test_invalid_ratio_degrades_without_crash(self):
        card = _card(accounting={"window_ns": 1000, "disturb_ratio": "bad", "wait_ratio": 0.5})
        result = quantify.build_attribution({}, {"cards": [card]})
        self.assertEqual(result["disturb_pct"], 0.0)
        self.assertEqual(result["wait_pct"], 50.0)
        self.assertTrue(any("数值非法" in item and "disturb_ratio" in item for item in result["degradation"]))

    def test_ratio_is_clamped(self):
        card = _card(accounting={"window_ns": 1000, "disturb_ratio": 1.5, "wait_ratio": -0.2})
        result = quantify.build_attribution({}, {"cards": [card]})
        self.assertEqual(result["disturb_pct"], 100.0)
        self.assertEqual(result["wait_pct"], 0.0)

    def test_converged_threshold_shared_with_workspace(self):
        self.assertEqual(quantify.RESIDUAL_CONVERGED_MAX, quantify.workspace.RESIDUAL_CONVERGED_MAX)


class TestWorstCard(unittest.TestCase):
    def test_multi_card_picks_worst(self):
        cards = [_card(card=0, disturb=0.05, wait=0.1), _card(card=1, disturb=0.3, wait=0.4)]
        result = quantify.build_attribution({}, {"cards": cards})
        self.assertEqual(result["basis"], "card1")
        self.assertEqual(result["disturb_pct"], 30.0)
        self.assertEqual(result["wait_pct"], 40.0)
        self.assertTrue(any("最差卡 card1" in item for item in result["degradation"]))


class TestWaitBreakdown(unittest.TestCase):
    def test_ratio_fallback(self):
        wait = {"interval_count": 2, "by_class": {"lock": {"ratio": 0.4}}}
        self.assertEqual(quantify._wait_breakdown(wait, 1000), {"锁/同步": 40.0})

    def test_unknown_class_kept(self):
        wait = {"interval_count": 1, "by_class": {"dma": {"share": 0.5}}}
        self.assertEqual(quantify._wait_breakdown(wait, 1000), {"dma": 50.0})

    def test_zero_window_keeps_breakdown(self):
        # by_class 是相对份额，与窗口时长无关；window_ns=0 不得静默丢分类
        self.assertEqual(quantify._wait_breakdown(_wait_ledger(), 0), {"网络": 60.0, "IO 等待": 40.0})

    def test_missing_by_class(self):
        self.assertEqual(quantify._wait_breakdown({"interval_count": 2}, 1000), {})

    def test_none_wait(self):
        self.assertEqual(quantify._wait_breakdown(None, 1000), {})


class TestAssemble(unittest.TestCase):
    def test_residual_floor_at_zero(self):
        result = quantify._assemble({"level": "无"}, 65.0, 60.0, {}, [])
        self.assertEqual(result["residual_pct"], 0.0)
        self.assertTrue(result["converged"])

    def test_boundary_30_is_converged(self):
        result = quantify._assemble({"level": "无"}, 20.0, 50.0, {}, [])
        self.assertEqual(result["residual_pct"], 30.0)
        self.assertTrue(result["converged"])

    def test_over_30_not_converged(self):
        result = quantify._assemble({"level": "无"}, 20.0, 49.0, {}, [])
        self.assertFalse(result["converged"])

    def test_both_none(self):
        result = quantify._assemble({"level": "无"}, None, None, {}, [])
        self.assertIsNone(result["residual_pct"])
        self.assertFalse(result["converged"])


class TestMain(unittest.TestCase):
    def test_writes_attribution_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            _write_json(
                os.path.join(session, "00-config", "config_triage.json"),
                {
                    "status": "RERUN",
                    "impact_level": "高",
                    "config_primary_candidate": True,
                    "findings_count": 1,
                    "action_count": 1,
                    "top_findings": [],
                },
            )
            _write_json(
                os.path.join(session, "01-trace", "phase_verdict_1.json"), {"cards": [_card()], "degradation": []}
            )
            buf = StringIO()
            with redirect_stdout(buf):
                code = quantify.main(["--session", session])
            self.assertEqual(code, 0)
            with open(os.path.join(session, "attribution.json"), "r", encoding="utf-8") as fh:
                result = json.load(fh)
            self.assertEqual(result["disturb_pct"], 20.0)
            self.assertEqual(result["wait_pct"], 50.0)
            self.assertTrue(result["config_typing"]["candidate"])
            with open(os.path.join(session, "manifest.json"), "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["phases"]["attribution"]["residual_pct"], 30.0)


if __name__ == "__main__":
    unittest.main()
