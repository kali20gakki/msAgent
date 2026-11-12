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
action_build.py 单元测试
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

import action_build

REQUIRED_FIELDS = (
    "action_id",
    "type",
    "scope",
    "timing",
    "target",
    "params",
    "commands",
    "rollback",
    "expected_effect",
    "verify",
    "risk",
    "revertible",
    "restart_required",
    "source",
    "source_id",
    "handoff",
    "level",
    "cost",
    "benefit",
    "explain",
)


def _trace_with_findings(findings, card=0, victim_tid=4321):
    return {
        "cards": [
            {
                "card": card,
                "status": "OBSERVED",
                "verdict": "FAIL",
                "victim": {"tid": victim_tid, "comm": "app"},
                "findings": findings,
            }
        ]
    }


class TestLoadTemplates(unittest.TestCase):
    def test_default_template(self):
        templates = action_build._load_templates()
        self.assertIn("hint_templates", templates)
        self.assertIn("wait_actions", templates)
        self.assertIn("cost_rank", templates)
        self.assertIn("benefit_rank", templates)


class TestFromConfig(unittest.TestCase):
    def test_fill_fields(self):
        config = {
            "actions": [
                {
                    "action_id": "cfg-1",
                    "type": "SYSCTL",
                    "risk": "high",
                    "level": "L1",
                    "target": {"scope": "system_runtime", "key": "kernel.sched"},
                    "params": {"value": 1},
                    "commands": ["sysctl -w x=1"],
                    "rollback": ["sysctl -w x=0"],
                    "explain": "调大调度粒度",
                }
            ]
        }
        actions = action_build._from_config(config)
        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertEqual(action["source"], "config-audit")
        self.assertEqual(action["source_id"], "cfg-1")
        self.assertEqual(action["level"], "L1")
        self.assertEqual(action["benefit"], "high")
        self.assertEqual(action["cost"], "low")
        self.assertTrue(action["revertible"])
        self.assertFalse(action["restart_required"])
        self.assertEqual(action["expected_effect"], "调大调度粒度")

    def test_defaults(self):
        config = {"actions": [{"risk": "low"}]}
        action = action_build._from_config(config)[0]
        self.assertEqual(action["type"], "SYSCTL")
        self.assertEqual(action["scope"], "system_runtime")
        self.assertEqual(action["level"], "L1")
        self.assertEqual(action["benefit"], "medium")
        self.assertEqual(action["target"], {"scope": "system_runtime"})
        self.assertEqual(action["timing"], "running")
        self.assertFalse(action["revertible"])
        self.assertFalse(action["restart_required"])

    def test_upstream_exec_fields_win(self):
        config = {
            "actions": [
                {
                    "level": "L3",
                    "timing": "next_start",
                    "revertible": False,
                    "restart_required": False,
                    "rollback": ["undo"],
                }
            ]
        }
        action = action_build._from_config(config)[0]
        self.assertEqual(action["timing"], "next_start")
        self.assertFalse(action["revertible"])
        self.assertFalse(action["restart_required"])

    def test_l3_defaults_to_reboot_and_not_revertible(self):
        action = action_build._from_config({"actions": [{"level": "L3"}]})[0]
        self.assertEqual(action["timing"], "reboot")
        self.assertFalse(action["revertible"])
        self.assertTrue(action["restart_required"])

    def test_l1_without_rollback_is_not_revertible(self):
        action = action_build._from_config({"actions": [{"level": "L1", "commands": ["sysctl -w x=1"]}]})[0]
        self.assertEqual(action["timing"], "running")
        self.assertFalse(action["revertible"])
        self.assertFalse(action["restart_required"])


class TestFromTrace(unittest.TestCase):
    def test_taskset_hint(self):
        templates = action_build._load_templates()["hint_templates"]
        trace = _trace_with_findings(
            [
                {
                    "finding_id": "T-001",
                    "verdict": "FAIL",
                    "action_hint": "TASKSET_T3",
                    "problem": "线程越界",
                    "suggested_cpuset": "4-7",
                }
            ]
        )
        actions = action_build._from_trace(trace, templates)
        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertEqual(action["type"], "TASKSET")
        self.assertEqual(action["source"], "tracing")
        self.assertEqual(action["source_id"], "T-001")
        self.assertEqual(action["level"], "L0")
        self.assertEqual(action["handoff"], "bind_tune_handoff")
        self.assertEqual(action["risk"], "medium")
        self.assertFalse(action["restart_required"])
        self.assertEqual(action["target"], {"pid": None, "tids": [4321]})
        self.assertEqual(action["params"]["suggested_cpuset"], "4-7")
        self.assertEqual(action["card"], 0)

    def test_bind_tune_requires_restart(self):
        templates = action_build._load_templates()["hint_templates"]
        trace = _trace_with_findings([{"verdict": "FAIL", "action_hint": "BIND_TUNE"}])
        action = action_build._from_trace(trace, templates)[0]
        self.assertEqual(action["type"], "LAUNCH_WRAPPER")
        self.assertTrue(action["restart_required"])
        self.assertEqual(action["level"], "L2")

    def test_irq_affinity_low_risk(self):
        templates = action_build._load_templates()["hint_templates"]
        trace = _trace_with_findings([{"verdict": "FAIL", "action_hint": "IRQ_AFFINITY"}])
        action = action_build._from_trace(trace, templates)[0]
        self.assertEqual(action["risk"], "low")
        self.assertEqual(action["level"], "L2")

    def test_skip_non_fail_and_unknown_hint(self):
        templates = action_build._load_templates()["hint_templates"]
        findings = [
            {"verdict": "PASS", "action_hint": "TASKSET_T3"},
            {"verdict": "FAIL", "action_hint": "UNKNOWN"},
            {"verdict": "FAIL"},
        ]
        self.assertEqual(action_build._from_trace(_trace_with_findings(findings), templates), [])

    def test_skip_unobserved_card(self):
        templates = action_build._load_templates()["hint_templates"]
        trace = {"cards": [{"status": "UNOBSERVED", "findings": [{"verdict": "FAIL", "action_hint": "TASKSET_T3"}]}]}
        self.assertEqual(action_build._from_trace(trace, templates), [])

    def test_missing_tid(self):
        templates = action_build._load_templates()["hint_templates"]
        trace = {
            "cards": [
                {
                    "card": 0,
                    "status": "OBSERVED",
                    "victim": {},
                    "findings": [{"verdict": "FAIL", "action_hint": "TASKSET_T3"}],
                }
            ]
        }
        action = action_build._from_trace(trace, templates)[0]
        self.assertEqual(action["target"]["tids"], [])


class TestFromWait(unittest.TestCase):
    def test_network_wait(self):
        templates = action_build._load_templates()["wait_actions"]
        attribution = {"wait_breakdown": {"网络": 20.0}}
        actions = action_build._from_wait(attribution, templates)
        self.assertEqual(len(actions), 1)
        action = actions[0]
        self.assertEqual(action["type"], "IRQ_AFFINITY")
        self.assertEqual(action["source"], "perf-triage")
        self.assertEqual(action["source_id"], "wait-网络")
        self.assertEqual(action["level"], "L0")
        self.assertEqual(action["handoff"], "config_audit_handoff")
        self.assertTrue(action["revertible"])
        self.assertFalse(action["restart_required"])
        self.assertEqual(action["params"], {"wait_class": "网络", "wait_pct": 20.0})
        self.assertIn("20.0%", action["expected_effect"])

    def test_lock_wait_next_start(self):
        templates = action_build._load_templates()["wait_actions"]
        actions = action_build._from_wait({"wait_breakdown": {"锁/同步": 12.5}}, templates)
        action = actions[0]
        self.assertEqual(action["type"], "CODE_SUGGESTION")
        self.assertFalse(action["revertible"])
        self.assertTrue(action["restart_required"])
        self.assertEqual(action["level"], "L4")

    def test_pick_largest_class(self):
        templates = action_build._load_templates()["wait_actions"]
        actions = action_build._from_wait({"wait_breakdown": {"锁/同步": 10.0, "网络": 30.0}}, templates)
        self.assertEqual(actions[0]["params"]["wait_class"], "网络")

    def test_below_threshold(self):
        templates = action_build._load_templates()["wait_actions"]
        self.assertEqual(action_build._from_wait({"wait_breakdown": {"网络": 4.9}}, templates), [])

    def test_gate_uses_effective_window_ratio(self):
        """睡眠内份额高但整体等待很轻时不产生动作；动作里记录的是占窗贡献。"""
        templates = action_build._load_templates()["wait_actions"]
        self.assertEqual(action_build._from_wait({"wait_breakdown": {"网络": 80.0}, "wait_pct": 5.0}, templates), [])
        action = action_build._from_wait({"wait_breakdown": {"网络": 80.0}, "wait_pct": 10.0}, templates)[0]
        self.assertEqual(action["params"]["wait_pct"], 8.0)

    def test_empty_breakdown(self):
        templates = action_build._load_templates()["wait_actions"]
        self.assertEqual(action_build._from_wait({}, templates), [])
        self.assertEqual(action_build._from_wait({"wait_breakdown": {}}, templates), [])

    def test_upstream_wait_no_handoff(self):
        templates = action_build._load_templates()["wait_actions"]
        action = action_build._from_wait({"wait_breakdown": {"上游数据": 8.0}}, templates)[0]
        self.assertEqual(action["handoff"], "")
        self.assertEqual(action["cost"], "high")
        self.assertEqual(action["benefit"], "low")


class TestSortAssign(unittest.TestCase):
    def test_order_and_renumber(self):
        templates = action_build._load_templates()
        actions = [
            {"source": "tracing", "benefit": "high", "cost": "low", "tag": "trace"},
            {"source": "config-audit", "benefit": "medium", "cost": "low", "tag": "cfg-med"},
            {"source": "config-audit", "benefit": "high", "cost": "medium", "tag": "cfg-high"},
            {"source": "perf-triage", "benefit": "medium", "cost": "medium", "tag": "wait"},
            {"source": "tracing", "benefit": "high", "cost": "high", "tag": "trace-high-cost"},
        ]
        ordered = action_build._sort_assign(actions, templates)
        self.assertEqual([item["tag"] for item in ordered], ["cfg-high", "cfg-med", "trace", "trace-high-cost", "wait"])
        self.assertEqual([item["action_id"] for item in ordered], ["A-001", "A-002", "A-003", "A-004", "A-005"])


class TestBuildActions(unittest.TestCase):
    def test_full_contract(self):
        templates = action_build._load_templates()
        config = {"actions": [{"action_id": "cfg-1", "risk": "high", "level": "L1"}]}
        trace = _trace_with_findings([{"finding_id": "T-001", "verdict": "FAIL", "action_hint": "TASKSET_T3"}])
        attribution = {"wait_breakdown": {"网络": 20.0}}
        actions = action_build.build_actions(config, trace, attribution, templates)
        self.assertEqual(len(actions), 3)
        self.assertEqual([a["action_id"] for a in actions], ["A-001", "A-002", "A-003"])
        for action in actions:
            for field in REQUIRED_FIELDS:
                self.assertIn(field, action)
        self.assertEqual(actions[0]["source"], "config-audit")

    def test_empty_inputs(self):
        templates = action_build._load_templates()
        self.assertEqual(action_build.build_actions({}, {}, {}, templates), [])


class TestMain(unittest.TestCase):
    def test_writes_action_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(os.path.join(session, "00-config"))
            os.makedirs(os.path.join(session, "01-trace"))
            with open(os.path.join(session, "00-config", "config_triage.json"), "w", encoding="utf-8") as fh:
                json.dump({"actions": [{"action_id": "cfg-1", "risk": "high"}]}, fh)
            with open(os.path.join(session, "01-trace", "phase_verdict_1.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    _trace_with_findings([{"finding_id": "T-001", "verdict": "FAIL", "action_hint": "TASKSET_T3"}]), fh
                )
            with open(os.path.join(session, "attribution.json"), "w", encoding="utf-8") as fh:
                json.dump({"wait_breakdown": {"网络": 20.0}}, fh)
            buf = StringIO()
            with redirect_stdout(buf):
                code = action_build.main(["--session", session])
            self.assertEqual(code, 0)
            with open(os.path.join(session, "actions", "action_plan.json"), "r", encoding="utf-8") as fh:
                plan = json.load(fh)
            self.assertEqual(plan["count"], 3)
            self.assertEqual(len(plan["actions"]), 3)
            with open(os.path.join(session, "manifest.json"), "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["phases"]["actions"]["count"], 3)


class TestTemplateValidation(unittest.TestCase):
    """动作模板缺失/损坏/非对象时必须结构化报错退出，不允许 traceback。"""

    def _run(self, argv):
        buf = StringIO()
        with redirect_stdout(buf):
            code = action_build.main(argv)
        return code, json.loads(buf.getvalue())

    def test_missing_template_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, result = self._run(["--session", tmp, "--templates", os.path.join(tmp, "absent.json")])
        self.assertEqual(code, 2)
        self.assertFalse(result["ok"])
        self.assertIn("动作模板不可用", result["error"])

    def test_template_not_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "templates.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(["not", "a", "dict"], fh)
            code, result = self._run(["--session", tmp, "--templates", path])
        self.assertEqual(code, 2)
        self.assertIn("格式非法", result["error"])


if __name__ == "__main__":
    unittest.main()
