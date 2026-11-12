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
report.py 单元测试
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

_SCRIPTS = os.path.abspath(
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
    )
)


def _load_skill_module(name):
    """按文件路径以唯一模块名加载，避免多 skill 同名 report/workspace 串味。"""
    spec = importlib.util.spec_from_file_location("perf_triage_" + name, os.path.join(_SCRIPTS, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_saved_workspace = sys.modules.get("workspace")
_saved_report = sys.modules.get("report")
workspace = _load_skill_module("workspace")
sys.modules["workspace"] = workspace  # report.py 内部 import workspace 需命中本 skill 实现
report = _load_skill_module("report")
for _name, _saved in (("workspace", _saved_workspace), ("report", _saved_report)):
    if _saved is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _saved


def _write_json(path, data):
    workspace.write_json(path, data)


def _read_text(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class TestResolveHost(unittest.TestCase):
    def test_trace_host_wins(self):
        trace = {"cards": [{"summary": {"meta": {"host": "tracehost"}}}]}
        host = report._resolve_host("/tmp/sess", {"host": "cfghost"}, trace)
        self.assertEqual(host, "tracehost")

    def test_config_host_second(self):
        with patch.object(report.socket, "gethostname", return_value="sockhost"):
            host = report._resolve_host("/tmp/sess", {"host": "cfghost"}, {"cards": []})
        self.assertEqual(host, "cfghost")

    def test_socket_hostname_fallback(self):
        with patch.object(report.socket, "gethostname", return_value="sockhost"):
            host = report._resolve_host("/tmp/sess", {}, {"cards": [{"summary": {}}]})
        self.assertEqual(host, "sockhost")

    def test_run_id_prefix_fallback(self):
        with patch.object(report.socket, "gethostname", side_effect=OSError):
            host = report._resolve_host("/tmp/nodeA_20260101-120000", {}, {})
        self.assertEqual(host, "nodeA")

    def test_run_id_prefix_wins_over_socket(self):
        with patch.object(report.socket, "gethostname", return_value="sockhost"):
            host = report._resolve_host("/tmp/nodeA_20260101-120000", {}, {})
        self.assertEqual(host, "nodeA")

    def test_generic_host_when_run_id_has_no_prefix(self):
        with patch.object(report.socket, "gethostname", side_effect=OSError):
            host = report._resolve_host("/tmp/session", {}, {})
        self.assertEqual(host, "host")


class TestConclusion(unittest.TestCase):
    def test_config_primary(self):
        text = report._conclusion({"config_primary_candidate": True, "impact_level": "高"}, {}, {})
        self.assertIn("配置主因候选", text)
        self.assertIn("高", text)

    def test_converged(self):
        text = report._conclusion({}, {}, {"converged": True, "residual_pct": 8.0})
        self.assertIn("已收敛", text)
        self.assertIn("8.0%", text)

    def test_cannot_attribute(self):
        text = report._conclusion({}, {}, {"disturb_pct": None, "wait_pct": None})
        self.assertIn("无法归因", text)

    def test_config_only(self):
        text = report._conclusion({"findings_count": 1}, {"cards": []}, {"disturb_pct": None, "wait_pct": None})
        self.assertIn("仅完成配置判型", text)

    def test_not_converged(self):
        text = report._conclusion(
            {}, {}, {"disturb_pct": 20.0, "wait_pct": 30.0, "residual_pct": 50.0, "converged": False}
        )
        self.assertIn("未收敛", text)
        self.assertIn("50.0%", text)


class TestConfigDetail(unittest.TestCase):
    def test_reused_with_origin(self):
        config = {
            "status": "REUSED",
            "findings_count": 2,
            "action_count": 1,
            "impact_level": "高",
            "source": "/ws/workspaces/nodeA/20260101-120000/00-config/findings.json",
        }
        detail = report._config_detail(config)
        self.assertIn("REUSED", detail)
        self.assertIn("Finding 2 条 / Action 1 条", detail)
        self.assertIn("复用自 20260101-120000", detail)

    def test_rerun_without_origin(self):
        config = {
            "status": "RERUN",
            "findings_count": 0,
            "action_count": 0,
            "source": "/ws/workspaces/nodeA/20260101-120000/00-config/findings.json",
        }
        detail = report._config_detail(config)
        self.assertNotIn("复用自", detail)

    def test_missing(self):
        self.assertIn("配置维缺失", report._config_detail({"status": "SKIPPED"}))
        self.assertIn("未执行", report._config_detail({}))


class TestTraceDetail(unittest.TestCase):
    def test_no_cards(self):
        self.assertIn("无可观测卡", report._trace_detail([]))

    def test_card_line(self):
        card = {
            "card": 0,
            "verdict": "FAIL",
            "coverage": 0.923,
            "accounting": {"disturb_ratio": 0.1, "wait_ratio": 0.5},
        }
        detail = report._trace_detail([card])
        self.assertEqual(detail, "卡0：FAIL · 覆盖率 92% · 打扰 10.0% · 等待 50.0%")

    def test_accounting_from_summary(self):
        card = {
            "card": 1,
            "verdict": "PASS",
            "coverage": 1.0,
            "summary": {"accounting": {"disturb_ratio": 0.02, "wait_ratio": 0.03}},
        }
        self.assertIn("打扰 2.0%", report._trace_detail([card]))

    def test_pct_defaults_to_zero(self):
        self.assertEqual(report._pct({"card": 0}, "disturb_ratio"), 0.0)

    def test_pct_handles_invalid_upstream_value(self):
        card = {"card": 0, "accounting": {"disturb_ratio": "bad", "wait_ratio": None}}
        self.assertEqual(report._pct(card, "disturb_ratio"), 0.0)
        self.assertEqual(report._pct(card, "wait_ratio"), 0.0)

    def test_trace_detail_handles_invalid_coverage(self):
        card = {
            "card": 0,
            "verdict": "FAIL",
            "coverage": "bad",
            "accounting": {"disturb_ratio": 0.1, "wait_ratio": 0.2},
        }
        detail = report._trace_detail([card])
        self.assertIn("覆盖率 0%", detail)


class TestBuildPayload(unittest.TestCase):
    def test_payload_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(os.path.join(session, "00-config"))
            os.makedirs(os.path.join(session, "01-trace", "card0"))
            for rel in ("00-config/cfg.html", "01-trace/card0/trace.html", "01-trace/top.html"):
                with open(os.path.join(session, rel), "w", encoding="utf-8"):
                    pass

            config = {
                "status": "RERUN",
                "findings_count": 1,
                "action_count": 1,
                "impact_level": "中",
                "host": "cfghost",
            }
            trace = {
                "status": "OBSERVED",
                "cards": [
                    {
                        "card": 0,
                        "status": "OBSERVED",
                        "verdict": "FAIL",
                        "coverage": 0.9,
                        "accounting": {"disturb_ratio": 0.1, "wait_ratio": 0.2},
                    },
                    {"card": 1, "status": "UNOBSERVED"},
                ],
            }
            attribution = {"converged": False, "residual_pct": 70.0, "degradation": ["d1"]}
            action_plan = {"actions": [{"action_id": "A-001"}]}

            payload = report.build_payload(session, config, trace, attribution, action_plan)
            self.assertEqual(payload["meta"]["run_id"], "nodeA_20260101-120000")
            self.assertEqual(payload["meta"]["host"], "cfghost")
            self.assertEqual(len(payload["cards"]), 1)
            self.assertEqual(payload["degradation"], ["d1"])
            self.assertEqual(
                [p["name"] for p in payload["phases"]], ["00 · 配置复核", "01 · 系统打扰", "02 · 代码效率"]
            )
            self.assertEqual(payload["phases"][0]["status"], "RERUN")
            self.assertEqual(payload["phases"][0]["reports"], [{"name": "cfg.html", "href": "00-config/cfg.html"}])
            self.assertEqual(
                payload["phases"][1]["reports"],
                [
                    {"name": "trace.html", "href": "01-trace/card0/trace.html"},
                    {"name": "top.html", "href": "01-trace/top.html"},
                ],
            )
            self.assertEqual(payload["phases"][2]["status"], "not_observable")


class TestRender(unittest.TestCase):
    def test_payload_injected_and_escaped(self):
        payload = {
            "meta": {"run_id": "r", "host": "h", "generated_at": "t"},
            "conclusion": "</script>",
            "attribution": {"converged": True},
            "cards": [],
            "actions": [{"action_id": "A-001"}],
            "degradation": [],
            "phases": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "nested", "index.html")
            path = report.render(payload, out)
            text = _read_text(path)
        self.assertNotIn("__PAYLOAD__", text)
        self.assertIn("A-001", text)
        self.assertIn("<\\/script>", text)

    def test_zero_pct_is_observed_not_missing(self):
        """已观测但打扰/等待均为 0 时，占比条不得显示成「无可观测运行时数据」。"""
        payload = {
            "meta": {"run_id": "r", "host": "h", "generated_at": "t"},
            "conclusion": "未收敛",
            "attribution": {"disturb_pct": 0.0, "wait_pct": 0.0, "residual_pct": 100.0, "converged": False},
            "cards": [],
            "actions": [],
            "degradation": [],
            "phases": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = report.render(payload, os.path.join(tmp, "index.html"))
            text = _read_text(path)
        self.assertIn("const observed = !(a.disturb_pct == null && a.wait_pct == null)", text)
        self.assertNotIn("known <= 0", text)


class TestUpdateManifest(unittest.TestCase):
    def test_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(session)
            workspace.manifest_upsert(session, "00-config", {"status": "RERUN"})
            payload = {
                "meta": {"run_id": "nodeA_20260101-120000", "host": "h", "generated_at": "2026-01-01 12:00:00"},
                "conclusion": "c",
                "attribution": {
                    "disturb_pct": 10.0,
                    "wait_pct": 20.0,
                    "wait_breakdown": {},
                    "efficiency": "not_observable",
                    "residual_pct": 70.0,
                    "converged": False,
                    "basis": "card0",
                },
                "actions": [{"action_id": "A-001"}, {"action_id": "A-002"}],
                "phases": [
                    {"name": "00", "status": "RERUN", "reports": [{"name": "cfg.html", "href": "00-config/cfg.html"}]}
                ],
            }
            path = report._update_manifest(session, payload)
            data = workspace.load_json(path)
        self.assertEqual(data["run_id"], "nodeA_20260101-120000")
        self.assertEqual(data["host"], "h")
        self.assertEqual(data["conclusion"], "c")
        self.assertEqual(data["attribution"]["residual_pct"], 70.0)
        self.assertEqual(data["reports"], {"index": "index.html", "stage": ["00-config/cfg.html"]})
        self.assertEqual(data["actions"], 2)
        self.assertEqual(data["phases"]["00-config"], {"status": "RERUN"})

    def test_custom_out_is_reflected_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "sess")
            os.makedirs(session)
            payload = {
                "meta": {"run_id": "sess", "host": "h", "generated_at": "t"},
                "conclusion": "c",
                "attribution": {},
                "actions": [],
                "phases": [],
            }
            inside = os.path.join(session, "reports", "custom.html")
            path = report._update_manifest(session, payload, inside)
            self.assertEqual(workspace.load_json(path)["reports"]["index"], "reports/custom.html")

            outside = os.path.join(tmp, "other.html")
            path = report._update_manifest(session, payload, outside)
            self.assertEqual(workspace.load_json(path)["reports"]["index"], "other.html")

    def test_repeated_render_does_not_duplicate_stage_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "sess")
            os.makedirs(session)
            payload = {
                "meta": {"run_id": "sess", "host": "h", "generated_at": "t"},
                "conclusion": "c",
                "attribution": {},
                "actions": [],
                "phases": [{"reports": [{"name": "cfg.html", "href": "00-config/cfg.html"}]}],
            }
            report._update_manifest(session, payload)
            path = report._update_manifest(session, payload)
            data = workspace.load_json(path)
        self.assertEqual(data["reports"]["stage"], ["00-config/cfg.html"])


class TestMain(unittest.TestCase):
    def test_writes_index_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(os.path.join(session, "actions"))
            os.makedirs(os.path.join(session, "01-trace"))
            _write_json(
                os.path.join(session, "00-config", "config_triage.json"), {"status": "RERUN", "host": "cfghost"}
            )
            _write_json(os.path.join(session, "01-trace", "phase_verdict_1.json"), {"cards": []})
            _write_json(
                os.path.join(session, "attribution.json"), {"converged": True, "residual_pct": 5.0, "degradation": []}
            )
            _write_json(os.path.join(session, "actions", "action_plan.json"), {"actions": [{"action_id": "A-001"}]})
            buf = StringIO()
            with redirect_stdout(buf):
                code = report.main(["--session", session])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.isfile(os.path.join(session, "index.html")))
            data = workspace.load_json(os.path.join(session, "manifest.json"))
            self.assertIn("已收敛", data["conclusion"])
            self.assertEqual(data["actions"], 1)
            self.assertIn("已收敛", _read_text(os.path.join(session, "index.html")))


if __name__ == "__main__":
    unittest.main()
