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
report.py 单元测试：payload 组装、得分与摘要、模板渲染、findings 落盘。
"""

import json
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

import report


def _item(item_id, status, domain="numa", name="检查项", value="1", benefit="", value_text=""):
    return {
        "id": item_id,
        "status": status,
        "domain": domain,
        "name": name,
        "value": value,
        "threshold": "0",
        "source": "/proc/x",
        "evidence": "证据",
        "benefit": benefit,
        "explain": "这是什么",
        "advice": "建议",
        "value_text": value_text,
    }


RESULTS = [
    _item("A8", "WARN", benefit="low", value=1),
    _item("C1", "WARN", domain="hugepage", benefit="medium", value={"Total": 0}),
    _item("A1", "CRIT", benefit="high"),
    _item("B1", "OK", domain="bind"),
    _item("H1", "FACT", domain="npu"),
]


class TestBuildPayload(unittest.TestCase):
    def setUp(self):
        self.payload = report.build_payload(RESULTS, [], [], {"host": "h"}, [])

    def test_counts(self):
        self.assertEqual(self.payload["counts"], {"CRIT": 1, "WARN": 2, "FACT": 1, "OK": 1})

    def test_domains_use_chinese_names(self):
        self.assertIn("A · NUMA 与拓扑", self.payload["domains"])
        self.assertIn("H · NPU 适配", self.payload["domains"])

    def test_ok_fact_drop_explain_and_advice(self):
        for item in self.payload["domains"]["B · 绑核与 CPU 隔离"]:
            self.assertNotIn("explain", item)
            self.assertNotIn("advice", item)

    def test_issues_keep_explain_and_advice(self):
        warn = [item for item in self.payload["domains"]["A · NUMA 与拓扑"] if item["status"] == "WARN"][0]
        self.assertEqual(warn["explain"], "这是什么")
        self.assertEqual(warn["advice"], "建议")

    def test_domain_sorted_by_status(self):
        ids = [item["id"] for item in self.payload["domains"]["A · NUMA 与拓扑"]]
        self.assertEqual(ids, ["A1", "A8"])

    def test_original_results_not_mutated(self):
        ok = [item for item in RESULTS if item["id"] == "B1"][0]
        self.assertIn("explain", ok)


class TestScoreAndTop(unittest.TestCase):
    def test_score(self):
        self.assertEqual(report._score(RESULTS), 100 - 20 - 10)

    def test_top5_crit_first_then_benefit(self):
        top = report._top5(RESULTS)
        self.assertEqual([item["id"] for item in top], ["A1", "C1", "A8"])

    def test_top5_caps_at_five(self):
        results = [_item("X%d" % i, "WARN") for i in range(8)]
        self.assertEqual(len(report._top5(results)), 5)


class TestTerminalSummary(unittest.TestCase):
    def test_uses_value_text(self):
        results = [_item("C5", "WARN", value={"a": 1}, value_text="/dev/dm-1（4.0 GB）")]
        text = report.terminal_summary(results, [], {"host": "h", "mode": "in-process"})
        self.assertIn("/dev/dm-1（4.0 GB）", text)
        self.assertNotIn('{"a": 1}', text)

    def test_finding_line(self):
        findings = [{"rule": "R001", "problem": "进程完全未绑核", "suggested_cpuset": "0-23"}]
        text = report.terminal_summary([], findings, {"host": "h", "mode": "in-process"})
        self.assertIn("[规则 R001]", text)
        self.assertIn("0-23", text)


class TestRenderAndSave(unittest.TestCase):
    def test_render_replaces_placeholder(self):
        with tempfile.TemporaryDirectory() as tmp:
            template = os.path.join(tmp, "tpl.html")
            with open(template, "w", encoding="utf-8") as handle:
                handle.write("<html>const DATA = __PAYLOAD__;</html>")
            out = os.path.join(tmp, "out", "report.html")
            payload = {"meta": {"host": "h"}}
            report.render_html(payload, template, out)
            with open(out, encoding="utf-8") as handle:
                text = handle.read()
        self.assertNotIn("__PAYLOAD__", text)
        self.assertIn('"host": "h"', text)

    def test_save_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "findings.json")
            report.save_findings([{"rule": "R001"}], [{"action_id": "A-A8"}], path)
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        self.assertEqual(data["findings"][0]["rule"], "R001")
        self.assertEqual(data["actions"][0]["action_id"], "A-A8")


if __name__ == "__main__":
    unittest.main()
