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
"""report_check.py 单元测试（report.md R 系规则 + evidence.json E 系规则）。"""

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
        "accuracy",
        "loss-gnorm-diff-analyzer",
        "scripts",
    ),
)

from report_check import check_evidence, check_report


def _ev():
    return {
        "case": "caseX",
        "phenomenon": {"description": "x"},
        "candidates": [
            {
                "candidate_id": "C1",
                "node": "n",
                "rank": 0,
                "basis": "b",
                "supporting_evidence_ids": ["E1"],
                "counterevidence_ids": [],
                "causal_chain": [{"desc": "d", "status": "已证实", "evidence_ids": ["E1"]}],
                "confidence": "高",
                "role": "根因",
                "qualification": {
                    "predicate_rewrite_test": "p",
                    "anchor_verification": "a",
                    "background_unexcluded": "b",
                    "verdict": "v",
                },
            }
        ],
        "evidence_log": [{"id": "E1", "source": "s", "metrics": "m", "usage": "u"}],
        "exclusion_ledger": [{"item": "i", "type": "t", "disposition": "d", "basis": "b"}],
        "open_questions": [],
        "root_cause_judgment": {"status": "已验证根因", "statement": "s", "confidence": "高", "unverified_links": []},
        "verification_methods_used": [],
    }


def _report(n_cands=1):
    lines = [
        "本报告属于 caseX。",
        "# 目标异常现象",
        "x",
        "## 数据与对齐质量",
        "x",
        "## 候选节点总表",
        "共 %d 个候选。" % n_cands,
        "## 根因判断",
        "x",
        "## 候选详析",
        "### 候选 1：n",
        "x",
        "## 后续验证建议",
        "x",
    ]
    return "\n".join(lines)


class TestReportCheck(unittest.TestCase):
    def _run(self, report_text, ev):
        with tempfile.TemporaryDirectory() as td:
            rp, ep = os.path.join(td, "report.md"), os.path.join(td, "e.json")
            with open(rp, "w", encoding="utf-8") as f:
                f.write(report_text)
            with open(ep, "w", encoding="utf-8") as f:
                json.dump(ev, f, ensure_ascii=False)
            issues = []
            n = check_report(rp, issues)
            check_evidence(ep, issues, n)
            return issues

    def test_valid_passes(self):
        self.assertEqual(self._run(_report(), _ev()), [])

    def test_report_rules(self):
        self.assertTrue(any(i.startswith("R4") for i in self._run(_report(2), _ev())))
        self.assertTrue(
            any(i.startswith("R1") for i in self._run(_report().replace("本报告属于 caseX。", "header"), _ev()))
        )
        self.assertTrue(any(i.startswith("R3") for i in self._run(_report() + "\n（待填：xxx）", _ev())))

    def test_evidence_rules(self):
        ev = _ev()
        ev["candidates"][0]["supporting_evidence_ids"] = ["E9"]
        ev["root_cause_judgment"]["status"] = "已验证根因（复核通过）"
        ev["exclusion_ledger"][0]["basis"] = ""
        joined = "\n".join(self._run(_report(), ev))
        self.assertIn("E5b", joined)
        self.assertIn("E7", joined)
        self.assertIn("E10", joined)


if __name__ == "__main__":
    unittest.main()
