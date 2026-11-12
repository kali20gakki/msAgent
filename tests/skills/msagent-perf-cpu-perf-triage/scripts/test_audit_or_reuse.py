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
audit_or_reuse.py 单元测试
"""

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
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
        "msagent-perf-cpu-perf-triage",
        "scripts",
    ),
)

import audit_or_reuse


def _run_main(args):
    buf = StringIO()
    with redirect_stdout(buf):
        code = audit_or_reuse.main(args)
    return code, buf.getvalue()


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fake_audit(root, body):
    script_dir = os.path.join(root, "scripts")
    os.makedirs(script_dir, exist_ok=True)
    with open(os.path.join(script_dir, "audit.py"), "w", encoding="utf-8") as fh:
        fh.write(body)


class TestSummarize(unittest.TestCase):
    def test_high_risk_and_strong_finding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "findings.json")
            _write_json(
                path,
                {
                    "findings": [
                        {"finding_id": "R001", "problem": "进程未绑核", "confidence": 0.9, "suggested_cpuset": "0-7"}
                    ],
                    "actions": [{"action_id": "A-1", "risk": "high"}],
                },
            )
            result = audit_or_reuse._summarize(path)
            self.assertEqual(result["findings_count"], 1)
            self.assertEqual(result["action_count"], 1)
            self.assertEqual(result["impact_level"], "高")
            self.assertTrue(result["config_primary_candidate"])
            self.assertEqual(result["top_findings"][0]["finding_id"], "R001")

    def test_low_confidence_medium_risk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "findings.json")
            _write_json(
                path,
                {
                    "findings": [{"finding_id": "R002", "confidence": 0.3}],
                    "actions": [{"risk": "medium"}],
                },
            )
            result = audit_or_reuse._summarize(path)
            self.assertEqual(result["impact_level"], "中")
            self.assertFalse(result["config_primary_candidate"])

    def test_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "findings.json")
            _write_json(path, {})
            result = audit_or_reuse._summarize(path)
            self.assertEqual(result["findings_count"], 0)
            self.assertEqual(result["action_count"], 0)
            self.assertEqual(result["impact_level"], "无")
            self.assertFalse(result["config_primary_candidate"])

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = audit_or_reuse._summarize(os.path.join(tmp, "none.json"))
            self.assertEqual(result["findings_count"], 0)
            self.assertFalse(result["config_primary_candidate"])

    def test_top_findings_limited_to_five(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "findings.json")
            _write_json(path, {"findings": [{"finding_id": "R%d" % i} for i in range(7)]})
            result = audit_or_reuse._summarize(path)
            self.assertEqual(result["findings_count"], 7)
            self.assertEqual(len(result["top_findings"]), 5)


class TestResolveAuditDir(unittest.TestCase):
    def test_default_sibling_skill(self):
        path = audit_or_reuse._resolve_audit_dir("")
        self.assertTrue(path)
        self.assertTrue(os.path.isfile(os.path.join(path, "audit.py")))
        self.assertEqual(os.path.basename(path), "scripts")

    def test_explicit_skill_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fake_audit(tmp, "raise SystemExit(0)\n")
            self.assertEqual(audit_or_reuse._resolve_audit_dir(tmp), os.path.join(tmp, "scripts"))

    def test_explicit_scripts_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            _fake_audit(tmp, "raise SystemExit(0)\n")
            self.assertEqual(
                audit_or_reuse._resolve_audit_dir(os.path.join(tmp, "scripts")), os.path.join(tmp, "scripts")
            )

    def test_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty")
            os.makedirs(empty)
            with patch.object(audit_or_reuse, "SKILLS_DIR", empty), patch.dict(os.environ, {"HOSTCPU_SKILLS_DIR": ""}):
                self.assertEqual(audit_or_reuse._resolve_audit_dir(""), "")


class TestMain(unittest.TestCase):
    def test_reused_fresh_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            findings = os.path.join(session, "00-config", "findings.json")
            _write_json(
                findings,
                {
                    "findings": [{"finding_id": "R001", "confidence": 0.9, "problem": "p"}],
                    "actions": [{"action_id": "A-1", "risk": "high"}],
                },
            )
            code, out = _run_main(["--session", session])
            self.assertEqual(code, 0)
            result = _read_json(os.path.join(session, "00-config", "config_triage.json"))
            self.assertEqual(result["status"], "REUSED")
            self.assertEqual(result["source"], findings)
            self.assertEqual(result["impact_level"], "高")
            self.assertTrue(result["config_primary_candidate"])
            self.assertEqual(result["degradation"], [])
            self.assertIn("REUSED", out)
            manifest = _read_json(os.path.join(session, "manifest.json"))
            self.assertEqual(manifest["phases"]["00-config"]["status"], "REUSED")

    def test_expired_findings_with_skip_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            findings = os.path.join(session, "00-config", "findings.json")
            _write_json(findings, {"findings": [{"finding_id": "R001", "confidence": 0.9}], "actions": []})
            code, _ = _run_main(["--session", session, "--max-age-min", "-1", "--skip-audit"])
            self.assertEqual(code, 0)
            result = _read_json(os.path.join(session, "00-config", "config_triage.json"))
            self.assertEqual(result["status"], "SKIPPED")
            # 过期 findings 作为旁证保留，不静默丢弃
            self.assertEqual(result["source"], findings)
            self.assertEqual(result["findings_count"], 1)
            self.assertTrue(result["config_primary_candidate"])
            self.assertTrue(any("已过期" in item for item in result["degradation"]))
            self.assertTrue(any("仅作旁证" in item for item in result["degradation"]))

    def test_skipped_without_findings(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(session)
            code, _ = _run_main(["--session", session, "--skip-audit"])
            self.assertEqual(code, 0)
            result = _read_json(os.path.join(session, "00-config", "config_triage.json"))
            self.assertEqual(result["status"], "SKIPPED")
            self.assertNotIn("findings_count", result)
            self.assertNotIn("config_primary_candidate", result)

    def test_unavailable_when_skill_missing(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as empty:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(session)
            with patch.object(audit_or_reuse, "SKILLS_DIR", empty), patch.dict(os.environ, {"HOSTCPU_SKILLS_DIR": ""}):
                code, _ = _run_main(["--session", session])
            self.assertEqual(code, 1)
            result = _read_json(os.path.join(session, "00-config", "config_triage.json"))
            self.assertEqual(result["status"], "UNAVAILABLE")
            self.assertTrue(any("未找到 config-audit" in item for item in result["degradation"]))

    def test_rerun_when_audit_succeeds(self):
        body = (
            "import json, os, sys\n"
            "argv = sys.argv[1:]\n"
            "out = argv[argv.index('--output') + 1]\n"
            "os.makedirs(out, exist_ok=True)\n"
            "with open(os.path.join(out, 'findings.json'), 'w', encoding='utf-8') as fh:\n"
            "    json.dump({'findings': [{'finding_id': 'R002', 'confidence': 0.8}],"
            " 'actions': []}, fh)\n"
            "print('{\"ok\": true}')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(session)
            fake = os.path.join(tmp, "fake-audit")
            _fake_audit(fake, body)
            code, _ = _run_main(["--session", session, "--config-audit-dir", fake])
            self.assertEqual(code, 0)
            result = _read_json(os.path.join(session, "00-config", "config_triage.json"))
            self.assertEqual(result["status"], "RERUN")
            self.assertEqual(result["source"], os.path.join(session, "00-config", "findings.json"))
            self.assertTrue(result["config_primary_candidate"])

    def test_failed_when_audit_nonzero(self):
        body = "import sys\nsys.stderr.write('boom')\nraise SystemExit(2)\n"
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(session)
            fake = os.path.join(tmp, "fake-audit")
            _fake_audit(fake, body)
            code, _ = _run_main(["--session", session, "--config-audit-dir", fake])
            self.assertEqual(code, 1)
            result = _read_json(os.path.join(session, "00-config", "config_triage.json"))
            self.assertEqual(result["status"], "FAILED")
            self.assertTrue(any("exit=2" in item and "boom" in item for item in result["degradation"]))

    def test_failed_when_audit_produces_no_findings(self):
        body = "print('{\"ok\": true}')\n"
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(session)
            fake = os.path.join(tmp, "fake-audit")
            _fake_audit(fake, body)
            code, _ = _run_main(["--session", session, "--config-audit-dir", fake])
            self.assertEqual(code, 1)
            result = _read_json(os.path.join(session, "00-config", "config_triage.json"))
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["source"], "")
            self.assertFalse(result.get("config_primary_candidate", False))
            self.assertTrue(any("未产出有效 findings.json" in item for item in result["degradation"]))


if __name__ == "__main__":
    unittest.main()
