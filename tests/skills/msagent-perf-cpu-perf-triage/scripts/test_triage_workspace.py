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
workspace.py 单元测试
"""

import importlib.util
import os
import sys
import tempfile
import unittest
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
    """按文件路径以唯一模块名加载，避免多 skill 同名 workspace/report 串味。"""
    spec = importlib.util.spec_from_file_location("perf_triage_" + name, os.path.join(_SCRIPTS, name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


workspace = _load_skill_module("workspace")


class TestSafeName(unittest.TestCase):
    def test_sanitize(self):
        self.assertEqual(workspace.safe_name("a/b c"), "a_b_c")
        self.assertEqual(workspace.safe_name("node-1.2"), "node-1.2")
        self.assertEqual(workspace.safe_name("__x__"), "x")

    def test_empty_to_unknown(self):
        self.assertEqual(workspace.safe_name(""), "unknown")
        self.assertEqual(workspace.safe_name(None), "unknown")
        self.assertEqual(workspace.safe_name("..."), "unknown")

    def test_truncate(self):
        self.assertEqual(len(workspace.safe_name("a" * 100)), 64)

    def test_path_traversal_is_neutralized(self):
        self.assertEqual(workspace.safe_name("../../etc"), "etc")
        self.assertEqual(workspace.safe_name("..\\..\\windows"), "windows")


class TestWorkspaceRoot(unittest.TestCase):
    def test_explicit_root_wins(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as env_dir:
            with patch.dict(os.environ, {"HOSTCPU_WORKSPACE": env_dir}):
                self.assertEqual(workspace.workspace_root(tmp), os.path.abspath(tmp))

    def test_env_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HOSTCPU_WORKSPACE": tmp}):
                self.assertEqual(workspace.workspace_root(), os.path.abspath(tmp))

    def test_cwd_fallback(self):
        with patch.dict(os.environ, {"HOSTCPU_WORKSPACE": ""}):
            self.assertEqual(workspace.workspace_root(), os.path.abspath(os.getcwd()))


class TestNewSession(unittest.TestCase):
    def test_create_phase_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = workspace.new_session(host="node/a", timestamp="20260101-120000", root=tmp)
            self.assertEqual(session["run_id"], "node_a_20260101-120000")
            self.assertEqual(session["host"], "node_a")
            self.assertEqual(session["timestamp"], "20260101-120000")
            expected = os.path.join(tmp, "workspaces", "node_a", "20260101-120000")
            self.assertEqual(session["root"], expected)
            for phase in workspace.PHASES:
                self.assertTrue(os.path.isdir(os.path.join(expected, phase)))

    def test_dry_run_does_not_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = workspace.new_session(host="node", timestamp="20260101-120000", root=tmp, create=False)
            self.assertFalse(os.path.exists(session["root"]))

    def test_timestamp_sanitized_before_path_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = workspace.new_session(host="node", timestamp="../escape", root=tmp, create=False)
            self.assertEqual(session["timestamp"], "escape")
            self.assertEqual(session["root"], os.path.join(tmp, "workspaces", "node", "escape"))

    def test_host_defaults_to_hostname(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(workspace.socket, "gethostname", return_value="myhost"):
                session = workspace.new_session(timestamp="20260101-120000", root=tmp, create=False)
            self.assertEqual(session["host"], "myhost")
            self.assertEqual(session["run_id"], "myhost_20260101-120000")


class TestPhaseDir(unittest.TestCase):
    def test_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = workspace.phase_dir(tmp, "01-trace")
            self.assertEqual(path, os.path.join(tmp, "01-trace"))
            self.assertTrue(os.path.isdir(path))

    def test_no_create(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = workspace.phase_dir(tmp, "actions", create=False)
            self.assertFalse(os.path.exists(path))


class TestJsonHelpers(unittest.TestCase):
    def test_load_json_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(workspace.load_json(os.path.join(tmp, "none.json"), default={"a": 1}), {"a": 1})
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("{not json")
            self.assertIsNone(workspace.load_json(bad))

    def test_write_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested", "data.json")
            workspace.write_json(path, {"中文": 1})
            self.assertEqual(workspace.load_json(path), {"中文": 1})
            leftovers = [name for name in os.listdir(os.path.dirname(path)) if name.startswith(".tmp-")]
            self.assertEqual(leftovers, [])


class TestManifestUpsert(unittest.TestCase):
    def test_creates_and_merges(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "nodeA_20260101-120000")
            os.makedirs(session)
            path = workspace.manifest_upsert(session, "00-config", {"status": "RERUN"})
            self.assertEqual(path, os.path.join(session, "manifest.json"))
            data = workspace.load_json(path)
            self.assertEqual(data["run_id"], "nodeA_20260101-120000")
            self.assertEqual(data["phases"]["00-config"], {"status": "RERUN"})
            self.assertIn("updated_at", data)

            workspace.manifest_upsert(session, "01-trace", {"status": "OBSERVED"})
            data = workspace.load_json(path)
            self.assertEqual(data["phases"]["00-config"], {"status": "RERUN"})
            self.assertEqual(data["phases"]["01-trace"], {"status": "OBSERVED"})

    def test_keep_existing_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "sess")
            os.makedirs(session)
            workspace.write_json(os.path.join(session, "manifest.json"), {"run_id": "custom"})
            workspace.manifest_upsert(session, "actions", {"count": 1})
            data = workspace.load_json(os.path.join(session, "manifest.json"))
            self.assertEqual(data["run_id"], "custom")
            self.assertEqual(data["phases"]["actions"], {"count": 1})

    def test_corrupt_manifest_backed_up_before_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "sess")
            os.makedirs(session)
            manifest_path = os.path.join(session, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as fh:
                fh.write("{broken json")
            workspace.manifest_upsert(session, "00-config", {"status": "RERUN"})
            data = workspace.load_json(manifest_path)
            self.assertEqual(data["phases"]["00-config"], {"status": "RERUN"})
            backups = [name for name in os.listdir(session) if ".corrupt-" in name]
            self.assertEqual(len(backups), 1)
            with open(os.path.join(session, backups[0]), "r", encoding="utf-8") as fh:
                self.assertIn("broken json", fh.read())

    def test_non_dict_manifest_treated_as_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            session = os.path.join(tmp, "sess")
            os.makedirs(session)
            manifest_path = os.path.join(session, "manifest.json")
            workspace.write_json(manifest_path, ["not", "a", "dict"])
            workspace.manifest_upsert(session, "01-trace", {"status": "OBSERVED"})
            data = workspace.load_json(manifest_path)
            self.assertEqual(data["phases"]["01-trace"], {"status": "OBSERVED"})
            self.assertTrue(any(".corrupt-" in name for name in os.listdir(session)))


if __name__ == "__main__":
    unittest.main()
