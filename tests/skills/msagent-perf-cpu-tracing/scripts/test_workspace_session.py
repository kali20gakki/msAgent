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

"""workspace.py：工作空间三级优先级、会话目录结构与主机名清洗。"""

import os
import sys
import tempfile
import unittest
from unittest import mock

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

import workspace


class TestWorkspaceRoot(unittest.TestCase):
    def test_explicit_root_wins(self):
        explicit = tempfile.mkdtemp(prefix="hostcpu-explicit-")
        with mock.patch.dict(os.environ, {"HOSTCPU_WORKSPACE": tempfile.mkdtemp(prefix="hostcpu-env-")}):
            self.assertEqual(workspace.workspace_root(explicit), os.path.abspath(explicit))

    def test_env_var_used_when_no_root(self):
        env_root = tempfile.mkdtemp(prefix="hostcpu-env-")
        with mock.patch.dict(os.environ, {"HOSTCPU_WORKSPACE": env_root}):
            self.assertEqual(workspace.workspace_root(), os.path.abspath(env_root))

    def test_cwd_fallback(self):
        env = {key: value for key, value in os.environ.items() if key != "HOSTCPU_WORKSPACE"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(workspace.workspace_root(), os.path.abspath(os.getcwd()))

    def test_root_is_abspath(self):
        relative = os.path.join("some", "relative", "root")
        self.assertEqual(workspace.workspace_root(relative), os.path.abspath(relative))


class TestSessionDir(unittest.TestCase):
    def test_layout_and_creation(self):
        root = tempfile.mkdtemp(prefix="hostcpu-ws-")
        path = workspace.session_dir(host="node-1", timestamp="20260101-000000", root=root)
        self.assertTrue(os.path.isdir(path))
        self.assertEqual(path, os.path.join(root, "workspaces", "node-1", "20260101-000000"))

    def test_nested_parents_created(self):
        root = tempfile.mkdtemp(prefix="hostcpu-ws-")
        path = workspace.session_dir(host="node-1", timestamp="20260101-000001", root=root)
        self.assertTrue(os.path.isdir(os.path.join(root, "workspaces")))
        self.assertTrue(os.path.isdir(os.path.join(root, "workspaces", "node-1")))
        self.assertTrue(os.path.isdir(path))

    def test_defaults_use_hostname_and_strftime(self):
        """未显式传入 host/timestamp 时，分别取主机名与本地时间格式。"""
        root = tempfile.mkdtemp(prefix="hostcpu-ws-")
        with mock.patch.object(workspace.socket, "gethostname", return_value="node-9.example.com"):
            with mock.patch.object(workspace.time, "strftime", return_value="20260102-030405"):
                path = workspace.session_dir(root=root)
        self.assertEqual(
            path,
            os.path.join(root, "workspaces", "node-9.example.com", "20260102-030405"),
        )


class TestSafeName(unittest.TestCase):
    def test_strips_path_segments(self):
        self.assertEqual(workspace.safe_name("host/../etc"), "host_.._etc")
        self.assertEqual(workspace.safe_name("a b"), "a_b")

    def test_empty_falls_back_and_truncates(self):
        self.assertEqual(workspace.safe_name(None), "unknown")
        self.assertEqual(len(workspace.safe_name("x" * 200)), 64)

    def test_session_dir_sanitizes_inputs(self):
        root = tempfile.mkdtemp(prefix="hostcpu-ws-")
        path = workspace.session_dir(host="../etc", timestamp="a/../../b", root=root)
        self.assertEqual(path, os.path.join(root, "workspaces", "etc", "a_.._.._b"))


class TestHostName(unittest.TestCase):
    def test_strips_domain_suffix(self):
        with mock.patch.object(workspace.socket, "gethostname", return_value="node-1.cluster.local"):
            self.assertEqual(workspace.host_name(), "node-1")

    def test_plain_hostname_unchanged(self):
        with mock.patch.object(workspace.socket, "gethostname", return_value="node-1"):
            self.assertEqual(workspace.host_name(), "node-1")


if __name__ == "__main__":
    unittest.main()
