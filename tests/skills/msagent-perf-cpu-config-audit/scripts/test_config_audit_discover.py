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
# MERCHANTABILITY OR FIT FOR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""discover.py 单元测试：进程分类优先级与同 pid 多卡合并。"""

import os
import sys
import unittest
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
        "msagent-perf-cpu-config-audit",
        "scripts",
    ),
)

import discover


class TestClassifyProcess(unittest.TestCase):
    def test_infer_keyword_wins_over_bare_python(self):
        self.assertEqual(discover._classify_process("python3", "vllm serve Qwen", True), "infer_main")

    def test_train_patterns(self):
        self.assertEqual(discover._classify_process("python3", "torchrun train.py", True), "train_main")

    def test_bare_python_defaults_to_train(self):
        self.assertEqual(discover._classify_process("python3", "main.py", True), "train_main")

    def test_no_npu_is_other(self):
        self.assertEqual(discover._classify_process("python3", "vllm serve", False), "other")


class TestDiscoverNpuAnchored(unittest.TestCase):
    def test_same_pid_merges_multiple_npus(self):
        processes = [
            {"pid": 42, "npu": 0},
            {"pid": 42, "npu": 1},
            {"pid": 43, "npu": 2},
        ]
        with (
            patch.object(discover, "_safe_comm", lambda pid: "python3"),
            patch.object(discover.procfs, "read_pid_cmdline", lambda pid: "vllm serve"),
            patch.object(discover.procfs, "read_pid_affinity", lambda pid: "0-7"),
        ):
            targets = discover.discover_npu_anchored(processes)
        self.assertEqual([t["pid"] for t in targets], [42, 43])
        self.assertEqual(targets[0]["npus"], [0, 1])
        self.assertEqual(targets[0]["role"], "infer_main")
        self.assertEqual(targets[1]["npus"], [2])

    def test_invalid_pid_skipped(self):
        with (
            patch.object(discover, "_safe_comm", lambda pid: "python3"),
            patch.object(discover.procfs, "read_pid_cmdline", lambda pid: "main.py"),
            patch.object(discover.procfs, "read_pid_affinity", lambda pid: "0-7"),
        ):
            targets = discover.discover_npu_anchored([{"pid": "bad"}, {"pid": 7}])
        self.assertEqual([t["pid"] for t in targets], [7])


if __name__ == "__main__":
    unittest.main()
