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
classify.py 单元测试：进程角色分类优先级与 targets 补全。
"""

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
        "msagent-perf-cpu-config-audit",
        "scripts",
    ),
)

import classify

try:
    import discover  # 采集层脚本；单 PR 分支（仅判定层）不含此模块
except ImportError:  # pragma: no cover - 分支差异
    discover = None


class TestClassify(unittest.TestCase):
    def test_monitor_whitelist_wins(self):
        self.assertEqual(classify.classify("npu-smi", "npu-smi info", True), "monitor")

    def test_container_runtime(self):
        self.assertEqual(classify.classify("containerd-shim", "", True), "container_runtime")

    def test_logger_and_compiler(self):
        self.assertEqual(classify.classify("fluentd", "", False), "logger")
        self.assertEqual(classify.classify("gcc", "", False), "compiler")

    def test_infer_main(self):
        self.assertEqual(classify.classify("python3", "python3 -m vllm.entrypoints", True), "infer_main")

    def test_train_main_by_cmdline(self):
        self.assertEqual(classify.classify("bash", "torchrun --nproc_per_node=8 train.py", True), "train_main")

    def test_train_main_by_python_comm(self):
        self.assertEqual(classify.classify("python3.11", "train.py", True), "train_main")

    def test_npu_but_unknown_command(self):
        self.assertEqual(classify.classify("vec_add.fatbin", "./vec_add", True), "other")

    def test_no_npu_is_other(self):
        self.assertEqual(classify.classify("python3", "vllm serve", False), "other")


class TestClassifyConsistency(unittest.TestCase):
    """与 discover 的分类口径一致（两侧合入后生效，单 PR 分支自动跳过）。"""

    @unittest.skipIf(discover is None, "需采集层脚本（与采集层 PR 合并后生效）")
    def test_npu_targets_match_discover(self):
        cases = [
            ("python3", "torchrun --nproc_per_node=8 train.py"),
            ("python3.11", "train.py"),
            ("python3", "python3 -m vllm.entrypoints"),
            ("vllm", ""),
            ("vec_add.fatbin", "./vec_add"),
        ]
        for comm, cmdline in cases:
            with self.subTest(comm=comm, cmdline=cmdline):
                self.assertEqual(
                    classify.classify(comm, cmdline, True),
                    discover._classify_process(comm, cmdline, True),
                )

    def test_comm_keyword_is_matched(self):
        """comm 含关键词、cmdline 为空时也应命中（与 discover 的 blob 匹配一致）。"""
        self.assertEqual(classify.classify("vllm", "", True), "infer_main")


class TestClassifyTargets(unittest.TestCase):
    def test_fills_role(self):
        targets = [
            {"comm": "python3", "cmdline": "vllm serve", "npus": [0, 1]},
            {"comm": "sshd", "cmdline": "", "npus": []},
        ]
        classify.classify_targets(targets)
        self.assertEqual([target["role"] for target in targets], ["infer_main", "other"])

    def test_keeps_existing_role(self):
        """discover 已产出的 role 不被覆盖（避免两套分类互相覆盖）。"""
        targets = [{"comm": "python3", "cmdline": "vllm serve", "npus": [0], "role": "train_main"}]
        classify.classify_targets(targets)
        self.assertEqual(targets[0]["role"], "train_main")


if __name__ == "__main__":
    unittest.main()
