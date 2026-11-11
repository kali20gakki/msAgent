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
audit.py 单元测试：快照阶段的 online 核集只读一次。

audit 依赖采集层脚本（topology/static_collect/discover 等），单 PR 分支
（仅判定层）导入会失败，此类用例自动跳过；两侧合入后（master）生效。
"""

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

try:
    import audit
except ImportError:  # pragma: no cover - 分支差异
    audit = None


@unittest.skipIf(audit is None, "需采集层脚本（与采集层 PR 合并后生效）")
class TestTakeSnapshot(unittest.TestCase):
    def test_online_cpus_read_once(self):
        """格式化与计数必须来自同一次核集读取。"""
        calls = []

        def fake_online():
            calls.append(1)
            return [0, 1, 2, 3]

        with (
            patch.object(audit.collect_mod, "collect", lambda pid=0: {"facts": []}),
            patch.object(audit.topology_mod, "online_cpus", fake_online),
            patch.object(audit.topology_mod, "format_cpu_list", lambda cpus: "0-3"),
        ):
            snapshot, online, num = audit._take_snapshot({"in_container": False}, [{"pid": 1}], 0)

        self.assertEqual(len(calls), 1)
        self.assertEqual(online, "0-3")
        self.assertEqual(num, 4)
        self.assertEqual(snapshot["preflight"], {"in_container": False})
        self.assertEqual(snapshot["targets"], [{"pid": 1}])


if __name__ == "__main__":
    unittest.main()
