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

"""tracefs_ctl.py 单元测试（仅 Windows 可执行部分）。

capture 依赖真实 tracefs 与 Linux 阻塞读语义，不在本文件覆盖。
"""

import os
import stat
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

from convert import tracefs_ctl
from convert.tracefs_ctl import TracefsError, TracefsInstance


class _InstanceCase(unittest.TestCase):
    """在临时目录里伪造一个 tracefs instance 目录树。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._restore_perms)
        self.base = os.path.join(self.tmp.name, "tracefs")
        os.makedirs(self.base)
        self.inst = TracefsInstance(self.base, "x")

    def _restore_perms(self):
        """Windows 下只读文件会阻塞临时目录清理，统一恢复可写。"""
        for root, _dirs, files in os.walk(self.tmp.name):
            for name in files:
                try:
                    os.chmod(os.path.join(root, name), stat.S_IWRITE | stat.S_IREAD)
                except OSError:
                    pass

    def touch(self, relpath, text=""):
        """在 base 下创建文件（自动建父目录），返回绝对路径。"""
        path = os.path.join(self.base, relpath.replace("/", os.sep))
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        return path

    def read(self, path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()


class TestCpuExpr(unittest.TestCase):
    """_cpu_expr：把核集压成 common_cpu 表达式。"""

    def test_contiguous_range(self):
        self.assertEqual(TracefsInstance._cpu_expr([0, 1, 2, 3]), "(common_cpu >= 0 && common_cpu <= 3)")

    def test_single_cpu(self):
        self.assertEqual(TracefsInstance._cpu_expr([3]), "(common_cpu == 3)")

    def test_discrete_cpus(self):
        self.assertEqual(
            TracefsInstance._cpu_expr([0, 2, 4]), "(common_cpu == 0 || common_cpu == 2 || common_cpu == 4)"
        )

    def test_empty_set(self):
        self.assertEqual(TracefsInstance._cpu_expr([]), "")

    def test_mixed_spans_dedup_and_sort(self):
        self.assertEqual(
            TracefsInstance._cpu_expr([5, 0, 1, 2, 2, 7, 8]),
            "(common_cpu >= 0 && common_cpu <= 2 || common_cpu == 5 || common_cpu >= 7 && common_cpu <= 8)",
        )


class TestSetFilters(_InstanceCase):
    """set_filters：cpu 与 comm 条件合并成一条表达式覆盖写。"""

    CPU_EXPR = "(common_cpu >= 0 && common_cpu <= 2 || common_cpu == 5)"

    def setUp(self):
        super().setUp()
        self.enable_paths = {}
        for group, name in (("sched", "sched_switch"), ("sched", "sched_wakeup"), ("irq", "irq_handler_entry")):
            path = self.touch("instances/x/events/%s/%s/enable" % (group, name))
            self.enable_paths["%s:%s" % (group, name)] = path
        self.inst.enabled = list(self.enable_paths.items())

    def _filter_path(self, event):
        return self.enable_paths[event][: -len("enable")] + "filter"

    def test_cpu_and_comm_merged_into_single_write(self):
        applied = self.inst.set_filters(cpus=[0, 1, 2, 5], comms=["python", "worker"])
        expected_sched = '%s && (comm == "python" || comm == "worker")' % self.CPU_EXPR
        self.assertEqual(applied["sched:sched_switch"], expected_sched)
        self.assertEqual(applied["sched:sched_wakeup"], expected_sched)
        # comm 条件只作用于 sched_switch / sched_wakeup
        self.assertEqual(applied["irq:irq_handler_entry"], self.CPU_EXPR)
        for event in ("sched:sched_switch", "sched:sched_wakeup"):
            content = self.read(self._filter_path(event))
            self.assertEqual(content, expected_sched)
            self.assertEqual(len(content.strip().splitlines()), 1)
        self.assertEqual(self.read(self._filter_path("irq:irq_handler_entry")), self.CPU_EXPR)

    def test_comm_only(self):
        applied = self.inst.set_filters(comms=["python"])
        self.assertEqual(
            applied,
            {
                "sched:sched_switch": '(comm == "python")',
                "sched:sched_wakeup": '(comm == "python")',
            },
        )

    def test_cpus_only(self):
        applied = self.inst.set_filters(cpus=[2])
        self.assertEqual(
            applied,
            {
                "sched:sched_switch": "(common_cpu == 2)",
                "sched:sched_wakeup": "(common_cpu == 2)",
                "irq:irq_handler_entry": "(common_cpu == 2)",
            },
        )

    def test_no_conditions_writes_nothing(self):
        self.assertEqual(self.inst.set_filters(), {})

    def test_unwritable_filter_yields_empty_value(self):
        """单个 filter 写失败时降级为空串，其余事件照常。

        不用 chmod 构造只读：root 无视权限位，用例会假失败且覆盖不到降级分支；
        这里 mock 掉 _write，让目标 filter 抛 PermissionError。
        """
        locked = os.path.abspath(self._filter_path("sched:sched_switch"))
        real_write = tracefs_ctl._write

        def _write(path, value):
            if os.path.abspath(path) == locked:
                raise PermissionError("filter 只读")
            real_write(path, value)

        with mock.patch.object(tracefs_ctl, "_write", new=_write):
            applied = self.inst.set_filters(cpus=[2])
        self.assertEqual(applied["sched:sched_switch"], "")
        # 其余事件不受影响
        self.assertEqual(applied["irq:irq_handler_entry"], "(common_cpu == 2)")
        self.assertEqual(self.read(self._filter_path("irq:irq_handler_entry")), "(common_cpu == 2)")


class TestResolveEvent(_InstanceCase):
    """enable_events / _resolve_event：裸名解析与错误分支。"""

    def setUp(self):
        super().setUp()
        self.touch("available_events", "sched:sched_switch\nsched:sched_wakeup\nirq:irq_handler_entry\n")

    def test_bare_name_resolved_through_available_events(self):
        event, path = self.inst._resolve_event("sched_switch")
        self.assertEqual(event, "sched:sched_switch")
        self.assertEqual(path, os.path.join(self.inst.path, "events", "sched", "sched_switch", "enable"))

    def test_full_group_name_hits_directly(self):
        event, path = self.inst._resolve_event("irq/irq_handler_entry")
        self.assertEqual(event, "irq/irq_handler_entry")
        self.assertEqual(path, os.path.join(self.inst.path, "events", "irq", "irq_handler_entry", "enable"))

    def test_ambiguous_name_raises(self):
        self.touch("available_events", "sched:sched_switch\nmisc:sched_switch\n")
        with self.assertRaises(TracefsError) as ctx:
            self.inst._resolve_event("sched_switch")
        self.assertIn("歧义", str(ctx.exception))

    def test_missing_event_raises(self):
        with self.assertRaises(TracefsError) as ctx:
            self.inst._resolve_event("no_such_event")
        self.assertIn("不存在", str(ctx.exception))

    def test_path_traversal_rejected(self):
        for event in ("../../etc/passwd", "/sched/sched_switch", "sched/..", "sched/", "sched/sched switch"):
            with self.assertRaises(TracefsError) as ctx:
                self.inst._resolve_event(event)
            self.assertIn("非法事件名", str(ctx.exception))

    def test_enable_events_writes_flag_and_tracks(self):
        for group, name in (("sched", "sched_switch"), ("irq", "irq_handler_entry")):
            self.touch("instances/x/events/%s/%s/enable" % (group, name))
        self.inst.enable_events(["sched_switch", "irq/irq_handler_entry"])
        self.assertEqual(self.read(os.path.join(self.inst.path, "events", "sched", "sched_switch", "enable")), "1")
        self.assertEqual(self.read(os.path.join(self.inst.path, "events", "irq", "irq_handler_entry", "enable")), "1")
        self.assertEqual([item[0] for item in self.inst.enabled], ["sched:sched_switch", "irq/irq_handler_entry"])


class TestTrimCpus(_InstanceCase):
    """trim_cpus：per_cpu 缺失时静默返回 0。"""

    def test_missing_per_cpu_returns_zero(self):
        self.assertEqual(self.inst.trim_cpus([0, 1]), 0)

    def test_trims_unwanted_cpus(self):
        paths = {}
        for cpu in (0, 1, 2):
            paths[cpu] = self.touch("instances/x/per_cpu/cpu%d/enable" % cpu)
        trimmed = self.inst.trim_cpus([0])
        self.assertEqual(trimmed, 2)
        self.assertEqual(self.read(paths[0]), "")
        self.assertEqual(self.read(paths[1]), "0")
        self.assertEqual(self.read(paths[2]), "0")


class TestReadNonblocking(unittest.TestCase):
    """_read_nonblocking：非法 fd 兜底为空串。"""

    def test_invalid_fd_returns_empty_bytes(self):
        self.assertEqual(tracefs_ctl._read_nonblocking(-1), b"")


class TestEnsure(_InstanceCase):
    """ensure：同名 instance 不能覆盖删除，instances 目录缺失要报错。"""

    def test_existing_instance_raises_instead_of_removing(self):
        os.makedirs(self.inst.path)
        with self.assertRaises(TracefsError) as ctx:
            self.inst.ensure()
        self.assertIn("已存在", str(ctx.exception))
        self.assertTrue(os.path.isdir(self.inst.path))

    def test_missing_instances_dir_raises(self):
        with self.assertRaises(TracefsError):
            self.inst.ensure()


class TestCaptureWindowCleanup(unittest.TestCase):
    """capture_window 的异常路径必须回收 instance，不留目录/事件开关。"""

    def test_cleanup_runs_when_ensure_fails(self):
        cleaned = []
        with tempfile.TemporaryDirectory() as base:
            with (
                mock.patch.object(TracefsInstance, "ensure", side_effect=TracefsError("clock 写入失败")),
                mock.patch.object(
                    TracefsInstance,
                    "cleanup",
                    autospec=True,
                    side_effect=lambda inst: cleaned.append(inst.name),
                ),
            ):
                with self.assertRaises(TracefsError):
                    tracefs_ctl.capture_window(base, [0], ["sched_switch"], 0.01, os.path.join(base, "trace.txt"))
        self.assertEqual(len(cleaned), 1)

    def test_instance_name_is_unique_per_capture(self):
        """实例名不再按毫秒取模（约 1000s 回绕），两次采集必须不同名。"""
        names = []

        def _ensure(inst):
            names.append(inst.name)
            raise TracefsError("stop")

        with tempfile.TemporaryDirectory() as base:
            with (
                mock.patch.object(TracefsInstance, "ensure", autospec=True, side_effect=_ensure),
                mock.patch.object(TracefsInstance, "cleanup", autospec=True),
            ):
                for _ in range(2):
                    with self.assertRaises(TracefsError):
                        tracefs_ctl.capture_window(base, [0], ["sched_switch"], 0.01, os.path.join(base, "trace.txt"))
        self.assertEqual(len(names), 2)
        self.assertNotEqual(names[0], names[1])
        for name in names:
            self.assertRegex(name, r"^hostcpu_\d{13}_[0-9a-f]{8}$")


if __name__ == "__main__":
    unittest.main()
