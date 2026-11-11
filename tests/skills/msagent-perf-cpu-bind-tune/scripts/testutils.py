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

"""msagent-perf-cpu-bind-tune 测试公共夹具。

提供：脚本目录导入、A2 实机 npu-smi 解析、A2/A3 合成拓扑、mock /proc 上下文。
只构造确定性的内存数据，不依赖真机、不写 skill 目录。
"""

import contextlib
import os
import sys
import tempfile
from unittest import mock

# 不在导入期给标准库 os 打补丁（进程级副作用会影响其它测试）：
# 无 getuid 的平台（Windows 本地跑测）只取一个常量占位，需要 uid 语义的用例
# 用下面的 mock_uid() 在用例内临时注入。
OWN_UID = os.getuid() if hasattr(os, "getuid") else 1000
# 与当前用户不同的 uid：root 环境下取 1000，非 root 取 0，避免断言依赖运行用户
OTHER_UID = 1000 if OWN_UID == 0 else 0


@contextlib.contextmanager
def mock_uid(uid=None):
    """用例内临时替换 os.getuid；退出时自动还原（create=True 兼容 Windows）。"""
    with mock.patch.object(os, "getuid", lambda: OWN_UID if uid is None else uid, create=True):
        yield


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上找到仓库根（同时含 skills/ 与 pyproject.toml），避免硬编码层级
_TEST_ROOT = _THIS_DIR
while os.path.dirname(_TEST_ROOT) != _TEST_ROOT and not (
    os.path.isdir(os.path.join(_TEST_ROOT, "skills")) and os.path.isfile(os.path.join(_TEST_ROOT, "pyproject.toml"))
):
    _TEST_ROOT = os.path.dirname(_TEST_ROOT)
SKILL_SCRIPTS = os.path.join(_TEST_ROOT, "skills", "host-cpu", "msagent-perf-cpu-bind-tune", "scripts")
if SKILL_SCRIPTS not in sys.path:
    sys.path.insert(0, SKILL_SCRIPTS)

import npu_query  # noqa: E402
import procfs  # noqa: E402
import topology  # noqa: E402

FIXTURES = os.path.join(_THIS_DIR, "fixtures")

DIST = [
    [10, 11, 24, 25, 24, 25, 24, 25],
    [11, 10, 25, 32, 25, 32, 25, 32],
    [24, 25, 10, 11, 24, 25, 24, 25],
    [25, 32, 11, 10, 25, 32, 25, 32],
    [24, 25, 24, 25, 10, 11, 24, 25],
    [25, 32, 25, 32, 11, 10, 25, 32],
    [24, 25, 24, 25, 24, 25, 10, 11],
    [25, 32, 25, 32, 25, 32, 11, 10],
]

# A2 实机 VLLM Worker 线程画像（comm, 数量）
THREAD_COUNTS = [
    ("VLLM::Worker", 34),
    ("AOE_RTKB", 16),
    ("hccl_watchdog_t", 8),
    ("CaffeTaskThread", 5),
    ("python", 3),
    ("ZMQbg/Reaper", 2),
    ("ZMQbg/IO/0", 2),
    ("TraceClientRecv", 2),
    ("release_thread", 2),
    ("PlogReportRecv", 2),
    ("MONITOR_0", 2),
    ("acl_thread", 2),
    ("WatchdogMonitor", 1),
    ("RT_RECYCLE_1", 1),
    ("RT_RECYCLE_0", 1),
    ("REPORT_RAS", 1),
    ("pt_autograd_2", 1),
    ("pt_autograd_1", 1),
    ("pt_autograd_0", 1),
    ("PlogFlush", 1),
    ("AtraceMonitor", 1),
    ("adx_data_dump_t", 1),
]

PID_INFO = {
    293354: {"ppid": 291298, "affinity": "146-165", "last_cpu": 150},
    293350: {"ppid": 291299, "affinity": "170-189", "last_cpu": 180},
    293351: {"ppid": 291300, "affinity": "50-69", "last_cpu": 60},
}


def load_a2_npu_raw():
    """读取 A2 实机 npu-smi 输出并返回 (info 解析, topo 解析)。"""
    with open(os.path.join(FIXTURES, "npu_raw.txt"), "r", encoding="utf-8") as f:
        raw = f.read()
    info_text, topo_text = raw.split("===TOPO===", 1)
    return npu_query.parse_info(info_text), npu_query.parse_topo(topo_text)


def a2_topo(info=None, topo_info=None):
    """A2 形态：8 node × 24 核，NPU→CPU 亲和来自 topo（PCIe/HCCS 直连）。"""
    info = info or load_a2_npu_raw()[0]
    topo_info = topo_info or load_a2_npu_raw()[1]
    nodes, cpus, l3 = [], [], []
    for n in range(8):
        start = n * 24
        cs = list(range(start, start + 24))
        nodes.append(
            {"node": n, "cpus": cs, "cpulist": "%d-%d" % (start, start + 23), "distance": DIST[n], "mem_gb": 251.8}
        )
        for c in cs:
            cpus.append(
                {
                    "cpu": c,
                    "node": n,
                    "core_id": c - start,
                    "package_id": n // 2,
                    "thread_siblings": [c],
                    "l3_shared": "%d-%d" % (start, start + 23),
                }
            )
        l3.append({"id": n, "shared_cpu_list": "%d-%d" % (start, start + 23), "cpus": cs, "node": n})

    devices = {d["id"]: d for d in info["devices"]}
    npus = []
    for nid, item in sorted(topo_info.items()):
        dev = devices.get(nid, {})
        npus.append(
            {
                "id": nid,
                "name": dev.get("name"),
                "health": dev.get("health"),
                "bus_id": dev.get("bus_id"),
                "hbm_used_mb": dev.get("hbm_used_mb"),
                "hbm_total_mb": dev.get("hbm_total_mb"),
                "power_w": dev.get("power_w"),
                "temp_c": dev.get("temp_c"),
                "affinity": item["affinity"],
                "affinity_cpus": item["affinity_cpus"],
                "affinity_source": "npu_smi_topo",
                "numa_node": topology.node_of_cpus(item["affinity_cpus"], nodes),
                "hccs": item["hccs"],
            }
        )

    return {
        "meta": {
            "host": "hostname-ojufk",
            "kernel": "5.10.0-oe2203.aarch64",
            "machine": "aarch64",
            "ts_ns": 0,
            "numa_source": "sysfs",
            "fabric": "hccs",
            "locality_mode": "topo_affinity",
            "affinity_granularity": "group",
            "package_count": 8,
            "die_count": 8,
        },
        "nodes": nodes,
        "cpus": cpus,
        "smt": {"enabled": False},
        "l3_domains": l3,
        "npu": npus,
        "nic": [
            {"name": "enp67s0f0", "numa_node": 2, "operstate": "up", "speed": "100000", "pci": "0000:43:00.0"},
            {"name": "enp61s0f0", "numa_node": 0, "operstate": "down", "speed": "-1", "pci": "0000:3d:00.0"},
        ],
        "storage": [
            {"name": "nvme0n1", "numa_node": 0, "size_gb": 7000, "rotational": "0", "pci": "0000:3e:00.0"},
            {"name": "nvme2n1", "numa_node": 4, "size_gb": 7000, "rotational": "0", "pci": "0000:be:00.0"},
        ],
    }


def a3_topo(nodes=8, cpus_per_node=80, npus=8, die_pairs=None):
    """A3 超节点形态：Phy-ID 矩阵、无 NPU→CPU 亲和，NPU 与各 node 等距。"""
    node_list = []
    for nid in range(nodes):
        cs = list(range(nid * cpus_per_node, (nid + 1) * cpus_per_node))
        node_list.append(
            {
                "node": nid,
                "cpus": cs,
                "cpulist": topology.cpus_to_str(cs),
                "distance": [10 if i == nid else 20 for i in range(nodes)],
                "mem_gb": 251.9,
            }
        )
    npu_list = [
        {
            "id": i,
            "name": "Ascend910",
            "health": "OK",
            "bus_id": None,
            "affinity": None,
            "affinity_cpus": [],
            "affinity_source": "none",
            "numa_node": None,
            "hccs": {},
        }
        for i in range(npus)
    ]
    meta = {
        "host": "a3-supernode",
        "kernel": "6.6.0",
        "machine": "aarch64",
        "ts_ns": 0,
        "numa_source": "sysfs",
        "fabric": "hccs_supernode",
        "locality_mode": "global_slice",
        "topo_format": "phy_matrix",
        "has_cpu_affinity": False,
        "die_count": npus,
        "package_count": max(1, npus // 2),
    }
    if die_pairs:
        meta["die_groups"] = die_pairs
        meta["die_count"] = sum(len(p) for p in die_pairs)
        meta["package_count"] = len(die_pairs)
    return {
        "meta": meta,
        "nodes": node_list,
        "cpus": [],
        "smt": {"enabled": True},
        "l3_domains": [],
        "npu": npu_list,
        "nic": [],
        "storage": [],
    }


def make_unit(unit_id, npu, all_cpus, pid=None, affinity=None, framework="vllm"):
    """构造一个最小 unit 输入（discover 之后的中间态）。"""
    return {
        "unit_id": unit_id,
        "unit_type": "rank",
        "framework": framework,
        "pid": pid if pid is not None else 1000 + unit_id,
        "pids": [pid if pid is not None else 1000 + unit_id],
        "primary_npu": npu,
        "npus": [npu],
        "npus_significant": [npu],
        "npu_node": None,
        "current_affinity": affinity if affinity is not None else topology.cpus_to_str(all_cpus),
        "current_affinity_cpus": list(all_cpus),
        "current_node": 0,
        "thread_roles": {"comm": 8, "io": 2, "compute": 20, "main": 2},
        "deviations": [],
        "numa_maps": {"ok": False},
    }


def fake_status(pid, uid=None):
    item = PID_INFO.get(int(pid))
    if not item:
        return None
    uid = OTHER_UID if uid is None else uid
    return {
        "Name": "VLLM::Worker",
        "State": "S (sleeping)",
        "PPid": str(item["ppid"]),
        "Threads": "90",
        "Cpus_allowed_list": item["affinity"],
        "Mems_allowed_list": "0-7",
        "Uid": "%d\t%d\t%d\t%d" % (uid, uid, uid, uid),
    }


def fake_threads(pid):
    pid = int(pid)
    out, tid = [], pid * 1000
    for comm, count in THREAD_COUNTS:
        for _ in range(count):
            out.append(
                {
                    "tid": tid,
                    "comm": comm,
                    "state": "S",
                    "affinity": PID_INFO[pid]["affinity"],
                    "last_cpu": PID_INFO[pid]["last_cpu"],
                }
            )
            tid += 1
    return out


class MockProcfs:
    """替换 procfs 的读取函数，模拟 A2 实机进程视图（无 /proc 依赖）。"""

    def __init__(self, uid=OTHER_UID, numa_maps_ok=False):
        self.uid = uid
        self.numa_maps_ok = numa_maps_ok
        self._saved = {}

    def __enter__(self):
        self._saved = {
            "read_status": procfs.read_status,
            "read_cmdline": procfs.read_cmdline,
            "read_cgroup": procfs.read_cgroup,
            "read_cgroup_cpuset": procfs.read_cgroup_cpuset,
            "read_cgroup_mems": procfs.read_cgroup_mems,
            "read_uid": procfs.read_uid,
            "read_last_cpu": procfs.read_last_cpu,
            "read_numa_maps": procfs.read_numa_maps,
            "read_threads": procfs.read_threads,
            "read_comm": procfs.read_comm,
            "list_tids": procfs.list_tids,
            "pid_exists": procfs.pid_exists,
        }
        procfs.read_status = lambda pid: fake_status(pid, self.uid)
        procfs.read_cmdline = lambda pid: "VLLM::Worker"
        procfs.read_cgroup = lambda pid: {
            "version": "v1",
            "path": "/docker/demo",
            "controllers": {"cpuset": "/docker/demo"},
        }
        procfs.read_cgroup_cpuset = lambda pid, cgroup=None: "0-191"
        procfs.read_cgroup_mems = lambda pid, cgroup=None: "0-7"
        procfs.read_uid = lambda pid: self.uid
        procfs.read_last_cpu = lambda pid: PID_INFO[int(pid)]["last_cpu"]
        procfs.read_numa_maps = lambda pid: (
            {"ok": True, "reason": None, "pages": {}, "total": 0}
            if self.numa_maps_ok
            else {"ok": False, "reason": "permission_denied", "pages": {}, "total": 0}
        )
        procfs.read_threads = fake_threads
        procfs.read_comm = lambda pid, tid=None: self._comm(tid if tid is not None else pid)
        procfs.list_tids = lambda pid: [int(pid)] + [t["tid"] for t in fake_threads(pid)]
        procfs.pid_exists = lambda pid: int(pid) in PID_INFO
        return self

    @staticmethod
    def _comm(tid):
        tid = int(tid)
        if tid in PID_INFO:
            return "VLLM::Worker"
        pid = tid // 1000
        idx = tid - pid * 1000
        for comm, count in THREAD_COUNTS:
            if idx < count:
                return comm
            idx -= count
        return "VLLM::Worker"

    def __exit__(self, *exc):
        for name, func in self._saved.items():
            setattr(procfs, name, func)
        return False


def temp_out():
    """测试产物目录：临时目录，不写 skill 目录。"""
    return tempfile.mkdtemp(prefix="hostcpu-bindtune-test-")
