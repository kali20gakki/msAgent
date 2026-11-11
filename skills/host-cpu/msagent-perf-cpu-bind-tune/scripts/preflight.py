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

"""执行前探测：npu-smi 存在性、容器/cgroup、工具可用性、形态判定。

preflight 只做可执行文件存在性探测，不调用 npu-smi、不解析其输出；
npu-smi 的实际调用与解析统一在 npu_query.py。
"""

import getpass
import logging
import os
import shutil

import npu_query
import procfs

_LOG = logging.getLogger(__name__)


def _exists(path):
    return os.path.exists(path)


def _readable(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.read(1)
        return True
    except Exception:
        return False


def _current_uid():
    return os.getuid() if hasattr(os, "getuid") else None


def _current_user():
    """当前用户名；无 tty / 受限环境 getuser() 可能抛错，逐级回退。"""
    try:
        return getpass.getuser()
    except Exception:
        return (os.environ.get("USER")
                or os.environ.get("LOGNAME")
                or os.environ.get("USERNAME")
                or "unknown")


def collect():
    npu_smi = npu_query.find_npu_smi()
    in_container = _exists("/.dockerenv") or _exists("/run/.containerenv")
    cgroup_v2 = _exists("/sys/fs/cgroup/cgroup.controllers")
    cgroup_path = "/proc/1/cgroup"
    try:
        with open(cgroup_path, "r", encoding="utf-8", errors="replace") as f:
            cgroup_text = f.read()
        if "docker" in cgroup_text or "kubepods" in cgroup_text or "containerd" in cgroup_text:
            in_container = True
    except Exception as exc:  # noqa: BLE001 - /proc/1/cgroup 不可读时保持容器初判
        _LOG.debug("read %s failed: %s", cgroup_path, type(exc).__name__)
        cgroup_text = ""

    proc_hidden = not _readable("/proc/1/status")
    numa_maps_self_ok = procfs.read_numa_maps(os.getpid()).get("ok", False)

    return {
        "npu_smi": npu_smi,
        "npu_smi_available": bool(npu_smi),
        "in_container": in_container,
        "cgroup_version": "v2" if cgroup_v2 else "v1",
        "has_taskset": bool(shutil.which("taskset")),
        "has_perf": bool(shutil.which("perf")),
        "uid": _current_uid(),
        "user": _current_user(),
        "is_root": _current_uid() == 0,
        "proc_hidden": proc_hidden,
        "numa_maps_self_ok": numa_maps_self_ok,
        "tracefs": {
            "new": _exists("/sys/kernel/tracing"),
            "old": _exists("/sys/kernel/debug/tracing"),
        },
        # 容器内、/proc/1 不可读、自身 numa_maps 不可读，任一成立都属于视图受限；
        # 保留分项字段便于定位受限来源。
        "view_limited": bool(in_container or proc_hidden or not numa_maps_self_ok),
    }


def blocked(reason, suggestion=None, preflight=None):
    return {
        "status": "blocked",
        "reason": reason,
        "suggestion": suggestion or "在 NPU 宿主机执行，或补齐驱动工具链后重试",
        "preflight": preflight or {},
    }


def detect_form(npu_info, pre):
    """根据 npu-smi 可用性与 NPU 占用自动判定形态。

    返回 (form, blocked_result)，form 为 in_run / pre_run。
    """
    if not pre.get("npu_smi_available"):
        return None, blocked("npu-smi 不可用", "安装/暴露 npu-smi，或在 NPU 宿主机执行", pre)
    if not npu_info.get("info_ok"):
        kind = npu_info.get("info_error_kind") or "call_failed"
        errors = "; ".join(npu_info.get("errors") or []) or "npu-smi info 调用失败"
        if kind == "permission_denied":
            return None, blocked(
                "npu-smi 权限不足",
                "使用 sudo 运行，或将当前用户加入 HwHiAiUser 组；本 skill 不自动提权",
                pre,
            )
        if kind == "timeout":
            return None, blocked("npu-smi 调用超时", "检查驱动/设备状态后重试", pre)
        if kind == "no_device":
            return None, blocked("未发现 NPU 设备", "确认 NPU 驱动与容器设备映射", pre)
        return None, blocked("npu-smi 调用失败", errors, pre)
    devices = npu_info.get("devices") or []
    if not devices:
        return None, blocked("未发现 NPU 设备", "确认 NPU 驱动与容器设备映射", pre)

    primaries = npu_query.primary_processes(npu_info.get("processes") or [])
    if primaries:
        return "in_run", None
    return "pre_run", None
