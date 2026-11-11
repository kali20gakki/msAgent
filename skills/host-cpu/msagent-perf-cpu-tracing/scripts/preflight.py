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

"""preflight：只探测 tracefs 可写性 / 挂载点 / 权限，不探测任何用户态采集器。

设计约束：不使用 trace-cmd / perf sched；tracefs 不可写即判「不可采集」。
"""

import os

CANDIDATES = ("/sys/kernel/tracing", "/sys/kernel/debug/tracing")


def _writable(path):
    """tracefs 根目录可写：要求存在可写的 trace 控制文件。"""
    try:
        return os.path.isfile(os.path.join(path, "trace")) and os.access(os.path.join(path, "trace"), os.W_OK)
    except OSError:
        return False


def _dir_writable(path):
    """目录可写：instances 是目录、没有 trace 文件，不能用 _writable 判定。"""
    try:
        return os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)
    except OSError:
        return False


def _probe(base):
    instances = os.path.join(base, "instances")
    info = {
        "path": base,
        "trace_writable": _writable(base),
        "instance_writable": _dir_writable(instances),
        "current_tracer": _read(os.path.join(base, "current_tracer")),
        "available_tracers": _read(os.path.join(base, "available_tracers")),
    }
    info["available"] = bool(info["trace_writable"] or info["instance_writable"])
    return info


def detect_tracefs():
    """遍历全部候选挂载点：优先返回可写的；都不可写时返回首个存在的（供诊断）。"""
    first = None
    for base in CANDIDATES:
        if not os.path.isdir(base):
            continue
        info = _probe(base)
        if first is None:
            first = info
        if info["available"]:
            return info
    return first or {"path": None, "available": False}


def _read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def run(path=None):
    """返回 preflight dict；available=False 时调用方应判「不可采集」。"""
    info = detect_tracefs() if not path else _probe(path)
    import platform
    info.update({
        "platform": platform.system(),
        "kernel": platform.release(),
        "in_container": os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"),
    })
    if not info.get("available"):
        info["hint"] = (
            "tracefs 不可写：容器需挂载 /sys/kernel/tracing（或 /sys/kernel/debug），"
            "或在宿主机执行采集后把 trace 文本带回 parse"
        )
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(run(), ensure_ascii=False, indent=2))
