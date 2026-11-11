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

"""/proc 读取子集：status / cmdline / cgroup / affinity / numa_maps / threads。

权限不足时返回结构化错误，不抛异常，由上层标注 view_limited。
"""

import logging
import os
import re

_LOG = logging.getLogger(__name__)


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except (FileNotFoundError, PermissionError) as exc:
        _LOG.debug("read %s failed: %s", path, type(exc).__name__)
        return None
    except Exception as exc:  # noqa: BLE001 - 进程退出等竞态统一降级
        _LOG.debug("read %s failed: %s: %s", path, type(exc).__name__, exc)
        return None


def pid_exists(pid):
    return os.path.isdir("/proc/%d" % int(pid))


def list_tids(pid):
    try:
        task_dir = "/proc/%d/task" % int(pid)
        return sorted(int(x) for x in os.listdir(task_dir) if x.isdigit())
    except Exception as exc:  # noqa: BLE001 - 进程退出/权限不足按不可读降级
        _LOG.debug("list %s failed: %s", pid, type(exc).__name__)
        return []


def read_comm(pid, tid=None):
    """读进程/线程 comm；tid 为空时读主线程。"""
    if tid is None:
        path = "/proc/%d/comm" % int(pid)
    else:
        path = "/proc/%d/task/%d/comm" % (int(pid), int(tid))
    return (_read_text(path) or "").strip()


def read_status(pid):
    text = _read_text("/proc/%d/status" % int(pid))
    if text is None:
        return None
    out = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def read_cmdline(pid):
    try:
        with open("/proc/%d/cmdline" % int(pid), "rb") as f:
            raw = f.read()
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    except Exception as exc:  # noqa: BLE001 - 权限不足/进程退出返回空 cmdline
        _LOG.debug("read cmdline %s failed: %s", pid, type(exc).__name__)
        return ""


def read_cgroup(pid):
    text = _read_text("/proc/%d/cgroup" % int(pid))
    if text is None:
        return {"version": None, "path": None, "controllers": {}, "error": "unreadable"}
    controllers = {}
    version = None
    path = None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, ctrl, cgpath = parts
        if ctrl == "":
            version = "v2"
            path = cgpath
        else:
            version = version or "v1"
            for name in ctrl.split(","):
                controllers[name] = cgpath
            path = path or cgpath
    return {"version": version, "path": path, "controllers": controllers}


def read_cgroup_cpuset(pid, cgroup=None):
    value = _cgroup_file(pid, cgroup, "cpuset.cpus.effective")
    if value:
        return value.strip()
    value = _cgroup_file(pid, cgroup, "cpuset.cpus")
    return value.strip() if value else None


def read_cgroup_mems(pid, cgroup=None):
    value = _cgroup_file(pid, cgroup, "cpuset.mems.effective")
    if value:
        return value.strip()
    value = _cgroup_file(pid, cgroup, "cpuset.mems")
    return value.strip() if value else None


def _cgroup_file(pid, cgroup, name):
    cg = cgroup or read_cgroup(pid)
    if not cg.get("path") or not cg.get("version"):
        return None
    if cg["version"] == "v2":
        base = "/sys/fs/cgroup" + cg["path"]
    else:
        # v1 每行一个控制器：只有拿到 cpuset 控制器的层级才可信；
        # 用其它控制器的 path 兜底会读到错误层级（更宽的上层核集）且不报错。
        cgpath = cg.get("controllers", {}).get("cpuset")
        if cgpath is None:
            return None
        base = "/sys/fs/cgroup/cpuset" + cgpath
    return _read_text(os.path.join(base, name))


def read_uid(pid):
    status = read_status(pid)
    if not status:
        return None
    parts = (status.get("Uid") or "").split()
    return int(parts[0]) if parts and parts[0].isdigit() else None


def read_mems_allowed(pid):
    status = read_status(pid)
    return status.get("Mems_allowed_list") if status else None


def _stat_processor(path):
    """读取 /proc/<pid>/stat 的 processor 字段（最近运行 CPU）。

    stat 第 39 字段（1-based）为 processor；因 comm 可含空格/括号，
    以最后一个 ')' 为界取其后字段（fields[36]，0-based）。
    """
    text = _read_text(path)
    if not text:
        return None
    rparen = text.rfind(")")
    if rparen < 0:
        return None
    fields = text[rparen + 2:].split()
    if len(fields) <= 36:
        return None
    try:
        return int(fields[36])
    except ValueError:
        return None


def read_last_cpu(pid):
    """无权限读 numa_maps 时的弱证据：进程最近运行的 CPU。"""
    return _stat_processor("/proc/%d/stat" % int(pid))


def read_numa_maps(pid):
    """返回 {ok, pages: {node: pages}, total, reason}。

    大进程 numa_maps 可达数百 KB，逐行流式读取，避免整文件载入内存。
    """
    path = "/proc/%d/numa_maps" % int(pid)
    pages = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for m in re.finditer(r"\bN(\d+)=(\d+)\b", line):
                    node = int(m.group(1))
                    pages[node] = pages.get(node, 0) + int(m.group(2))
    except PermissionError:
        return {"ok": False, "reason": "permission_denied", "pages": {}, "total": 0}
    except FileNotFoundError:
        return {"ok": False, "reason": "not_found", "pages": {}, "total": 0}
    except OSError as exc:  # 进程退出等竞态：文件消失按不可读降级
        return {"ok": False, "reason": str(exc), "pages": {}, "total": 0}

    total = sum(pages.values())
    return {"ok": True, "reason": None, "pages": pages, "total": total}


def read_threads(pid, limit=None):
    """读取线程 comm / affinity；无权限的线程跳过。

    limit 只影响读取条数（读到即停），用于展示场景避免遍历数千线程。
    """
    task_dir = "/proc/%d/task" % int(pid)
    try:
        tids = sorted(int(x) for x in os.listdir(task_dir) if x.isdigit())
    except Exception as exc:  # noqa: BLE001 - 进程退出/权限不足按不可读降级
        _LOG.debug("list threads %s failed: %s", pid, type(exc).__name__)
        return []

    out = []
    for tid in tids:
        if limit is not None and len(out) >= limit:
            break
        comm = _read_text("/proc/%d/task/%d/comm" % (int(pid), tid))
        status = _read_text("/proc/%d/task/%d/status" % (int(pid), tid))
        if status is None:
            continue
        affinity = None
        state = None
        for line in status.splitlines():
            if line.startswith("Cpus_allowed_list:"):
                affinity = line.split(":", 1)[1].strip()
            elif line.startswith("State:"):
                state = line.split(":", 1)[1].strip()
        out.append({
            "tid": tid,
            "comm": (comm or "").strip(),
            "state": state,
            "affinity": affinity,
            "last_cpu": _stat_processor("/proc/%d/task/%d/stat" % (int(pid), tid)),
        })
    return out


def summarize(pid):
    status = read_status(pid)
    if status is None:
        return {"pid": int(pid), "exists": False}
    cg = read_cgroup(pid)
    return {
        "pid": int(pid),
        "exists": True,
        "name": status.get("Name"),
        "state": status.get("State"),
        "ppid": status.get("PPid"),
        "threads": status.get("Threads"),
        "cpus_allowed_list": status.get("Cpus_allowed_list"),
        "mems_allowed_list": status.get("Mems_allowed_list"),
        "cmdline": read_cmdline(pid),
        "cgroup": cg,
        "cgroup_cpuset": read_cgroup_cpuset(pid, cg),
        "cgroup_mems": read_cgroup_mems(pid, cg),
        "uid": read_uid(pid),
        "mems_allowed": status.get("Mems_allowed_list"),
        "last_cpu": read_last_cpu(pid),
        "numa_maps": read_numa_maps(pid),
    }


if __name__ == "__main__":
    import json
    import sys
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else os.getpid()
    data = summarize(pid)
    data["thread_list"] = read_threads(pid, limit=20)
    print(json.dumps(data, ensure_ascii=False, indent=2))
