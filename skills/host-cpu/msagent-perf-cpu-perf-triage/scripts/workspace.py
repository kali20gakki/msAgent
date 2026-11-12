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

"""perf-triage 工作空间与产物目录解析（与 3 个基础 Skill 同构 + 编排层扩展）。

约定：
  会话根 = <workspace>/workspaces/<host>/<timestamp>/，与基础 Skill 落盘口径一致；
  阶段子目录 = 00-config / 01-trace / 02-perf / actions（目录序 = 阶段序）；
  汇总页 index.html 与 manifest.json 固定在会话根。
优先级：显式 root 参数 > HOSTCPU_WORKSPACE > 当前工作目录（cwd）。
"""

import json
import os
import re
import socket
import tempfile
import time

PHASES = ("00-config", "01-trace", "02-perf", "actions")
RESIDUAL_CONVERGED_MAX = 30.0  # 残余占比上限，quantify/report 共用同一口径


def safe_name(value):
    """外部输入进路径前清洗：仅保留字母数字/点/横线/下划线，截断 64。"""
    cleaned = re.sub(r"[^\w.-]+", "_", str(value or "")).strip("._")
    return (cleaned or "unknown")[:64]


def workspace_root(root=None):
    if root:
        return os.path.abspath(os.path.expanduser(root))
    env = os.environ.get("HOSTCPU_WORKSPACE")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.abspath(os.getcwd())


def new_session(host=None, timestamp=None, root=None, create=True):
    """建立会话目录，返回会话描述 dict。

    create=False 时只计算路径不落盘（调试/预演用）。
    """
    host = safe_name(host or socket.gethostname())
    ts = safe_name(timestamp or time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    base = os.path.join(workspace_root(root), "workspaces", host, ts)
    if create:
        for name in PHASES:
            os.makedirs(os.path.join(base, name), exist_ok=True)
    return {"run_id": "%s_%s" % (host, ts), "host": host,
            "timestamp": ts, "root": base}


def phase_dir(session_root, phase, create=True):
    path = os.path.join(session_root, phase)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    """原子写 JSON：先写同目录临时文件再 os.replace，避免中断留下半截文件。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=parent or None)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def manifest_upsert(session_root, section, data):
    """更新 manifest.json 的某个阶段段（保持其余段不变）。

    manifest 是各阶段共享的汇总文件，本函数按阶段串行调用；写回走原子替换。
    文件存在但解析失败时先备份为 .corrupt-<ts> 再重建，避免静默清空其它阶段数据。
    """
    path = os.path.join(session_root, "manifest.json")
    manifest = {}
    if os.path.isfile(path):
        loaded = load_json(path)
        if isinstance(loaded, dict):
            manifest = loaded
        else:
            os.replace(path, "%s.corrupt-%s" % (path, time.strftime("%Y%m%d-%H%M%S")))
    manifest.setdefault("run_id", os.path.basename(os.path.normpath(session_root)))
    manifest.setdefault("phases", {})
    manifest["phases"][section] = data
    manifest["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    write_json(path, manifest)
    return path
