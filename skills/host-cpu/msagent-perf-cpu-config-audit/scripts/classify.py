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
# -*- coding: utf-8 -*-
"""classify.py — 进程角色分类（process-roles 规则，与 bind-tune 口径一致）。"""
from __future__ import annotations

import re
from typing import Any, Dict, List

# comm 白名单（命中即归类，顺序即优先级）
MONITOR_COMM = {"npu-smi", "node_exporter", "dcgm-exporter", "telescope", "telegraf"}
CONTAINER_COMM_RE = re.compile(r"^(containerd-shim|runc|dockerd|containerd|crio)")
LOGGER_COMM = {"fluentd", "filebeat", "rsyslogd", "logagent"}
COMPILER_COMM = {"gcc", "g++", "cc1", "make", "pip", "jupyter"}
TRAIN_CMDLINE_RE = re.compile(r"(torchrun|msrun|deepspeed|accelerate|torchelast)", re.I)
INFER_CMDLINE_RE = re.compile(r"(vllm|mindie|sglang|tritonserver)", re.I)
PYTHON_COMM_RE = re.compile(r"^python[0-9.]*$")


def classify(comm: str, cmdline: str, has_npu: bool) -> str:
    """分类优先级：白名单 → NPU 锚点 → comm/cmdline 模式 → other。

    模式匹配范围与 discover._classify_process 一致（comm + cmdline），
    避免同一进程在发现层与分类层得到不同角色。
    """
    if comm in MONITOR_COMM:
        return "monitor"
    if CONTAINER_COMM_RE.match(comm):
        return "container_runtime"
    if comm in LOGGER_COMM:
        return "logger"
    if comm in COMPILER_COMM:
        return "compiler"
    if not has_npu:
        return "other"
    blob = f"{comm} {cmdline}"
    if INFER_CMDLINE_RE.search(blob):
        return "infer_main"
    if TRAIN_CMDLINE_RE.search(blob) or PYTHON_COMM_RE.match(comm):
        return "train_main"
    return "other"


def classify_targets(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """只补全缺失的 role，不覆盖 discover 已产出的分类（分类事实源唯一）。"""
    for target in targets:
        if target.get("role"):
            continue
        target["role"] = classify(
            target.get("comm", ""),
            target.get("cmdline", ""),
            bool(target.get("npus")),
        )
    return targets
