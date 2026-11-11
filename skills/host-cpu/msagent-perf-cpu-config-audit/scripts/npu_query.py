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
"""npu_query.py — npu-smi 唯一封装点。

本 skill 内 npu-smi 子进程调用与输出解析仅此一处（脚本分工要点「npu-smi 单点封装」）；
discover.py 的 NPU 锚点消费本模块的卡↔进程映射。
安全约束：子进程调用仅固定命令行（无 shell=True、无用户可控参数拼接）。

已按 npu-smi 26.1.1（openEuler aarch64）真实输出校准：
- info 总览：每卡两行（NPU 行 + Chip 行），`|` 分列；双 die 机型（910C）每封装
  两行 NPU 行（按 NPU ID 去重）；
- info 进程分节：表头含 "Process id"，列序 NPU | Chip | Process id | name | mem | container-pid；
- info -l：封装级权威计数（Total Count + 每封装 Product Name / Chip Count），
  "-l 的 NPU ID" 语义为物理封装；
- info -m：die 级映射（NPU ID | Chip ID | Chip Logic ID | Chip Phy-ID | Chip Name，
  Mcu 行过滤）；Logic ID 即软件设备号 davinciN；
- info -t topo：双形态——NPUx 封装级矩阵（PCIe/Box8，末列 CPU Affinity）或
  Phy-ID die 级矩阵（910C，关系符号 SIO/HCCS_SW/UB）；fabric 由关系符号判定，
  fabric 与 CPU Affinity 是两条独立信号。
"""
from __future__ import annotations

import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

NPSMI_BIN = "npu-smi"
_TIMEOUT = 15  # 秒

# topo 关系符号（fabric 判定依据；UB=灵衢统一总线，HCCS_SW=交换互联，
# SIO=同封装 die 片内直连，PHB/PIX/SYS=PCIe 树层级）
_RELATION_RE = re.compile(r"^[A-Z][A-Z_0-9]*$")
# 矩阵自指标记（对角线）与表头文字，均不计入关系符号
_SELF_MARKS = {"X", "SELF"}
_HEADER_MARKS = {"CPU", "AFFINITY"}


def _is_relation(token: str) -> bool:
    # 表头列标签（NPU0/Phy-ID0）与 "CPU Affinity" 字样不是关系符号
    if re.match(r"^(NPU|Phy-ID)\d+$", token):
        return False
    upper = token.upper()
    return (bool(_RELATION_RE.match(token))
            and upper not in _SELF_MARKS
            and upper not in _HEADER_MARKS)


def _run(args: List[str]) -> Optional[str]:
    """运行 npu-smi 固定子命令，失败/超时返回 None。"""
    cmd = [NPSMI_BIN] + args
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_TIMEOUT,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return proc.stdout.decode("utf-8", errors="replace")
    except ValueError:
        return None


def _split_cells(line: str) -> List[str]:
    """按 `|` 分列并去空白。"""
    return [cell.strip() for cell in line.split("|") if cell.strip()]


def parse_info(text: str) -> Dict[str, Any]:
    """解析 npu-smi info 总览（每卡 NPU 行 + Chip 行，`|` 分列）。

    NPU 行首格为 'NPU  Name' 合并格（如 '0     910B2'）；
    Chip 行首格为纯数字、次列为 PCI ID；
    输出后半段含进程分节（表头 "Process id"），其数据行与 NPU 行形似，
    遇到该表头后跳过整个分节，只解析总览节。
    """
    result: Dict[str, Any] = {"npus": [], "parse_ok": True}
    in_process_section = False
    seen_npus: set = set()
    for line in text.splitlines():
        if "Process id" in line:
            in_process_section = True
            continue
        if in_process_section:
            continue  # 进程节数据行由 parse_processes 单独消费
        if "|" not in line:
            continue
        cells = _split_cells(line)
        if not cells:
            continue
        # 跳过分隔行与表头
        if set(cells[0]) <= {"=", "+"} or cells[0] in ("NPU   Name", "Chip"):
            continue
        first_tokens = cells[0].split()
        if len(first_tokens) >= 2 and first_tokens[0].isdigit():
            npu_id = int(first_tokens[0])
            name = first_tokens[1]
        else:
            continue  # Chip 行/其他行：不计入卡列表
        if re.match(r"^[0-9A-Fa-f]{4}:", name):
            continue  # 防御：型号格误为 PCI ID
        if name.isdigit():
            continue  # 双 die 机型 'NPU Chip' 合并格（名称为纯数字），非 NPU 行
        if npu_id in seen_npus:
            continue  # 双 die 机型每封装两行 NPU 行，按 NPU ID 去重（首行胜出）
        seen_npus.add(npu_id)
        entry: Dict[str, Any] = {"npu": npu_id, "name": name}
        if len(cells) >= 2:
            entry["health"] = cells[1]
        result["npus"].append(entry)
    return result


def parse_processes(text: str) -> List[Dict[str, Any]]:
    """解析进程占用分节（表头含 "Process id"）。

    真实列序（npu-smi 26.1.1）：NPU+Chip 为合并首格（如 '0       0'），
    其后为 Process id | Process name | Process memory | container pid。
    行例：| 0       0 | 293350 | VLLMWorker | 117 | NA |
    """
    processes: List[Dict[str, Any]] = []
    in_proc_section = False
    for line in text.splitlines():
        if "Process id" in line:
            in_proc_section = True
            continue
        if not in_proc_section or "|" not in line:
            continue
        cells = _split_cells(line)
        if len(cells) < 3:
            continue  # 分隔行 / 空行
        # 首格 = NPU + Chip 合并（两个数字），次格 = pid
        head_tokens = cells[0].split()
        if len(head_tokens) >= 2 and head_tokens[0].isdigit() and cells[1].isdigit():
            entry = {"npu": int(head_tokens[0]), "pid": int(cells[1]),
                     "name": cells[2] if len(cells) > 2 else ""}
        elif len(head_tokens) == 1 and head_tokens[0].isdigit() and cells[1].isdigit():
            # 兼容列序：首格仅 NPU，次格 Chip，第三格 pid
            if len(cells) >= 4 and cells[2].isdigit():
                entry = {"npu": int(head_tokens[0]), "pid": int(cells[2]),
                         "name": cells[3] if len(cells) > 3 else ""}
            else:
                continue
        else:
            continue
        if len(cells) >= 4 and cells[3].isdigit():
            entry["mem_mb"] = int(cells[3])
        processes.append(entry)
    return processes


def _to_int(text: str) -> Optional[int]:
    try:
        return int(text.strip())
    except ValueError:
        return None


def parse_info_l(text: str) -> Dict[str, Any]:
    """解析 npu-smi info -l（封装级权威计数）。

    -l 的 "NPU ID" 语义为物理封装（双 die 机型也只返回封装数），
    Total Count 为封装总数，每封装附 Product Name 与 Chip Count。
    """
    result: Dict[str, Any] = {"total_count": 0, "npus": []}
    current: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "Total Count":
            result["total_count"] = _to_int(value) or 0
        elif key == "NPU ID":
            if current is not None:
                result["npus"].append(current)
            current = {"npu": _to_int(value), "name": "", "chips": 0}
        elif key == "Product Name" and current is not None:
            current["name"] = value
        elif key == "Chip Count" and current is not None:
            current["chips"] = _to_int(value) or 0
    if current is not None:
        result["npus"].append(current)
    return result


def parse_info_m(text: str) -> Dict[str, Any]:
    """解析 npu-smi info -m（die 级映射）。

    列序随版本不同：新版 NPU ID | Chip ID | Chip Logic ID | Chip Phy-ID | Chip Name，
    旧版（910B2）无 Chip Phy-ID 列且名称含空格。双 die 机型（910C）每封装
    2 die + 1 Mcu 行；Chip ID 为封装内 die 序，Chip Logic ID 为软件设备号
    （对应 /dev/davinciN）；Mcu 行与表头行过滤。
    返回 {"dies": [...], "groups": [[0,1],...], "logic_ids": [...]}，
    groups 按封装聚合（同封装 die 对 = 绑核不可拆分单位）。
    """
    dies: List[Dict[str, Any]] = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) < 4:
            continue
        npu_id = _to_int(tokens[0])
        chip_id = _to_int(tokens[1])
        logic_id = _to_int(tokens[2])
        if npu_id is None or chip_id is None or logic_id is None:
            continue  # 表头 / Mcu 行
        # 第 4 列可能是 Phy-ID（数字），也可能直接是名称（旧版）
        phy_id = _to_int(tokens[3])
        name = " ".join(tokens[4:] if phy_id is not None else tokens[3:])
        if not name or name.upper().startswith("MCU"):
            continue
        dies.append({"npu": npu_id, "chip": chip_id, "logic_id": logic_id,
                     "phy_id": phy_id if phy_id is not None else logic_id,
                     "name": name})
    groups: List[List[int]] = []
    for npu_id in sorted({d["npu"] for d in dies}):
        logic = sorted(d["logic_id"] for d in dies if d["npu"] == npu_id)
        if logic:
            groups.append(logic)
    return {"dies": dies, "groups": groups,
            "logic_ids": sorted(d["logic_id"] for d in dies)}


def _fabric_from_relations(relations: set) -> str:
    """从 topo 关系符号判定 fabric（NPU 间互联平面，与 CPU Affinity 独立）。"""
    if not relations:
        return "unknown"
    if "UB" in relations:
        return "ub"
    if "HCCS_SW" in relations:
        return "hccs_sw"
    if any(r.startswith("HCCS") for r in relations):
        return "hccs"
    if relations & {"PHB", "PIX", "SYS"}:
        return "pcie"
    return "unknown"


def parse_topo(text: str) -> Dict[str, Any]:
    """解析 topo 表（双形态，fabric 与 CPU Affinity 为两条独立信号）。

    形态 1（npu，PCIe/Box8 如 910B）：首列 NPUx，末列 CPU Affinity 核集；
    形态 2（die，910C/A3 超节点）：首列 Phy-ID，格内为关系符号
    （SIO=同封装 die 片内直连；HCCS_SW/UB=交换互联，NPU 间等距）。
    返回 {"affinity": [...], "rows": N, "form": npu|die|unknown,
          "fabric": ub|hccs_sw|hccs|pcie|unknown, "relations": [...],
          "sio_pairs": [[a, b], ...]}
    """
    affinity: List[Dict[str, Any]] = []
    relations: set = set()
    form = "unknown"
    rows = 0
    matrix: Dict[int, List[Tuple[int, str]]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        npu_match = re.match(r"^NPU(\d+)\s+(.*)$", stripped)
        # die 级行标签实测为 "Phy-ID0"（表头首列同样带 Phy-ID 前缀），
        # 兼容纯数字标签（部分版本/离线样本）
        phy_match = re.match(r"^(?:Phy-ID)?\s*(\d+)\s+(.*)$", stripped)
        if npu_match:
            npu_id = int(npu_match.group(1))
            rest = npu_match.group(2).split()
            # 表头同样以 "NPU0 NPU1 ..." 开头（无关系符号），需跳过避免多计一行
            symbols = [t for t in rest if _is_relation(t)]
            if not symbols:
                continue
            if form == "unknown":
                form = "npu"
            rows += 1
            relations.update(symbols)
            # 末段为 CPU 亲和核集（如 144-167；跨段用逗号）
            if rest:
                tail = rest[-1]
                if re.fullmatch(r"[0-9,\-]+", tail):
                    affinity.append({"npu": npu_id, "cpus": tail})
        elif phy_match and form in ("unknown", "die"):
            # 保留列号：格内 "-" 等非关系符号会被过滤，过滤后下标 != Phy-ID 列号
            cells = [(col, token) for col, token in enumerate(phy_match.group(2).split())
                     if _is_relation(token)]
            if not cells:
                continue
            if form == "unknown":
                form = "die"
            rows += 1
            phy = _to_int(phy_match.group(1))
            if phy is None:
                continue
            matrix[phy] = cells
            relations.update(token for _, token in cells)
    # SIO 配对（die 级矩阵）：SIO 所在列号即对端 Phy-ID（列头按 Phy-ID 顺序）
    sio_pairs: List[List[int]] = []
    if form == "die" and "SIO" in relations:
        seen_pairs: set = set()
        for phy, cells in matrix.items():
            for col, sym in cells:
                if sym == "SIO":
                    pair = (min(phy, col), max(phy, col))
                    if pair[0] != pair[1] and pair not in seen_pairs:
                        seen_pairs.add(pair)
                        sio_pairs.append([pair[0], pair[1]])
                    break
        sio_pairs.sort()
    return {"affinity": affinity, "rows": rows, "form": form,
            "fabric": _fabric_from_relations(relations),
            "relations": sorted(relations), "sio_pairs": sio_pairs}


def query_snapshot() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """一次 npu-smi info 同时产出总览与卡↔进程映射。

    单次执行可能耗时数秒（卡多/驱动繁忙），总览与进程分节取自同一份文本，
    既避免重复执行，也保证两处结论来自同一时刻的快照。
    """
    text = _run(["info"])
    if text is None:
        return {"available": False}, []
    return {"available": True, **parse_info(text)}, parse_processes(text)


def query_processes() -> List[Dict[str, Any]]:
    """查询卡↔进程映射（npu-smi info 全量输出中抓进程分节）。"""
    text = _run(["info"])
    if text is None:
        return []
    return parse_processes(text)


def query_topo() -> Dict[str, Any]:
    """npu-smi info -t topo：双形态解析（封装级 NPUx / die 级 Phy-ID 矩阵）。"""
    text = _run(["info", "-t", "topo"])
    if text is None:
        return {"affinity": [], "rows": 0, "form": "unknown",
                "fabric": "unknown", "relations": [], "sio_pairs": []}
    return parse_topo(text)


def query_info_l() -> Optional[Dict[str, Any]]:
    """npu-smi info -l：封装级权威计数（Total Count + 每封装 Chip Count）。"""
    text = _run(["info", "-l"])
    if text is None:
        return None
    return parse_info_l(text)


def query_die_map() -> Optional[Dict[str, Any]]:
    """npu-smi info -m：die 级映射（同封装 die 分组 + Logic ID ↔ davinciN）。"""
    text = _run(["info", "-m"])
    if text is None:
        return None
    return parse_info_m(text)
