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

"""npu-smi 唯一封装点：设备、卡↔进程、die 映射、HCCS/CPU 亲和。

本模块不依赖其他 skill，自行读取：
1. `npu-smi info`（设备 + 进程）；
2. `npu-smi info -t topo`（NPU 间关系 + 推荐 CPU 亲和）；
3. `npu-smi info -l`（封装级权威计数）；
4. `npu-smi info -m`（die 级映射：同封装 die 分组 + Logic ID ↔ davinciN）。

双 die 机型（910C/A3）一颗物理封装含 2 个 die：设备/进程口径为封装，
软件设备与 topo 矩阵为 die 级；同封装 die 对经 SIO 片内直连，绑核不可跨 node 拆分。

不做进程发现、不做绑核判定，避免与 discover/model 职责重叠。
"""

import logging
import os
import re
import shutil
import subprocess

from cpu_list import parse_cpulist

_LOG = logging.getLogger(__name__)

# npu-smi 常见安装路径候选（驱动版本/发行版差异）；优先 PATH，再按序探测。
# 不在此列的路径可由后续版本追加，调用方不感知。
NPU_SMI_CANDIDATES = (
    "/usr/local/sbin/npu-smi",
    "/usr/local/bin/npu-smi",
    "/usr/bin/npu-smi",
)


def find_npu_smi():
    """返回 npu-smi 可执行路径；不存在时返回 None。"""
    path = shutil.which("npu-smi")
    if path:
        return path
    for cand in NPU_SMI_CANDIDATES:
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _run(cmd, timeout=15):
    """执行外部命令并统一返回 (rc, stdout, stderr)。

    安全约束：cmd 必须是参数列表形式（禁止 shell=True，避免拼接注入）；
    统一超时兜底，npu-smi 卡死时不阻塞主流程（rc=124 表征超时）。
    """
    try:
        p = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, text=True, errors="replace"
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:  # noqa: BLE001 - 上层统一降级
        _LOG.debug("run %s failed: %s: %s", cmd, type(exc).__name__, exc)
        return 125, "", str(exc)


def _classify_error(text):
    t = (text or "").lower()
    if "permission denied" in t or "eacces" in t:
        return "permission_denied"
    if "no such file" in t or "not found" in t:
        return "not_found"
    if "timeout" in t:
        return "timeout"
    if "no device" in t or "no npu" in t:
        return "no_device"
    return "call_failed"


# npu-smi info 行正则：设备首/次行、进程行与空卡行（导入时编译一次）
_DEV_FIRST = re.compile(
    r"^\|\s*(\d+)\s+(\S+)\s*\|\s*(\S+)\s*\|\s*([\d.]+)\s+(\d+)\s+(\d+)\s*/\s*(\d+)\s*\|"
)
_DEV_SECOND = re.compile(
    r"^\|\s*(\d+)\s*\|\s*([0-9A-Fa-f:.]+)\s*\|\s*([\d.]+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)\s*\|"
)
# 进程行兼容两种列数：新版 npu-smi（26.x）行尾带 "Process id in container"
# 列，旧版（如 25.3.rc1.2）无该列；container 列缺省时置 None。
_PROC_LINE = re.compile(
    r"^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(\d+)\s*(?:\|\s*(.+?)\s*)?\|\s*$"
)
_EMPTY_LINE = re.compile(r"^\|\s*No running processes found in NPU\s+(\d+)")


def _parse_dev_first(m, devices):
    """解析 NPU 首行：设备基础信息，双 die 时只补空缺字段。"""
    npu = int(m.group(1))
    entry = {
        "id": npu,
        "name": m.group(2),
        "health": m.group(3),
        "power_w": float(m.group(4)),
        "temp_c": int(m.group(5)),
        "hugepages_used": int(m.group(6)),
        "hugepages_total": int(m.group(7)),
    }
    if npu in devices:
        # 双 die 机型每封装两行 NPU 行：首行已记录，后续行只补空缺
        for key, value in entry.items():
            devices[npu].setdefault(key, value)
    else:
        devices[npu] = entry
    return npu


def _parse_dev_second(m, devices, current_npu):
    """解析设备第二行（chip 明细）：归属上一行 NPU，明细入 chips 并更新顶层。"""
    # 第二行的首个数字是 chip id（A2 上恒为 0），NPU id 取上一行；
    # 不能把第二行当成新 NPU，否则所有 HBM/ Bus-Id 都会覆盖到 NPU0。
    chip = int(m.group(1))
    npu = current_npu if current_npu is not None else chip
    dev = devices.setdefault(npu, {"id": npu})
    chip_entry = {
        "chip": chip,
        "bus_id": m.group(2),
        "aicore_pct": float(m.group(3)),
        "mem_used_mb": int(m.group(4)),
        "mem_total_mb": int(m.group(5)),
        "hbm_used_mb": int(m.group(6)),
        "hbm_total_mb": int(m.group(7)),
    }
    # 双 die 机型每封装有 chip 0/1 两行：明细入 chips，顶层字段保留
    # 最后一块 die 的取值（与既有单 die 行为一致，不臆造求和）
    dev.setdefault("chips", []).append(chip_entry)
    dev.update(chip_entry)


def _parse_proc_line(m):
    """解析进程行：容器 id 列缺省或为 npu-smi 的 NA 占位时统一置 None。"""
    cid = (m.group(6) or "").strip()
    return {
        "npu": int(m.group(1)),
        "chip": int(m.group(2)),
        "pid": int(m.group(3)),
        "name": m.group(4).strip(),
        "mem_mb": int(m.group(5)),
        "container_id": None if cid in ("", "NA") else cid,
    }


def parse_info(text):
    """解析 npu-smi info 的设备表与进程表。"""
    devices = {}
    processes = []
    empty_npus = []
    current_npu = None

    for line in text.splitlines():
        m = _DEV_FIRST.match(line)
        if m:
            current_npu = _parse_dev_first(m, devices)
            continue
        m = _DEV_SECOND.match(line)
        if m:
            _parse_dev_second(m, devices, current_npu)
            continue
        m = _PROC_LINE.match(line)
        if m:
            processes.append(_parse_proc_line(m))
            continue
        m = _EMPTY_LINE.match(line)
        if m:
            empty_npus.append(int(m.group(1)))

    return {
        "devices": sorted(devices.values(), key=lambda x: x["id"]),
        "processes": processes,
        "empty_npus": sorted(set(empty_npus)),
    }


def parse_topo(text):
    """解析 npu-smi info -t topo。

    兼容两种输出形态：
    - PCIe 形态：行首 NPU<n>，末列为 CPU Affinity（无此列则只解析关系）；
    - 超节点形态：Phy-ID 矩阵（SIO / HCCS_SW），表头含 CPU Affinity 时末列才是亲和核。

    返回 {npu_id: {"id", "affinity", "affinity_cpus", "hccs"}}。
    """
    out = {}
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    has_affinity = any("CPU Affinity" in ln for ln in lines)

    for line in lines:
        m = re.match(r"^\s*(?:NPU|Phy-ID)(\d+)\s+(.+?)\s*$", line)
        if not m:
            continue
        if "Phy-ID" in m.group(2) or "NPU" in m.group(2):
            continue
        npu = int(m.group(1))
        parts = m.group(2).split()
        if not parts:
            continue

        affinity = None
        cpus = []
        relations = parts
        if has_affinity and len(parts) >= 2:
            affinity = parts[-1]
            cpus = parse_cpulist(affinity)
            if cpus:
                relations = parts[:-1]
            else:
                affinity = None
        if not cpus:
            affinity = None

        hccs = {}
        for idx, rel in enumerate(relations):
            hccs[idx] = rel
        out[npu] = {
            "id": npu,
            "affinity": affinity,
            "affinity_cpus": cpus,
            "hccs": hccs,
        }
    return out


def parse_info_l(text):
    """解析 npu-smi info -l（封装级权威计数）。

    -l 的 "NPU ID" 语义为物理封装（双 die 机型也只返回封装数），
    Total Count 为封装总数，每封装附 Product Name 与 Chip Count。
    """
    result = {"total_count": 0, "npus": []}
    current = None
    for line in (text or "").splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if key == "Total Count":
            try:
                result["total_count"] = int(value)
            except ValueError:
                pass
        elif key == "NPU ID":
            if current is not None:
                result["npus"].append(current)
            try:
                current = {"npu": int(value), "name": "", "chips": 0}
            except ValueError:
                current = None
        elif key == "Product Name" and current is not None:
            current["name"] = value
        elif key == "Chip Count" and current is not None:
            try:
                current["chips"] = int(value)
            except ValueError:
                pass
    if current is not None:
        result["npus"].append(current)
    return result


def parse_info_m(text):
    """解析 npu-smi info -m（die 级映射）。

    列序随版本不同：新版 NPU ID | Chip ID | Chip Logic ID | Chip Phy-ID | Chip Name，
    旧版（910B2）无 Chip Phy-ID 列且名称含空格。Chip Logic ID 即软件设备号
    （对应 /dev/davinciN）；Mcu 行与表头行过滤。
    返回 {"dies": [...], "groups": [[0,1],...], "logic_ids": [...]}；
    groups 按封装聚合，同封装 die 对即绑核不可拆分单位（双 die 机型 2 个一组）。
    """
    dies = []
    for line in (text or "").splitlines():
        tokens = line.split()
        if len(tokens) < 4:
            continue
        try:
            npu_id, chip_id, logic_id = int(tokens[0]), int(tokens[1]), int(tokens[2])
        except ValueError:
            continue  # 表头 / Mcu 行（'-'）
        phy_id = None
        if len(tokens) > 3:
            try:
                phy_id = int(tokens[3])
            except ValueError:
                phy_id = None
        name = " ".join(tokens[4:] if phy_id is not None else tokens[3:])
        if not name or name.upper().startswith("MCU"):
            continue
        dies.append({"npu": npu_id, "chip": chip_id, "logic_id": logic_id,
                     "phy_id": phy_id if phy_id is not None else logic_id,
                     "name": name})
    groups = []
    for npu_id in sorted({d["npu"] for d in dies}):
        logic = sorted(d["logic_id"] for d in dies if d["npu"] == npu_id)
        if logic:
            groups.append(logic)
    return {"dies": dies, "groups": groups,
            "logic_ids": sorted(d["logic_id"] for d in dies)}


def topo_meta(text):
    """拓扑输出形态判定：是否带 CPU Affinity、关系符号、互联形态提示。

    关系符号语义：
    - UB：NPU 间走灵衢（Unified Bus），A5/950 超节点常见；
    - HCCS / HCCS_SW / SIO：昇腾卡间高速互联（含经交换）；
    - PHB / PIX / PXB / SYS：PCIe 通路（同/跨 NUMA）。
    注意：UB 与「是否输出 CPU Affinity」是两条独立信息——UB 描述 NPU 间平面，
    CPU Affinity 描述 NPU 主机口到本机 CPU 的就近关系，UB 形态仍可能给出亲和列。
    """
    text = text or ""
    # Legend 段会列出全部关系符号的含义（含未实际出现的 SYS/PHB 等），
    # 只从矩阵正文判定关系，避免把 PCIe 层级误判成本机 fabric
    body = text.split("Legend:")[0]
    rels = set(re.findall(r"\b(HCCS_SW|HCCS|SIO|UB|PHB|PIX|PXB|SYS)\b", body))
    has_affinity = "CPU Affinity" in body
    if has_affinity:
        fmt = "affinity"
    elif "Phy-ID" in text:
        fmt = "phy_matrix"
    elif "NPU" in text:
        fmt = "npuid_matrix"
    else:
        fmt = "unknown"
    if "UB" in rels:
        fabric = "ub"
    elif rels & {"SIO", "HCCS_SW"}:
        fabric = "hccs_supernode"
    elif "HCCS" in rels:
        fabric = "hccs"
    elif rels & {"PHB", "PIX", "PXB", "SYS"}:
        fabric = "pcie"
    else:
        fabric = "unknown"
    return {
        "format": fmt,
        "has_affinity": has_affinity,
        "relations": sorted(rels),
        "fabric_hint": fabric,
        "has_nic_columns": "NIC" in text,
    }


def query():
    """读取 npu-smi 信息；不存在/失败时返回 available=False 与原因。"""
    result = {
        "available": False,
        "path": None,
        "info_ok": False,
        "topo_ok": False,
        "devices": [],
        "processes": [],
        "empty_npus": [],
        "topo": {},
        "topo_meta": {},
        "package_count": None,
        "die_map": {},
        "errors": [],
        "info_error_kind": None,
        "topo_error_kind": None,
        "package_error_detail": None,
        "die_error_detail": None,
    }
    path = find_npu_smi()
    result["path"] = path
    if not path:
        result["errors"].append("npu-smi not found")
        result["info_error_kind"] = "not_found"
        return result

    result["available"] = True
    rc, out, err = _run([path, "info"])
    if rc != 0:
        detail = (err or out).strip()
        result["errors"].append("npu-smi info failed: %s" % detail)
        result["info_error_kind"] = _classify_error(detail)
        return result
    result["info_ok"] = True
    info = parse_info(out)
    result.update(info)

    rc, out, err = _run([path, "info", "-t", "topo"])
    if rc == 0:
        result["topo_ok"] = True
        result["topo"] = parse_topo(out)
        result["topo_meta"] = topo_meta(out)
    else:
        detail = (err or out).strip()
        result["errors"].append("npu-smi info -t topo failed: %s" % detail)
        result["topo_error_kind"] = _classify_error(detail)

    # 封装级权威计数（可选增强，缺失不阻断；保留 stderr 供排障）
    rc, out, err = _run([path, "info", "-l"])
    if rc == 0:
        result["package_count"] = parse_info_l(out).get("total_count") or None
    else:
        detail = (err or out).strip()[:400]
        result["package_error_detail"] = detail or None
        _LOG.debug("npu-smi info -l failed: rc=%s %s", rc, detail)

    # die 级映射（双 die 机型的分组依据，缺失时不臆造 die 信息）
    rc, out, err = _run([path, "info", "-m"])
    if rc == 0:
        result["die_map"] = parse_info_m(out)
    else:
        detail = (err or out).strip()[:400]
        result["die_error_detail"] = detail or None
        _LOG.debug("npu-smi info -m failed: rc=%s %s", rc, detail)

    return result


def primary_processes(processes):
    """按 pid 聚合，选取进程内存最大的 NPU 作为 primary_npu。

    同一进程可能在多张卡上有少量控制内存（如 117MB），
    以大内存条目（如 48GB）作为主卡，避免把控制内存误判为多卡占用。
    """
    by_pid = {}
    for p in processes:
        by_pid.setdefault(p["pid"], []).append(p)

    out = []
    for pid, items in by_pid.items():
        primary = max(items, key=lambda x: x["mem_mb"])
        out.append({
            "pid": pid,
            "name": primary["name"],
            "primary_npu": primary["npu"],
            "primary_mem_mb": primary["mem_mb"],
            "npus": sorted({x["npu"] for x in items}),
            "entries": items,
        })
    return sorted(out, key=lambda x: x["primary_npu"])


def unknown_hbm_npus(npu_info):
    """有 HBM 占用但无可见进程的 NPU：占用者进程不可见（容器/权限）时的唯一线索。

    discover 与 preflight 共用本实现，避免同一判定两处维护。
    """
    busy = {}
    for p in npu_info.get("processes") or []:
        busy[p["npu"]] = busy.get(p["npu"], 0) + p.get("mem_mb", 0)
    out = []
    for dev in npu_info.get("devices") or []:
        used = dev.get("hbm_used_mb") or 0
        if used > 0 and busy.get(dev["id"], 0) == 0:
            out.append({"npu": dev["id"], "hbm_used_mb": used})
    return out


if __name__ == "__main__":
    import json
    q = query()
    q["primary_processes"] = primary_processes(q["processes"])
    print(json.dumps(q, ensure_ascii=False, indent=2))
