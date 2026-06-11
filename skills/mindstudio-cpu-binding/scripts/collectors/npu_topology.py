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

"""Collect Ascend NPU topology and NUMA mapping details."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from scripts.collectors.availability import Availability
from scripts.topology_collect import parse_npu_smi_topo


def parse_npu_smi_info(text: str) -> dict[str, Any]:
    devices: dict[str, dict[str, Any]] = {}
    processes: dict[str, list[int]] = {}
    pending_device_id: str | None = None
    in_process_section = False

    for raw_line in text.splitlines():
        cells = _pipe_cells(raw_line)
        if not cells:
            continue
        if any(cell == "Process id" for cell in cells):
            in_process_section = True
            continue
        if cells[0] == "NPU" or cells[0].startswith("NPU "):
            continue
        if cells[0].startswith("No running processes found"):
            continue

        if in_process_section:
            process = _parse_process_row(cells)
            if process is None:
                continue
            device_id, pid = process
            processes.setdefault(device_id, []).append(pid)
            continue

        device_id = _parse_device_id(cells[0])
        if device_id is not None and len(cells) >= 2 and not _is_chip_row(cells[0]):
            devices.setdefault(device_id, {})
            devices[device_id].update(
                {
                    "device_id": device_id,
                    "logical_id": device_id,
                    "model": _parse_device_model(cells[0]),
                    "health": _first_token(cells[1]) if len(cells) > 1 else "unknown",
                }
            )
            pending_device_id = device_id
            continue

        if pending_device_id is not None and len(cells) >= 2:
            bus_id = _parse_pci_bus_id(cells[1])
            if bus_id:
                devices.setdefault(pending_device_id, {})["pci_bus_id"] = bus_id
                pending_device_id = None

    return {"devices": devices, "processes": processes}


def collect_npu_topology(
    npu_smi_info_text: str,
    npu_smi_topo_text: str,
    numa_nodes: list[dict[str, Any]],
    npu_map: dict[str, dict[str, Any]] | None = None,
    sys_root: Path = Path("/sys"),
) -> tuple[dict[str, Any], Availability]:
    availability = Availability()
    info = parse_npu_smi_info(npu_smi_info_text)
    topo, topo_missing = parse_npu_smi_topo(npu_smi_topo_text, numa_nodes)
    for field in topo_missing:
        availability.add_missing(field)

    devices_by_id = _merge_devices(info["devices"], topo.get("devices", []))
    npu_map = npu_map or {}
    for device_id, override in npu_map.items():
        devices_by_id.setdefault(str(device_id), {"device_id": str(device_id), "logical_id": str(device_id)})
        devices_by_id[str(device_id)].update(override)

    devices: list[dict[str, Any]] = []
    for device_id in sorted(devices_by_id, key=_device_sort_key):
        device = devices_by_id[device_id]
        pci_bus_id = device.get("pci_bus_id") or ""
        user_override = npu_map.get(device_id, {})
        if "numa_node" in user_override:
            numa_node = _safe_int(user_override.get("numa_node"))
            numa_source = "user_map"
        else:
            numa_node = _read_pci_numa_node(sys_root, pci_bus_id)
            numa_source = "pci" if numa_node is not None else "topo_affinity"
            if numa_node is None:
                numa_node = _safe_int(device.get("numa_node"))
                if numa_node is not None:
                    availability.add_partial(f"npu_topology.devices[{device_id}].numa_node")

        if numa_node is None:
            numa_source = None
            availability.add_missing(f"npu_topology.devices[{device_id}].numa_node")

        devices.append(
            {
                "device_id": device_id,
                "logical_id": str(device["logical_id"] if device.get("logical_id") is not None else device_id),
                "pci_bus_id": str(pci_bus_id),
                "model": device.get("model"),
                "numa_node": numa_node,
                "numa_source": numa_source,
                "local_cpus": str(device.get("local_cpus") or ""),
                "health": device.get("health") or "unknown",
                "links": list(device.get("links") or []),
            }
        )

    if not devices:
        availability.add_missing("npu_topology.devices")

    return (
        {
            "vendor": "ascend",
            "devices": devices,
            "processes_by_device": info["processes"],
            "source": "npu-smi info + npu-smi info -t topo",
        },
        availability,
    )


def _merge_devices(
    info_devices: dict[str, dict[str, Any]], topo_devices: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result = {str(device_id): dict(device) for device_id, device in info_devices.items()}
    for topo_device in topo_devices:
        device_id = str(topo_device.get("device_id"))
        result.setdefault(device_id, {"device_id": device_id, "logical_id": device_id})
        for key, value in topo_device.items():
            current = result[device_id].get(key)
            if current is None or current == "" or current == []:
                result[device_id][key] = value
    return result


def _pipe_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _parse_process_row(cells: list[str]) -> tuple[str, int] | None:
    if len(cells) < 2:
        return None
    device_id = _parse_device_id(cells[0])
    pid = _safe_int(cells[1])
    if device_id is None or pid is None:
        return None
    return device_id, pid


def _is_chip_row(value: str) -> bool:
    return _parse_device_id(value) == "0" and value.strip() == "0"


def _parse_device_id(value: str) -> str | None:
    match = re.match(r"\s*(\d+)\b", value)
    if not match:
        return None
    return match.group(1)


def _parse_device_model(value: str) -> str | None:
    parts = value.split()
    return parts[1] if len(parts) > 1 else None


def _parse_pci_bus_id(value: str) -> str | None:
    match = re.search(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F]", value)
    if not match:
        return None
    return match.group(0).upper()


def _read_pci_numa_node(sys_root: Path, pci_bus_id: str) -> int | None:
    if not pci_bus_id:
        return None
    value = _read_int(sys_root / "bus" / "pci" / "devices" / pci_bus_id / "numa_node")
    if value is None:
        value = _read_int(sys_root / "bus" / "pci" / "devices" / pci_bus_id.lower() / "numa_node")
    if value is None:
        return None
    if value < 0:
        return None
    return value


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None


def _first_token(value: str) -> str | None:
    parts = value.split()
    return parts[0] if parts else None


def _device_sort_key(device_id: str) -> tuple[int, str]:
    value = _safe_int(device_id)
    if value is None:
        return 10**9, device_id
    return value, device_id


def _safe_int(  # pylint: disable=duplicate-code
    value: Any,
) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
