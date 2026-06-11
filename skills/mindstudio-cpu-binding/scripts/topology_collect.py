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

"""Topology collection/parser prototype for mindstudio-cpu-binding."""

from __future__ import annotations

import argparse
import platform
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cpulist import count_cpu_list, numa_nodes_for_cpu_list
from scripts.script_utils import read_or_run, write_json_output


def collect_topology_from_text(lscpu_text: str, npu_smi_topo_text: str | None = None) -> dict[str, Any]:
    system, numa_topology = parse_lscpu(lscpu_text)
    npu_topology, npu_missing = parse_npu_smi_topo(npu_smi_topo_text or "", numa_topology["nodes"])
    missing = []
    if not numa_topology["nodes"]:
        missing.append("numa_topology.nodes")
    missing.extend(npu_missing)
    return {
        "system": system,
        "numa_topology": numa_topology,
        "npu_topology": npu_topology,
        "availability": {
            "complete": not missing,
            "missing": missing,
            "partial": [],
            "errors": [],
        },
        "raw_refs": {
            "lscpu": "text",
            "npu_smi_topo": "text" if npu_smi_topo_text else None,
        },
    }


def parse_lscpu(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = _parse_lscpu_fields(text)
    total_logical_cpus = _int_or_none(fields.get("CPU(s)"))
    threads_per_core = _int_or_none(fields.get("Thread(s) per core"))
    cores_per_socket = _int_or_none(fields.get("Core(s) per socket"))
    sockets = _int_or_none(fields.get("Socket(s)"))
    total_physical_cores = None
    if cores_per_socket is not None and sockets is not None:
        total_physical_cores = cores_per_socket * sockets

    numa_nodes = []
    for key, value in fields.items():
        match = re.fullmatch(r"NUMA node(\d+) CPU\(s\)", key)
        if not match:
            continue
        cpus = value.strip()
        logical_count = count_cpu_list(cpus)
        physical_count = logical_count
        if threads_per_core and threads_per_core > 0:
            physical_count = logical_count // threads_per_core
        numa_nodes.append(
            {
                "node": int(match.group(1)),
                "cpus": cpus,
                "logical_cpu_count": logical_count,
                "physical_core_count": physical_count,
            }
        )

    return (
        {
            "os": "Linux",
            "kernel": platform.release(),
            "architecture": fields.get("Architecture"),
            "cpu_model": fields.get("Model name") or fields.get("Vendor ID"),
            "total_logical_cpus": total_logical_cpus,
            "total_physical_cores": total_physical_cores,
            "sockets": sockets,
            "smt_enabled": bool(threads_per_core and threads_per_core > 1),
            "online_cpus": fields.get("On-line CPU(s) list"),
            "isolated_cpus": None,
        },
        {
            "nodes": sorted(numa_nodes, key=lambda node: node["node"]),
            "distance_matrix": [],
        },
    )


def parse_npu_smi_topo(text: str, numa_nodes: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return {"vendor": "ascend", "devices": [], "source": "npu-smi info -t topo"}, ["npu_topology.devices"]

    header = lines[0].split()
    npu_columns = _npu_columns(header)
    affinity_index = _affinity_index(header)
    devices: dict[str, dict[str, Any]] = {}
    links_by_device: dict[str, list[dict[str, str]]] = {}
    missing = []

    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        device_id = _parse_npu_id(parts[0])
        if device_id is None:
            continue
        affinity_part_index = affinity_index + 1 if affinity_index is not None else None
        affinity = (
            parts[affinity_part_index] if affinity_part_index is not None and affinity_part_index < len(parts) else ""
        )
        numa_node = _numa_for_affinity(affinity, numa_nodes)
        if numa_node is None:
            missing.append(f"npu_topology.devices[{device_id}].numa_node")
        devices[device_id] = {
            "device_id": device_id,
            "logical_id": device_id,
            "pci_bus_id": "",
            "numa_node": numa_node,
            "local_cpus": affinity,
            "health": "unknown",
            "links": [],
        }
        links_by_device[device_id] = _parse_links(device_id, parts, npu_columns)

    for device_id, links in links_by_device.items():
        devices[device_id]["links"] = links

    return (
        {
            "vendor": "ascend",
            "devices": [devices[key] for key in sorted(devices, key=int)],
            "source": "npu-smi info -t topo",
        },
        _dedupe(missing),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    lscpu_text = read_or_run(args.lscpu_file, ["lscpu"])
    npu_smi_topo_text = read_or_run(args.npu_smi_topo_file, ["npu-smi", "info", "-t", "topo"], optional=True)
    result = collect_topology_from_text(lscpu_text, npu_smi_topo_text)
    output = write_json_output(args.out, result)
    print(f"Generated {output}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect or parse CPU/NPU/NUMA topology into mindstudio-cpu-binding JSON"
    )
    parser.add_argument(
        "--lscpu-file",
        help="Read lscpu output from a text file instead of running lscpu",
    )
    parser.add_argument(
        "--npu-smi-topo-file",
        help="Read npu-smi topo output from a text file instead of running npu-smi",
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    return parser.parse_args(argv)


def _parse_lscpu_fields(text: str) -> dict[str, str]:
    fields = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def _npu_columns(header: list[str]) -> dict[str, int]:
    columns = {}
    for index, column in enumerate(header):
        device_id = _parse_npu_id(column)
        if device_id is not None:
            columns[device_id] = index
    return columns


def _affinity_index(header: list[str]) -> int | None:
    for index in range(len(header) - 1):
        if header[index] == "CPU" and header[index + 1] == "Affinity":
            return index
    for index, column in enumerate(header):
        if column in {"Affinity", "NUMA"}:
            return index
    return None


def _parse_npu_id(value: str) -> str | None:
    match = re.fullmatch(r"(?:NPU|Phy-ID)(\d+)", value.strip())
    if not match:
        return None
    return match.group(1)


def _numa_for_affinity(affinity: str, numa_nodes: list[dict[str, Any]]) -> int | None:
    nodes = sorted(numa_nodes_for_cpu_list(affinity, numa_nodes))
    if not nodes:
        return None
    return nodes[0]


def _parse_links(source_device: str, parts: list[str], npu_columns: dict[str, int]) -> list[dict[str, str]]:
    links = []
    for target_device, column_index in npu_columns.items():
        if int(source_device) >= int(target_device):
            continue
        part_index = column_index + 1
        if part_index >= len(parts):
            continue
        link_type = parts[part_index]
        if link_type in {"X", "NA", "-"}:
            continue
        links.append({"target": target_device, "type": link_type})
    return links


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    match = re.search(r"\d+", value)
    if not match:
        return None
    return int(match.group(0))


def _dedupe(values: list[str]) -> list[str]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
