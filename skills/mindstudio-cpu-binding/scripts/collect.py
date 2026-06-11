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

"""Build a Host CPU Snapshot by composing collector modules."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.collectors.availability import Availability
from scripts.collectors.key_processes import collect_key_processes
from scripts.collectors.linux_cgroup import CgroupReader, collect_cgroup_for_pid
from scripts.collectors.linux_cpu import (
    collect_cpu_topology,
    collect_distance_matrix,
    collect_smt_siblings,
)
from scripts.collectors.linux_proc import ProcReader, collect_process
from scripts.collectors.npu_topology import collect_npu_topology
from scripts.collectors.runtime_env import collect_pytorch_env
from scripts.collectors.runtime_sample import (
    ProcSampleReader,
    collect_proc_runtime_sample,
    merge_runtime_sample,
)
from scripts.cpulist import parse_cpu_list
from scripts.snapshot import write_json
from scripts.topology_collect import parse_lscpu


@dataclass
class CollectConfig:
    pids: list[int]
    out: Path
    scenario: str = "unknown"
    framework: str = "pytorch"
    device_type: str = "npu"
    optimization_goal: str = "unknown"
    process_model: str = "unknown"
    deployment: str = "unknown"
    container_pid_mode: str = "auto"
    rank_map: str | None = None
    extra_keywords: list[str] = field(default_factory=list)
    sample_seconds: float = 10.0
    top_threads: int = 10
    torch_num_threads: int | None = None
    torch_num_interop_threads: int | None = None
    dataloader_workers: int | None = None
    dataloader_pin_memory: bool | None = None
    dataloader_prefetch_factor: int | None = None
    npu_map: dict[str, dict[str, Any]] | None = None
    no_runtime_sample: bool = False
    no_raw: bool = False
    raw_dir: Path | None = None
    proc_root: Path = Path("/proc")
    cgroup_root: Path = Path("/sys/fs/cgroup")
    sys_root: Path = Path("/sys")
    lscpu_text: str | None = None
    lscpu_e_text: str | None = None
    npu_smi_info_text: str | None = None
    npu_smi_topo_text: str | None = None


def collect_snapshot(config: CollectConfig) -> dict[str, Any]:
    availability = Availability()
    _validate_pids(config)

    raw_refs: dict[str, Any] = {}
    raw_dir = config.raw_dir or config.out.parent / "raw"
    lscpu_text = _get_text(
        config.lscpu_text,
        ["lscpu"],
        optional=False,
        availability=availability,
        component="lscpu",
    )
    lscpu_e_text = _get_text(
        config.lscpu_e_text,
        ["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ,MHZ"],
        optional=True,
        availability=availability,
        component="lscpu-e",
    )
    npu_smi_info_text = _get_text(
        config.npu_smi_info_text,
        ["npu-smi", "info"],
        optional=True,
        availability=availability,
        component="npu-smi-info",
    )
    npu_smi_topo_text = _get_text(
        config.npu_smi_topo_text,
        ["npu-smi", "info", "-t", "topo"],
        optional=True,
        availability=availability,
        component="npu-smi-topo",
    )

    if not config.no_raw:
        raw_refs.update(_write_raw_texts(raw_dir, lscpu_text, lscpu_e_text, npu_smi_info_text, npu_smi_topo_text))

    system, numa_topology = parse_lscpu(lscpu_text)
    cpu_topology = collect_cpu_topology(lscpu_e_text, collect_smt_siblings(config.sys_root))
    numa_topology["distance_matrix"] = collect_distance_matrix(config.sys_root)
    if not numa_topology["distance_matrix"]:
        availability.add_partial("numa_topology.distance_matrix")

    npu_topology, npu_availability = collect_npu_topology(
        npu_smi_info_text,
        npu_smi_topo_text,
        numa_topology["nodes"],
        npu_map=config.npu_map,
        sys_root=config.sys_root,
    )
    availability.merge(npu_availability)

    proc_reader = ProcReader(config.proc_root)
    processes = []
    for pid in config.pids:
        process, proc_availability = collect_process(proc_reader, pid)
        availability.merge(proc_availability)
        if process is not None:
            processes.append(process)

    rank_mapping = _parse_rank_map(config.rank_map)
    _apply_rank_map(processes, rank_mapping)
    _annotate_thread_numa(processes, numa_topology["nodes"])

    cgroup = _collect_cgroups(config, availability)
    pytorch = _collect_runtime_env(config, processes, availability)
    runtime_sample = _collect_runtime_sample(config, processes, numa_topology["nodes"], availability)
    key_processes = collect_key_processes(
        processes,
        npu_topology=npu_topology,
        target_pids=config.pids,
        extra_keywords=config.extra_keywords,
        proc_root=config.proc_root,
    )

    snapshot = {
        "schema_version": "0.1.1",
        "collection": {
            "collector": "mindstudio-cpu-binding",
            "mode": "snapshot",
        },
        "workload": {
            "scenario": config.scenario,
            "framework": config.framework,
            "device_type": config.device_type,
            "optimization_goal": config.optimization_goal,
            "process_model": config.process_model,
            "deployment": config.deployment,
            "container_pid_mode": config.container_pid_mode,
            "target_pids": list(config.pids),
            "rank_mapping": rank_mapping,
        },
        "system": system,
        "cpu_topology": cpu_topology,
        "numa_topology": numa_topology,
        "npu_topology": npu_topology,
        "processes": processes,
        "cgroup": cgroup,
        "pytorch": pytorch,
        "key_processes": key_processes,
        "runtime_sample": runtime_sample,
        "availability": availability.to_dict(),
        "raw_refs": raw_refs,
    }
    write_json(config.out, snapshot)
    return snapshot


def _validate_pids(config: CollectConfig) -> None:
    missing = [pid for pid in config.pids if not (config.proc_root / str(pid)).is_dir()]
    if missing:
        raise FileNotFoundError(f"PID(s) not found: {missing}")


def _get_text(  # pylint: disable=duplicate-code
    injected: str | None,
    command: list[str],
    optional: bool,
    availability: Availability,
    component: str,
) -> str:
    if injected is not None:
        return injected
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=15)  # nosec B603
        return output.decode("utf-8", errors="ignore")
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        if optional:
            availability.add_partial(component)
            return ""
        availability.add_error(component, str(exc))
        raise


def _write_raw_texts(raw_dir: Path, lscpu_text: str, lscpu_e_text: str, npu_info: str, npu_topo: str) -> dict[str, Any]:
    refs = {
        "lscpu": "lscpu.txt",
        "lscpu_e": "lscpu-e.txt",
        "npu_smi_info": "npu-smi-info.txt" if npu_info else None,
        "npu_smi_topo": "npu-smi-topo.txt" if npu_topo else None,
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "lscpu.txt").write_text(lscpu_text, encoding="utf-8")
    (raw_dir / "lscpu-e.txt").write_text(lscpu_e_text, encoding="utf-8")
    if npu_info:
        (raw_dir / "npu-smi-info.txt").write_text(npu_info, encoding="utf-8")
    if npu_topo:
        (raw_dir / "npu-smi-topo.txt").write_text(npu_topo, encoding="utf-8")
    return refs


def _collect_cgroups(config: CollectConfig, availability: Availability) -> dict[str, Any]:
    reader = CgroupReader(cgroup_root=config.cgroup_root, proc_root=config.proc_root)
    version = None
    groups = []
    for pid in config.pids:
        cgroup, cgroup_availability = collect_cgroup_for_pid(reader, pid)
        availability.merge(cgroup_availability)
        if cgroup is None:
            continue
        version = version or cgroup.get("version")
        groups.extend(cgroup.get("process_groups", []))
    return {"version": version, "process_groups": groups}


def _collect_runtime_env(
    config: CollectConfig, processes: list[dict[str, Any]], availability: Availability
) -> dict[str, Any]:
    if not processes:
        availability.add_missing("pytorch")
        return collect_pytorch_env("", "", None, None, None, None, None)["pytorch"]

    process = processes[0]
    pid = int(process["pid"])
    environ_path = config.proc_root / str(pid) / "environ"
    try:
        environ = environ_path.read_text(encoding="utf-8", errors="ignore")
    except (FileNotFoundError, PermissionError, OSError):
        availability.add_partial(f"pytorch.env[{pid}]")
        environ = ""
    return collect_pytorch_env(
        process.get("command") or "",
        environ,
        config.torch_num_threads,
        config.torch_num_interop_threads,
        config.dataloader_workers,
        config.dataloader_pin_memory,
        config.dataloader_prefetch_factor,
    )["pytorch"]


def _collect_runtime_sample(
    config: CollectConfig,
    processes: list[dict[str, Any]],
    numa_nodes: list[dict[str, Any]],
    availability: Availability,
) -> dict[str, Any]:
    if config.no_runtime_sample:
        return _empty_runtime_sample()
    try:
        reader = ProcSampleReader(config.proc_root)
        before = reader.snapshot(config.pids)
        time.sleep(config.sample_seconds)
        after = reader.snapshot(config.pids)
        sample, sample_availability = collect_proc_runtime_sample(
            before,
            after,
            sample_seconds=config.sample_seconds,
            clock_ticks=getattr(os, "sysconf")("SC_CLK_TCK"),
            top_threads=config.top_threads,
            cpu_to_numa=_cpu_to_numa(numa_nodes),
            loadavg_text=reader.read_loadavg(),
        )
    except (OSError, ValueError) as exc:
        availability.add_partial("runtime_sample")
        availability.add_error("runtime_sample", str(exc))
        return _empty_runtime_sample()
    availability.merge(sample_availability)
    merge_runtime_sample(processes, sample)
    return sample


def _empty_runtime_sample() -> dict[str, Any]:
    return {
        "sample_seconds": 0,
        "process_cpu_percent_total": None,
        "process_cpu_percent_by_pid": {},
        "system_loadavg": [],
        "cpu_usage_by_numa": [],
        "threads": [],
        "top_threads": [],
        "cpu_migration_observed": False,
    }


def _annotate_thread_numa(processes: list[dict[str, Any]], numa_nodes: list[dict[str, Any]]) -> None:
    cpu_to_numa = _cpu_to_numa(numa_nodes)
    for process in processes:
        cpu = process.get("current_cpu")
        process["numa_node"] = cpu_to_numa.get(cpu) if cpu is not None else None
        for thread in process.get("threads", []):
            current_cpu = thread.get("current_cpu")
            thread["numa_node"] = cpu_to_numa.get(current_cpu) if current_cpu is not None else None


def _cpu_to_numa(numa_nodes: list[dict[str, Any]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for node in numa_nodes:
        node_id = node.get("node")
        if node_id is None:
            continue
        try:
            cpus = parse_cpu_list(node.get("cpus"))
        except ValueError:
            continue
        for cpu in cpus:
            result[cpu] = int(node_id)
    return result


def _parse_rank_map(rank_map: str | None) -> list[dict[str, Any]]:
    if not rank_map:
        return []
    result = []
    for entry in rank_map.split(","):
        rank_part, separator, rest = entry.partition("=")
        if not separator:
            raise ValueError(f"invalid rank-map entry: missing equals sign in {entry}")
        if not rank_part:
            raise ValueError(f"invalid rank-map entry: missing rank name in {entry}")
        pid_part, colon, npu_part = rest.partition(":")
        if not pid_part:
            raise ValueError(f"invalid rank-map entry: missing pid in {entry}")
        if not colon:
            raise ValueError(f"invalid rank-map entry: missing colon before NPU device in {entry}")
        try:
            pid = int(pid_part)
        except ValueError as exc:
            raise ValueError(f"invalid rank-map entry: PID must be integer in {entry}") from exc
        result.append({"rank": rank_part, "pid": pid, "npu_device": npu_part or None})
    return result


def _apply_rank_map(processes: list[dict[str, Any]], rank_mapping: list[dict[str, Any]]) -> None:
    by_pid = {int(item["pid"]): item for item in rank_mapping}
    for process in processes:
        mapping = by_pid.get(int(process["pid"]))
        if mapping is None:
            continue
        process["rank"] = mapping.get("rank")
        process["npu_device"] = mapping.get("npu_device")


def load_npu_map(path: str | None) -> dict[str, dict[str, Any]] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("npu-map must be a JSON object")
    return {str(key): value for key, value in data.items() if isinstance(value, dict)}
