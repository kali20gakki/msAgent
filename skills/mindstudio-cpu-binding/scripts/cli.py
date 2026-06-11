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

"""Command line entry point for the minimum runnable loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collect import CollectConfig, collect_snapshot, load_npu_map
from scripts.diagnose import diagnose
from scripts.planner import generate_plan
from scripts.process_discovery import discover_processes_from_text
from scripts.report import render_report
from scripts.snapshot import load_snapshot, write_json
from scripts.topology_collect import collect_topology_from_text


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mindstudio-cpu-binding minimum runnable loop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze a Host CPU Snapshot and generate plan/report outputs")
    analyze.add_argument("--snapshot", required=True, help="Path to snapshot JSON")
    analyze.add_argument("--out", required=True, help="Output directory")
    analyze.add_argument(
        "--executor",
        choices=["dry-run", "taskset", "internal-script"],
        default="dry-run",
        help="Execution backend to preview in the plan",
    )

    collect_topology = subparsers.add_parser(
        "collect-topology", help="Collect or parse CPU/NPU/NUMA topology into JSON"
    )
    collect_topology.add_argument(
        "--lscpu-file",
        help="Read lscpu output from a text file instead of running lscpu",
    )
    collect_topology.add_argument(
        "--npu-smi-topo-file",
        help="Read npu-smi topo output from a text file instead of running npu-smi",
    )
    collect_topology.add_argument("--out", required=True, help="Output topology JSON path")

    discover_processes = subparsers.add_parser(
        "discover-processes",
        help="Discover candidate NPU/PyTorch/LLM serving processes",
    )
    discover_processes.add_argument("--ps-file", help="Read ps output from a text file instead of running ps")
    discover_processes.add_argument(
        "--npu-smi-info-file",
        help="Read npu-smi info output from a text file instead of running npu-smi",
    )
    discover_processes.add_argument("--keyword", help="Additional keyword substring to include matching processes")
    discover_processes.add_argument("--out", required=True, help="Output process discovery JSON path")

    collect = subparsers.add_parser("collect", help="Collect a full Host CPU Snapshot")
    collect.add_argument(
        "--pid",
        action="append",
        type=int,
        required=True,
        help="Target PID; may be repeated",
    )
    collect.add_argument("--scenario", choices=["training", "inference", "unknown"], default="unknown")
    collect.add_argument("--framework", default="pytorch")
    collect.add_argument("--device-type", default="npu")
    collect.add_argument(
        "--optimization-goal",
        choices=["throughput", "latency", "stability", "isolation", "unknown"],
        default="unknown",
    )
    collect.add_argument(
        "--process-model",
        choices=["single-process", "multi-rank", "multi-instance", "unknown"],
        default="unknown",
    )
    collect.add_argument("--rank-map", help="Comma-separated rank0=123:npu0 mappings")
    collect.add_argument(
        "--deployment",
        choices=["baremetal", "docker", "kubernetes", "slurm", "unknown"],
        default="unknown",
    )
    collect.add_argument("--container-pid-mode", choices=["auto", "host", "container"], default="auto")
    collect.add_argument(
        "--extra-keywords",
        nargs="*",
        default=[],
        help="Additional key process/thread substrings",
    )
    collect.add_argument("--sample-seconds", type=_positive_float, default=10.0)
    collect.add_argument("--top-threads", type=_non_negative_int, default=10)
    collect.add_argument("--torch-num-threads", type=_non_negative_int)
    collect.add_argument("--torch-num-interop-threads", type=_non_negative_int)
    collect.add_argument("--dataloader-workers", type=_non_negative_int)
    collect.add_argument("--dataloader-pin-memory", choices=["true", "false"])
    collect.add_argument("--dataloader-prefetch-factor", type=int)
    collect.add_argument("--npu-map", help="JSON file with device_id -> mapping overrides")
    collect.add_argument("--no-runtime-sample", action="store_true")
    collect.add_argument("--no-raw", action="store_true")
    collect.add_argument("--out", required=True, help="Output snapshot JSON path")
    collect.add_argument("--raw-dir", help="Directory for raw command outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "collect-topology":
        return _collect_topology(args)
    if args.command == "discover-processes":
        return _discover_processes(args)
    if args.command == "collect":
        return _collect(args)
    raise ValueError(f"unsupported command: {args.command}")


def _analyze(args: argparse.Namespace) -> int:
    try:
        snapshot = load_snapshot(args.snapshot)
        findings = diagnose(snapshot)
        plan = generate_plan(snapshot, findings, executor_backend=args.executor)

        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "plan.json", plan)
        render_report(snapshot, findings, plan, output_dir / "report.html")
    except (OSError, ValueError, KeyError) as exc:
        print(f"analyze error: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {output_dir / 'plan.json'}")
    print(f"Generated {output_dir / 'report.html'}")
    return 0


def _collect_topology(args: argparse.Namespace) -> int:
    if not args.lscpu_file:
        print(
            "collect-topology error: --lscpu-file is required by the shared CLI; "
            "run scripts/topology_collect.py for live collection",
            file=sys.stderr,
        )
        return 1
    lscpu_text = Path(args.lscpu_file).read_text(encoding="utf-8")
    npu_smi_topo_text = Path(args.npu_smi_topo_file).read_text(encoding="utf-8") if args.npu_smi_topo_file else ""
    topology = collect_topology_from_text(lscpu_text, npu_smi_topo_text)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, topology)
    print(f"Generated {output}")
    return 0


def _discover_processes(args: argparse.Namespace) -> int:
    if not args.ps_file:
        print(
            "discover-processes error: --ps-file is required by the shared CLI; "
            "run scripts/process_discovery.py for live discovery",
            file=sys.stderr,
        )
        return 1
    ps_text = Path(args.ps_file).read_text(encoding="utf-8")
    npu_smi_info_text = Path(args.npu_smi_info_file).read_text(encoding="utf-8") if args.npu_smi_info_file else ""
    discovery = discover_processes_from_text(ps_text, npu_smi_info_text, keyword=args.keyword)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, discovery)
    print(f"Generated {output}")
    return 0


def _collect(args: argparse.Namespace) -> int:
    output = Path(args.out)
    try:
        snapshot = collect_snapshot(
            CollectConfig(
                pids=args.pid,
                out=output,
                scenario=args.scenario,
                framework=args.framework,
                device_type=args.device_type,
                optimization_goal=args.optimization_goal,
                process_model=args.process_model,
                deployment=args.deployment,
                container_pid_mode=args.container_pid_mode,
                rank_map=args.rank_map,
                extra_keywords=args.extra_keywords,
                sample_seconds=args.sample_seconds,
                top_threads=args.top_threads,
                torch_num_threads=args.torch_num_threads,
                torch_num_interop_threads=args.torch_num_interop_threads,
                dataloader_workers=args.dataloader_workers,
                dataloader_pin_memory=_parse_optional_bool(args.dataloader_pin_memory),
                dataloader_prefetch_factor=args.dataloader_prefetch_factor,
                npu_map=load_npu_map(args.npu_map),
                no_runtime_sample=args.no_runtime_sample,
                no_raw=args.no_raw,
                raw_dir=Path(args.raw_dir) if args.raw_dir else None,
            )
        )
    except FileNotFoundError as exc:
        print(f"collect error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"collect error: {exc}", file=sys.stderr)
        return 1
    availability = snapshot["availability"]
    print(f"Generated {output}")
    print(
        "Availability: "
        f"complete={availability['complete']} "
        f"missing={len(availability['missing'])} "
        f"partial={len(availability['partial'])} "
        f"errors={len(availability['errors'])}"
    )
    return 0


def _parse_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
