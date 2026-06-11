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

"""Non-interactive process discovery prototype for mindstudio-cpu-binding."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.collectors.sanitize import sanitize_command
from scripts.script_utils import read_or_run, write_json_output

ROLE_PATTERNS = [
    ("api_server", re.compile(r"api[_-]?server|openai", re.IGNORECASE)),
    ("scheduler", re.compile(r"scheduler|manager", re.IGNORECASE)),
    ("tokenizer", re.compile(r"tokenizer", re.IGNORECASE)),
    ("engine_worker", re.compile(r"worker|engine|local[_-]?rank|rank", re.IGNORECASE)),
    ("serving_main", re.compile(r"vllm|sglang|launch_server", re.IGNORECASE)),
    ("runtime", re.compile(r"dev\d+_sq(?:_task)?|hccl", re.IGNORECASE)),
]


def discover_processes_from_text(
    ps_text: str, npu_smi_info_text: str | None = None, keyword: str | None = None
) -> dict[str, Any]:
    all_processes = parse_ps_output(ps_text)
    npu_by_pid = parse_npu_smi_info(npu_smi_info_text or "")
    candidates = []
    for process in all_processes:
        if _is_kernel_thread(process):
            continue
        role_hint = _role_hint(process)
        is_npu_process = process["pid"] in npu_by_pid
        matches_keyword = _matches_keyword(process, keyword)
        if role_hint == "unknown" and not is_npu_process and not matches_keyword:
            continue
        enriched = dict(process)
        enriched["role_hint"] = role_hint if role_hint != "unknown" else "npu_process"
        enriched["npu_device"] = npu_by_pid.get(process["pid"])
        enriched["children"] = []
        candidates.append(enriched)

    candidate_pids = {process["pid"] for process in candidates}
    for process in candidates:
        process["children"] = [child["pid"] for child in candidates if child["ppid"] == process["pid"]]
        if process["ppid"] not in candidate_pids:
            process["tree_root"] = True
        else:
            process["tree_root"] = False

    return {
        "schema_version": "0.1.0",
        "summary": {
            "candidate_process_count": len(candidates),
            "npu_process_count": sum(1 for process in candidates if process.get("npu_device") is not None),
            "tree_root_count": sum(1 for process in candidates if process.get("tree_root")),
        },
        "processes": sorted(candidates, key=lambda process: process["pid"]),
        "availability": {"complete": True, "missing": [], "partial": [], "errors": []},
    }


def parse_ps_output(text: str) -> list[dict[str, Any]]:
    processes = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("PID "):
            continue
        parts = stripped.split(None, 3)
        if len(parts) < 3:
            continue
        pid_text, ppid_text, comm = parts[:3]
        command = parts[3] if len(parts) == 4 else comm
        if not pid_text.isdigit() or not ppid_text.isdigit():
            continue
        processes.append(
            {
                "pid": int(pid_text),
                "ppid": int(ppid_text),
                "comm": comm,
                "command": sanitize_command(command),
            }
        )
    return processes


def parse_npu_smi_info(text: str) -> dict[int, str]:
    result = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        npu_match = re.search(r"\d+", cells[0])
        if not npu_match or not cells[1].isdigit():
            continue
        result[int(cells[1])] = npu_match.group(0)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ps_text = read_or_run(args.ps_file, ["ps", "-eo", "pid,ppid,comm,args"])
    npu_smi_info_text = read_or_run(args.npu_smi_info_file, ["npu-smi", "info"], optional=True)
    result = discover_processes_from_text(ps_text, npu_smi_info_text, keyword=args.keyword)
    output = write_json_output(args.out, result)
    print(f"Generated {output}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover candidate NPU/PyTorch/LLM serving processes")
    parser.add_argument("--ps-file", help="Read ps output from a text file instead of running ps")
    parser.add_argument(
        "--npu-smi-info-file",
        help="Read npu-smi info output from a text file instead of running npu-smi",
    )
    parser.add_argument(
        "--keyword",
        help="Additional keyword or regex-like substring to include matching processes",
    )
    parser.add_argument("--out", required=True, help="Output process discovery JSON path")
    return parser.parse_args(argv)


def _is_kernel_thread(process: dict[str, Any]) -> bool:
    pid = int(process.get("pid", -1))
    ppid = int(process.get("ppid", -1))
    return pid == 2 or ppid == 2


def _role_hint(process: dict[str, Any]) -> str:
    haystack = f"{process.get('comm', '')} {process.get('command', '')}"
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(haystack):
            return role
    return "unknown"


def _matches_keyword(process: dict[str, Any], keyword: str | None) -> bool:
    if not keyword:
        return False
    haystack = f"{process.get('pid')} {process.get('comm', '')} {process.get('command', '')}".lower()
    return keyword.lower() in haystack


if __name__ == "__main__":
    raise SystemExit(main())
