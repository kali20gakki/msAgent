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

"""Affinity plan generation."""

from __future__ import annotations

from typing import Any

from scripts.cpulist import format_cpu_list, intersect_cpu_lists, parse_cpu_list
from scripts.diagnose import recommended_numa_cpus
from scripts.snapshot import effective_cpu_list


def generate_plan(
    snapshot: dict[str, Any],
    findings: list[dict[str, Any]],
    executor_backend: str = "dry-run",
) -> dict[str, Any]:
    actions = [_action_for_process(snapshot, process, executor_backend) for process in snapshot.get("processes", [])]
    actions = [action for action in actions if action]
    return {
        "schema_version": "0.1.0",
        "executor_backend": executor_backend,
        "requires_confirmation": True,
        "risk_level": "low" if actions else "info",
        "summary": _summary(findings, actions),
        "recommendations": _recommendations(findings),
        "apply_actions": actions,
        "rollback_actions": [_rollback_action(action, executor_backend) for action in actions],
        "rollback_state_preview": {
            "schema_version": "0.1.0",
            "executor_backend": executor_backend,
            "actions": [_rollback_state_action(action) for action in actions],
        },
        "findings": findings,
    }


def _action_for_process(
    snapshot: dict[str, Any], process: dict[str, Any], executor_backend: str
) -> dict[str, Any] | None:
    pid = int(process.get("pid"))
    current = process.get("cpus_allowed_list")
    effective = effective_cpu_list(snapshot, process)
    preferred = recommended_numa_cpus(snapshot, process)
    if not current or not effective or not preferred:
        return None

    target_cpus = intersect_cpu_lists(effective, preferred)
    if not target_cpus:
        target_cpus = parse_cpu_list(effective)
    current_cpus = parse_cpu_list(current)
    if target_cpus == current_cpus:
        return None

    target = format_cpu_list(target_cpus)
    action_id = f"bind-pid-{pid}"
    apply_command = _apply_command(executor_backend, pid, target)
    rollback_command = _apply_command(executor_backend, pid, current)
    return {
        "action_id": action_id,
        "pid": pid,
        "rank": process.get("rank"),
        "npu_device": process.get("npu_device"),
        "current_cpu_list": current,
        "effective_cpu_list": effective,
        "target_cpu_list": target,
        "executor_backend": executor_backend,
        "apply_command": apply_command,
        "rollback_command": rollback_command,
        "requires_confirmation": True,
        "status": "preview",
        "risk": "low",
        "reason": "将进程 CPU range 收敛到 cgroup 允许范围与推荐 NUMA CPU 的交集。",
    }


def _rollback_action(action: dict[str, Any], executor_backend: str) -> dict[str, Any]:
    return {
        "action_id": f"rollback-{action['action_id']}",
        "pid": action["pid"],
        "target_cpu_list": action["current_cpu_list"],
        "executor_backend": executor_backend,
        "rollback_command": action["rollback_command"],
        "requires_confirmation": True,
        "status": "preview",
    }


def _rollback_state_action(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_id": action["action_id"],
        "pid": action["pid"],
        "process_start_time": None,
        "before": {"cpus_allowed_list": action["current_cpu_list"]},
        "target": {"cpus_allowed_list": action["target_cpu_list"]},
        "after": {"cpus_allowed_list": None},
        "status": "pending",
        "apply_command": action["apply_command"],
        "rollback_command": action["rollback_command"],
    }


def _apply_command(executor_backend: str, pid: int, cpu_list: str) -> str:
    if executor_backend == "internal-script":
        return f"internal-bind --apply --pid {pid} --cpu-list {cpu_list}"
    return f"taskset -cp {cpu_list} {pid}"


def _summary(findings: list[dict[str, Any]], actions: list[dict[str, Any]]) -> str:
    high = sum(1 for finding in findings if finding.get("severity") == "high")
    medium = sum(1 for finding in findings if finding.get("severity") == "medium")
    return f"发现 {len(findings)} 个问题，其中 high={high}, medium={medium}; 生成 {len(actions)} 个低风险绑核预览动作。"


def _recommendations(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    for finding in findings:
        for item in finding.get("recommendations", []):
            recommendations.append(
                {
                    "source_finding": finding.get("id"),
                    "severity": finding.get("severity"),
                    "recommendation": item,
                }
            )
    return recommendations
