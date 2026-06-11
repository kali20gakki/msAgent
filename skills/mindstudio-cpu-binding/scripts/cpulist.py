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

"""CPU list parsing helpers."""

from __future__ import annotations


def parse_cpu_list(value: str | None) -> set[int]:
    if value is None:
        return set()
    text = str(value).strip()
    if not text:
        return set()

    cpus: set[int] = set()
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = _parse_cpu_number(start_text, token)
            end = _parse_cpu_number(end_text, token)
            if end < start:
                raise ValueError(f"invalid CPU range: {token}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(_parse_cpu_number(token, token))
    return cpus


def format_cpu_list(cpus: set[int] | list[int] | tuple[int, ...]) -> str:
    ordered = sorted(set(cpus))
    if not ordered:
        return ""

    ranges: list[str] = []
    start = prev = ordered[0]
    for cpu in ordered[1:]:
        if cpu == prev + 1:
            prev = cpu
            continue
        ranges.append(_format_range(start, prev))
        start = prev = cpu
    ranges.append(_format_range(start, prev))
    return ",".join(ranges)


def count_cpu_list(value: str | None) -> int:
    return len(parse_cpu_list(value))


def intersect_cpu_lists(left: str | None, right: str | None) -> set[int]:
    return parse_cpu_list(left) & parse_cpu_list(right)


def subtract_cpu_lists(left: str | None, right: str | None) -> set[int]:
    return parse_cpu_list(left) - parse_cpu_list(right)


def is_subset_cpu_list(candidate: str | None, allowed: str | None) -> bool:
    candidate_cpus = parse_cpu_list(candidate)
    allowed_cpus = parse_cpu_list(allowed)
    return bool(candidate_cpus) and candidate_cpus.issubset(allowed_cpus)


def cpus_by_numa(numa_nodes: list[dict]) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for node in numa_nodes:
        node_id = node.get("node")
        if node_id is None:
            continue
        result[int(node_id)] = parse_cpu_list(node.get("cpus"))
    return result


def numa_nodes_for_cpu_list(cpu_list: str | None, numa_nodes: list[dict]) -> set[int]:
    cpus = parse_cpu_list(cpu_list)
    matched: set[int] = set()
    for node_id, node_cpus in cpus_by_numa(numa_nodes).items():
        if cpus & node_cpus:
            matched.add(node_id)
    return matched


def _parse_cpu_number(token: str, original: str) -> int:
    try:
        value = int(token)
    except ValueError as exc:
        raise ValueError(f"invalid CPU value in {original}: {token}") from exc
    if value < 0:
        raise ValueError(f"negative CPU number: {original}")
    return value


def _format_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"
