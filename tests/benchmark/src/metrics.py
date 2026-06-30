from __future__ import annotations

from collections import Counter
from typing import Any


def build_case_metrics(
    case_id: str,
    trace: dict[str, Any],
    judge_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    agent_tokens = trace.get("token_usage", {})
    judge_tokens = judge_result.get("token_usage", {}) if judge_result is not None else {}
    agent_available = bool(agent_tokens.get("available", True))
    judge_available = bool(judge_tokens.get("available", True)) if judge_result is not None else True
    agent_duration = int(trace.get("duration_ms") or 0)
    judge_duration = int(judge_result.get("duration_ms") or 0) if judge_result is not None else 0
    agent_msagent_duration = extract_msagent_session_duration(trace)
    judge_msagent_duration = (
        normalize_optional_int(judge_result.get("msagent_session_duration_ms")) if judge_result is not None else None
    )
    return {
        "case_id": case_id,
        "duration_ms": {
            "agent": agent_duration,
            "judge": judge_duration,
            "total": agent_duration + judge_duration,
        },
        "msagent_session_duration_ms": {
            "agent": agent_msagent_duration,
            "judge": judge_msagent_duration,
            "total": sum_optional_ints(agent_msagent_duration, judge_msagent_duration),
        },
        "token_usage": {
            "agent": agent_tokens,
            "judge": judge_tokens,
            "total": {
                "available": agent_available and judge_available,
                "input_tokens": int(agent_tokens.get("input_tokens", 0)) + int(judge_tokens.get("input_tokens", 0)),
                "output_tokens": int(agent_tokens.get("output_tokens", 0)) + int(judge_tokens.get("output_tokens", 0)),
                "total_tokens": int(agent_tokens.get("total_tokens", 0)) + int(judge_tokens.get("total_tokens", 0)),
            },
        },
        "tool_calls": {
            "agent": summarize_tool_calls(trace),
        },
        "judge": {
            "enabled": judge_result is not None,
        },
    }


def extract_msagent_session_duration(trace: dict[str, Any]) -> int | None:
    for event in trace.get("events", []):
        if event.get("type") == "agent_run":
            return normalize_optional_int(event.get("msagent_session_duration_ms"))
    return None


def normalize_optional_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return round(float(value))
    return None


def sum_optional_ints(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def summarize_tool_calls(trace: dict[str, Any]) -> dict[str, Any]:
    events = trace.get("events", [])
    results_by_item_id = {
        str(event.get("item_id")): event
        for event in events
        if event.get("type") == "tool_result" and event.get("item_id") is not None
    }

    calls = []
    by_tool: Counter[str] = Counter()
    for event in events:
        if event.get("type") != "tool_call":
            continue
        tool = str(event.get("tool") or "unknown")
        by_tool[tool] += 1
        tool_input = event.get("input", {})
        item_id = event.get("item_id")
        result = results_by_item_id.get(str(item_id)) if item_id is not None else None
        output = result.get("output", {}) if isinstance(result, dict) else {}
        calls.append(
            {
                "step": event.get("step"),
                "tool": tool,
                "item_id": item_id,
                "input": tool_input,
                "exit_code": output.get("exit_code") if isinstance(output, dict) else None,
                "status": output.get("status") if isinstance(output, dict) else None,
            }
        )

    return {
        "count": len(calls),
        "by_tool": dict(sorted(by_tool.items())),
        "calls": calls,
    }
