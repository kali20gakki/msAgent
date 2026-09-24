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

"""Structured parsers for events emitted by ``msagent --trace-jsonl``."""

from __future__ import annotations

import json
from typing import Any


def _validate_traces(traces: list[dict]) -> None:
    """Fail early when callers pass data that is not a JSON object list."""
    if not isinstance(traces, list):
        raise TypeError("traces must be a list of dictionaries")

    for index, event in enumerate(traces):
        if not isinstance(event, dict):
            raise TypeError(f"trace event at index {index} is {type(event).__name__}, expected dict")


def _validate_tool_name(tool_name: str) -> None:
    """Reject empty tool filters, which would otherwise hide test mistakes."""
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("tool_name must be a non-empty string")


def get_tool_calls(traces: list[dict], tool_name: str) -> list[dict]:
    """Return matching ``tool_call`` events in their original trace order."""
    _validate_traces(traces)
    _validate_tool_name(tool_name)

    return [event for event in traces if event.get("type") == "tool_call" and event.get("tool") == tool_name]


def get_tool_results(traces: list[dict], tool_name: str) -> list[dict]:
    """Return matching ``tool_result`` events in their original trace order."""
    _validate_traces(traces)
    _validate_tool_name(tool_name)

    return [event for event in traces if event.get("type") == "tool_result" and event.get("tool") == tool_name]


def get_token_usage(traces: list[dict]) -> dict:
    """Extract token usage from the latest completed session.

    A trace should normally contain exactly one ``session_finished`` event. The
    reverse scan is defensive: if multiple sessions were concatenated, the most
    recent session is the relevant one. Missing or malformed usage is returned
    as an empty dictionary so tests can produce a direct assertion failure.
    """
    _validate_traces(traces)

    for event in reversed(traces):
        if event.get("type") != "session_finished":
            continue
        token_usage: Any = event.get("token_usage")
        return token_usage if isinstance(token_usage, dict) else {}
    return {}


def is_mcp_tool_name(
    actual_name: object,
    server_name: str,
    tool_name: str,
) -> bool:
    """Match an MCP tool name across adapter separator variants.

    ``langchain-mcp-adapters`` has emitted both ``server__tool`` and
    ``server_tool`` across versions. The server name itself is intentionally
    kept unchanged, so ``msprof-mcp`` remains distinguishable from another MCP
    server exposing a function with the same raw name.
    """
    _validate_tool_name(server_name)
    _validate_tool_name(tool_name)
    if not isinstance(actual_name, str):
        return False
    return actual_name in {
        f"{server_name}__{tool_name}",
        f"{server_name}_{tool_name}",
    }


def get_mcp_tool_calls(
    traces: list[dict],
    server_name: str,
    tool_name: str,
) -> list[dict]:
    """Return calls to one raw tool owned by a specific MCP server."""
    _validate_traces(traces)
    return [
        event
        for event in traces
        if event.get("type") == "tool_call" and is_mcp_tool_name(event.get("tool"), server_name, tool_name)
    ]


def get_mcp_tool_results(
    traces: list[dict],
    server_name: str,
    tool_name: str,
) -> list[dict]:
    """Return results from one raw tool owned by a specific MCP server."""
    _validate_traces(traces)
    return [
        event
        for event in traces
        if event.get("type") == "tool_result" and is_mcp_tool_name(event.get("tool"), server_name, tool_name)
    ]


def get_tool_result_text(event: dict) -> str:
    """Extract text from a traced tool result, including MCP content blocks.

    msagent records ``ToolMessage.content`` as text. MCP adapters may place the
    actual server response directly in that string or serialize it as a JSON
    content-block list such as ``[{"type": "text", "text": "..."}]``.
    """
    if not isinstance(event, dict) or event.get("type") != "tool_result":
        raise ValueError("event must be a tool_result dictionary")
    output = event.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"tool_result output must be a dictionary: {event!r}")
    content = output.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"tool_result content must be non-empty text: {output!r}")

    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        return content

    blocks = decoded if isinstance(decoded, list) else [decoded]
    text_parts = [block["text"] for block in blocks if isinstance(block, dict) and isinstance(block.get("text"), str)]
    return "\n".join(text_parts) if text_parts else content


def get_tool_result_json(event: dict) -> dict:
    """Parse a tool result's textual payload as a JSON object."""
    content = get_tool_result_text(event)
    try:
        payload: Any = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool result is not valid JSON: {content!r}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"tool result JSON must be an object, got {type(payload).__name__}")
    return payload
