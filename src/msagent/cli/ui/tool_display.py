"""Shared tool display helpers for terminal rendering and live activity."""

from __future__ import annotations

import json
from typing import Any, Mapping

HIDDEN_TOOL_NAMES = frozenset({"write_todos"})
SUBAGENT_ORIGIN_LABEL = "Subagent"
PRIORITY_TOOL_ARG_KEYS = ("command", "cmd", "script", "cwd", "workdir", "timeout", "timeout_ms")


def truncate_preview_text(text: str, max_length: int) -> str:
    """Keep previews compact while preserving omitted-length context."""
    if max_length <= 0 or len(text) <= max_length:
        return text

    suffix = f"... ({len(text)} chars)"
    keep = max(8, max_length - len(suffix))
    if keep >= len(text):
        return text
    return f"{text[:keep]}{suffix}"


def stringify_tool_arg(value: Any, max_length: int) -> str:
    """Convert tool args to a compact single-line string."""
    if isinstance(value, str):
        text = value.replace("\r\n", "\n").replace("\n", " | ")
    elif isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)

    return truncate_preview_text(text, max_length)


def build_tool_arg_items(tool_args: Mapping[str, Any], *, max_value_length: int) -> list[tuple[str, str]]:
    """Prepare compact key/value pairs for tool arg rendering."""
    return [(str(key), stringify_tool_arg(value, max_value_length)) for key, value in tool_args.items()]


def order_tool_arg_items(tool_args: Mapping[str, Any]) -> list[tuple[str, Any]]:
    """Keep command-like arguments first so shell invocations stay visible."""
    priority_map = {name: index for index, name in enumerate(PRIORITY_TOOL_ARG_KEYS)}
    indexed_items = [(index, str(key), value) for index, (key, value) in enumerate(tool_args.items())]
    indexed_items.sort(key=lambda item: (priority_map.get(item[1], len(priority_map)), item[0]))
    return [(key, value) for _, key, value in indexed_items]


def resolve_origin_label(
    *,
    indent_level: int = 0,
    namespace: tuple | None = None,
    origin_label: str | None = None,
) -> str | None:
    """Resolve a human-friendly origin tag for nested subagent output."""
    if origin_label:
        return origin_label
    if indent_level > 0 or namespace:
        return SUBAGENT_ORIGIN_LABEL
    return None


def should_hide_tool_name(name: str | None) -> bool:
    """Return whether a tool should be hidden from normal tool-call display."""
    return bool(name) and name in HIDDEN_TOOL_NAMES


def should_hide_tool_call(tool_call: Mapping[str, Any]) -> bool:
    """Return whether a tool call payload should be hidden from display."""
    return should_hide_tool_name(str(tool_call.get("name", "") or ""))
