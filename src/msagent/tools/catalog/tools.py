"""Catalog tools for browsing and invoking available runtime tools."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field

from msagent.tools.factory import ToolFactory


def _context_value(context: Any, key: str) -> Any:
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def _runtime_tool_catalog(runtime: Any) -> list[Any]:
    context = getattr(runtime, "context", None)
    if context is None:
        return []
    return list(_context_value(context, "tool_catalog") or [])


def _initializer_tool_catalog() -> list[Any]:
    try:
        from msagent.cli.bootstrap.initializer import initializer
    except Exception:
        return []
    return list(getattr(initializer, "cached_tools_in_catalog", []) or [])


def _fallback_tool_catalog() -> list[Any]:
    cached = _initializer_tool_catalog()
    if cached:
        return cached
    return list(ToolFactory().get_catalog_tools())


class FetchToolsInput(BaseModel):
    pattern: str = Field(default=".*", description="Regex used to filter tools by name/description")


@tool("fetch_tools", args_schema=FetchToolsInput)
async def fetch_tools(*, pattern: str = ".*", runtime: Any = None) -> str:
    """List tool names available in current runtime context."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ToolException(f"Invalid regex pattern: {exc}") from exc

    catalog = _runtime_tool_catalog(runtime)
    if not catalog:
        catalog = _fallback_tool_catalog()

    matched = []
    for tool_obj in catalog:
        name = getattr(tool_obj, "name", "")
        description = getattr(tool_obj, "description", "")
        haystack = f"{name}\n{description}"
        if compiled.search(haystack):
            matched.append(name)

    return "\n".join(matched)


class GetToolInput(BaseModel):
    tool_name: str = Field(description="Name of the tool to inspect")


@tool("get_tool", args_schema=GetToolInput)
async def get_tool(*, tool_name: str, runtime: Any = None) -> str:
    """Get tool schema and metadata in JSON format."""
    catalog = _runtime_tool_catalog(runtime)
    if not catalog:
        catalog = _fallback_tool_catalog()

    tool_obj = next(
        (tool for tool in catalog if getattr(tool, "name", "") == tool_name),
        None,
    )
    if tool_obj is None:
        return json.dumps(
            {
                "error": f"Tool '{tool_name}' not found",
                "available_tools": [
                    getattr(tool, "name", "")
                    for tool in catalog
                    if getattr(tool, "name", "")
                ],
            },
            ensure_ascii=False,
        )

    schema = getattr(tool_obj, "tool_call_schema", None) or getattr(tool_obj, "args_schema", None)
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_json = schema.model_json_schema()
    elif isinstance(schema, dict):
        schema_json = schema
    else:
        schema_json = {"type": "object", "properties": {}}

    payload = {
        "name": getattr(tool_obj, "name", tool_name),
        "description": getattr(tool_obj, "description", ""),
        "parameters": schema_json,
    }
    return json.dumps(payload, ensure_ascii=False)


class RunToolInput(BaseModel):
    tool_name: str = Field(description="Tool name to run")
    tool_args: dict[str, Any] = Field(default_factory=dict, description="Tool args")


@tool("run_tool", args_schema=RunToolInput)
async def run_tool(*, tool_name: str, tool_args: dict[str, Any], runtime: Any = None) -> Any:
    """Invoke a tool from the runtime catalog by name."""
    catalog = _runtime_tool_catalog(runtime)
    if not catalog:
        catalog = _fallback_tool_catalog()

    tool_obj = next(
        (tool for tool in catalog if getattr(tool, "name", "") == tool_name),
        None,
    )
    if tool_obj is None:
        return {
            "error": f"Tool '{tool_name}' not found",
            "available_tools": [
                getattr(tool, "name", "")
                for tool in catalog
                if getattr(tool, "name", "")
            ],
        }

    payload = dict(tool_args or {})
    try:
        return await tool_obj.ainvoke(payload | {"runtime": runtime})
    except TypeError:
        return await tool_obj.ainvoke(payload)

