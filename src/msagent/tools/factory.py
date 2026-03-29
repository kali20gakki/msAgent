"""Tool adapter layer for deepagents runtime integration."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel

_TOOL_TIMEOUT_METADATA_KEY = "_msagent_timeout_wrapped"


@dataclass(frozen=True, slots=True)
class ToolPreview:
    """Lightweight tool descriptor used by `/tools` when runtime metadata is absent."""

    name: str
    description: str


class ToolFactory:
    """Factory and adapters for tool runtime behavior."""

    _BUILTIN_TOOL_PREVIEWS: tuple[ToolPreview, ...] = (
        ToolPreview("ls", "List files and directories"),
        ToolPreview("read_file", "Read a file from the workspace"),
        ToolPreview("write_file", "Write file content to the workspace"),
        ToolPreview("edit_file", "Apply targeted edits to an existing file"),
        ToolPreview("glob", "Find files by glob patterns"),
        ToolPreview("grep", "Search text content in files"),
        ToolPreview("execute", "Execute shell commands in the local backend"),
        ToolPreview("write_todos", "Manage todo list state (provided by deepagents)"),
        ToolPreview("fetch_tools", "List available tools in the current runtime"),
        ToolPreview("get_tool", "Inspect schema/details for a specific tool"),
        ToolPreview("run_tool", "Invoke a tool by name with explicit arguments"),
        ToolPreview("fetch_skills", "List available skills in the current runtime"),
        ToolPreview("get_skill", "Read skill instructions from SKILL.md"),
        ToolPreview("web_search", "Search the web and return results with source URLs"),
    )

    def get_impl_tools(self) -> list[ToolPreview]:
        return list(self._BUILTIN_TOOL_PREVIEWS)

    def get_internal_tools(self) -> list[ToolPreview]:
        return []

    def get_catalog_tools(self) -> list[ToolPreview]:
        return list(self._BUILTIN_TOOL_PREVIEWS)

    def get_skill_catalog_tools(self) -> list[ToolPreview]:
        return [preview for preview in self._BUILTIN_TOOL_PREVIEWS if "skill" in preview.name]

    def get_impl_module_map(self) -> dict[str, str]:
        return {preview.name: "deepagents" for preview in self._BUILTIN_TOOL_PREVIEWS}

    def get_internal_module_map(self) -> dict[str, str]:
        return {}

    def wrap_tools_with_timeout(
        self,
        tools: list[BaseTool],
        timeout_seconds: float | None,
        *,
        source: str,
    ) -> list[BaseTool]:
        if timeout_seconds is None:
            return tools
        return [
            self.wrap_tool_with_timeout(tool, timeout_seconds, source=source)
            for tool in tools
        ]

    def wrap_tool_with_timeout(
        self,
        tool: BaseTool,
        timeout_seconds: float,
        *,
        source: str,
    ) -> BaseTool:
        metadata = dict(getattr(tool, "metadata", {}) or {})
        wrapped_timeout = metadata.get(_TOOL_TIMEOUT_METADATA_KEY)
        if isinstance(wrapped_timeout, (int, float)):
            return tool

        tool_name = getattr(tool, "name", "unknown_tool")
        description = getattr(tool, "description", "") or f"Wrapped tool {tool_name}"
        args_schema = getattr(tool, "args_schema", None)
        accepts_runtime = self._tool_accepts_runtime(tool)

        async def _ainvoke_with_timeout(*, runtime: Any = None, **kwargs: Any) -> Any:
            payload = dict(kwargs)
            if accepts_runtime and runtime is not None:
                payload["runtime"] = runtime
            try:
                return await asyncio.wait_for(
                    tool.ainvoke(payload),
                    timeout=float(timeout_seconds),
                )
            except asyncio.TimeoutError as exc:
                raise ToolException(
                    f"Tool '{tool_name}' timed out after {timeout_seconds:.0f}s. "
                    "Please narrow the scope and retry."
                ) from exc

        def _invoke_with_timeout(*, runtime: Any = None, **kwargs: Any) -> Any:
            payload = dict(kwargs)
            if accepts_runtime and runtime is not None:
                payload["runtime"] = runtime
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_ainvoke_with_timeout(**payload))

            # If we're already on an event loop, rely on async path upstream.
            if loop.is_running():
                raise ToolException(
                    f"Tool '{tool_name}' must run asynchronously in this context."
                )
            return tool.invoke(payload)

        wrapped_metadata = {
            **metadata,
            _TOOL_TIMEOUT_METADATA_KEY: float(timeout_seconds),
            "msagent_tool_source": source,
        }

        return StructuredTool.from_function(
            name=tool_name,
            description=description,
            func=_invoke_with_timeout,
            coroutine=_ainvoke_with_timeout,
            args_schema=args_schema,
            infer_schema=args_schema is None,
            metadata=wrapped_metadata,
        )

    @staticmethod
    def _tool_accepts_runtime(tool: BaseTool) -> bool:
        args_schema = getattr(tool, "args_schema", None)
        if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
            model_fields = getattr(args_schema, "model_fields", None)
            if isinstance(model_fields, dict) and "runtime" in model_fields:
                return True
            # Compatibility for pydantic v1 style models.
            legacy_fields = vars(args_schema).get("__fields__")
            if isinstance(legacy_fields, dict) and "runtime" in legacy_fields:
                return True

        for attr in ("func", "coroutine"):
            callable_obj = getattr(tool, attr, None)
            if callable_obj is None:
                continue
            try:
                signature = inspect.signature(callable_obj)
            except (TypeError, ValueError):
                continue
            if "runtime" in signature.parameters:
                return True
        return False

