"""Middleware to evict oversized tool outputs into virtual filesystem."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command


class ToolResultEvictionMiddleware(AgentMiddleware[Any, Any, Any]):
    """Delegate large-result handling to deepagents FilesystemMiddleware.

    This middleware only participates in tool-call wrapping, so we can configure
    `tool_token_limit_before_evict` without introducing duplicate filesystem tools.
    """

    def __init__(
        self,
        *,
        backend: Any,
        tool_token_limit_before_evict: int,
    ) -> None:
        self._delegate = FilesystemMiddleware(
            backend=backend,
            tool_token_limit_before_evict=tool_token_limit_before_evict,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        result = self._delegate.wrap_tool_call(request, handler)
        return self._normalize_tool_result(result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        result = await self._delegate.awrap_tool_call(request, handler)
        return self._normalize_tool_result(result)

    @classmethod
    def _normalize_tool_result(
        cls,
        result: ToolMessage | Command,
    ) -> ToolMessage | Command:
        if isinstance(result, ToolMessage):
            return cls._normalize_tool_message(result)

        update = getattr(result, "update", None)
        if not isinstance(update, dict):
            return result

        messages = update.get("messages")
        if not isinstance(messages, list):
            return result

        for idx, message in enumerate(messages):
            if isinstance(message, ToolMessage):
                messages[idx] = cls._normalize_tool_message(message)
        return result

    @staticmethod
    def _normalize_tool_message(message: ToolMessage) -> ToolMessage:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return message

        normalized_content = ToolResultEvictionMiddleware._coerce_content_to_text(
            content
        )
        try:
            message.content = normalized_content
            return message
        except Exception:
            return message.model_copy(update={"content": normalized_content})

    @staticmethod
    def _coerce_content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (bytes, bytearray)):
            return bytes(content).decode("utf-8", errors="replace")
        try:
            return json.dumps(content, ensure_ascii=False, default=str)
        except TypeError:
            return str(content)
