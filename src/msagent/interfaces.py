"""Shared backend/frontend contracts for msagent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

AgentEventType = Literal["text", "tool_call", "tool_result", "error", "done"]


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """Token usage snapshot for the latest assistant response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class AgentStatus:
    """Frontend-facing status payload."""

    is_initialized: bool
    error_message: str
    session_number: int
    provider: str
    model: str
    connected_servers: tuple[str, ...]
    loaded_skills: tuple[str, ...]
    usage: UsageSnapshot | None = None


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Typed chat event shared between backend and UI layers."""

    type: AgentEventType
    content: str | None = None
    full_name: str | None = None
    server: str | None = None
    tool: str | None = None
    payload: Any | None = None
    duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Compatibility helper for older dict-based consumers."""
        data: dict[str, Any] = {"type": self.type}
        if self.content is not None:
            data["content"] = self.content
        if self.full_name is not None:
            data["full_name"] = self.full_name
        if self.server is not None:
            data["server"] = self.server
        if self.tool is not None:
            data["tool"] = self.tool
        if self.payload is not None:
            if self.type == "tool_call":
                data["input"] = self.payload
            elif self.type == "tool_result":
                data["output"] = self.payload
            else:
                data["payload"] = self.payload
        if self.duration_s is not None:
            data["duration_s"] = self.duration_s
        return data


class AgentBackend(Protocol):
    """Frontend/backend separation boundary."""

    async def initialize(self) -> bool: ...

    async def shutdown(self) -> None: ...

    async def chat(self, user_input: str) -> str: ...

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]: ...

    async def stream_chat_events(self, user_input: str) -> AsyncGenerator[AgentEvent, None]: ...

    def clear_history(self) -> None: ...

    def start_new_session(self) -> int: ...

    def find_local_files(self, query: str, limit: int = 8) -> list[str]: ...

    def get_status(self) -> AgentStatus: ...
