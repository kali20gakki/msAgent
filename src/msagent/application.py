"""Application service layer for msagent frontends."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal

from .interfaces import AgentBackend, AgentEvent, AgentStatus

UserIntentType = Literal["chat", "clear", "new_session", "exit", "ignore"]


@dataclass(frozen=True, slots=True)
class UserIntent:
    """Parsed user intent from raw input."""

    type: UserIntentType
    message: str = ""


class ChatApplicationService:
    """Coordinates frontend actions with backend use-cases."""

    _EXIT_COMMANDS = frozenset({"/exit", "/quit", "/q", ":q"})
    _CLEAR_COMMANDS = frozenset({"/clear"})
    _NEW_SESSION_COMMANDS = frozenset({"/new", "/new-session", "/session new"})

    def __init__(self, backend: AgentBackend):
        self._backend = backend

    @property
    def backend(self) -> AgentBackend:
        return self._backend

    async def initialize(self) -> bool:
        return await self._backend.initialize()

    async def shutdown(self) -> None:
        await self._backend.shutdown()

    def get_status(self) -> AgentStatus:
        return self._backend.get_status()

    def resolve_user_input(self, raw_input: str) -> UserIntent:
        text = raw_input.strip()
        if not text:
            return UserIntent("ignore")

        normalized = " ".join(text.lower().split())
        if normalized in self._EXIT_COMMANDS:
            return UserIntent("exit")
        if normalized in self._CLEAR_COMMANDS:
            return UserIntent("clear")
        if normalized in self._NEW_SESSION_COMMANDS:
            return UserIntent("new_session")
        return UserIntent("chat", message=text)

    async def chat(self, user_input: str) -> str:
        return await self._backend.chat(user_input)

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        async for chunk in self._backend.chat_stream(user_input):
            yield chunk

    async def stream_chat_events(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        async for event in self._backend.stream_chat_events(user_input):
            yield event

    def clear_history(self) -> None:
        self._backend.clear_history()

    def start_new_session(self) -> int:
        return self._backend.start_new_session()

    def find_local_files(self, query: str, limit: int = 8) -> list[str]:
        return self._backend.find_local_files(query, limit=limit)
