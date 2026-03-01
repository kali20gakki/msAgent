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

    _EXIT_COMMANDS = frozenset({"/exit"})
    _CLEAR_COMMANDS = frozenset({"/clear"})
    _NEW_SESSION_COMMANDS = frozenset({"/new"})
    _COMMAND_HELP: tuple[tuple[str, str], ...] = (
        ("/new", "开启新会话并清空上下文"),
        ("/clear", "清空当前会话历史"),
        ("/exit", "退出 msAgent"),
    )

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

    def find_commands(self, query: str, limit: int = 8) -> list[tuple[str, str]]:
        normalized_query = " ".join(query.strip().lower().split())
        if not normalized_query:
            return list(self._COMMAND_HELP[:limit])

        scored: list[tuple[int, int, str, str]] = []
        for idx, (command, description) in enumerate(self._COMMAND_HELP):
            normalized_command = " ".join(command.lower().split())
            if normalized_command == normalized_query:
                score = 0
            elif normalized_command.startswith(normalized_query):
                score = 1
            elif normalized_query in normalized_command:
                score = 2
            else:
                continue
            scored.append((score, idx, command, description))

        scored.sort(key=lambda item: (item[0], item[1], len(item[2]), item[2]))
        return [(command, description) for _, _, command, description in scored[:limit]]
