"""Tests for application service layer."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from msagent.application import ChatApplicationService
from msagent.interfaces import AgentEvent, AgentStatus, UsageSnapshot


class FakeBackend:
    def __init__(self) -> None:
        self.initialized = False
        self.shutdown_called = False
        self.messages: list[str] = []

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def shutdown(self) -> None:
        self.shutdown_called = True

    async def chat(self, user_input: str) -> str:
        self.messages.append(user_input)
        return f"reply:{user_input}"

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        yield f"{user_input}-1"
        yield f"{user_input}-2"

    async def stream_chat_events(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        yield AgentEvent(type="text", content=user_input)
        yield AgentEvent(type="done", duration_s=0.1)

    def clear_history(self) -> None:
        self.messages.clear()

    def start_new_session(self) -> int:
        self.messages.clear()
        return 2

    def find_local_files(self, query: str, limit: int = 8) -> list[str]:
        return [f"{query}-{limit}"]

    def get_status(self) -> AgentStatus:
        return AgentStatus(
            is_initialized=self.initialized,
            error_message="",
            session_number=1,
            provider="openai",
            model="gpt",
            connected_servers=("filesystem",),
            loaded_skills=("review",),
            usage=UsageSnapshot(prompt_tokens=1, completion_tokens=2, total_tokens=3),
        )


def test_resolve_user_input_commands() -> None:
    service = ChatApplicationService(FakeBackend())

    assert service.resolve_user_input("").type == "ignore"
    assert service.resolve_user_input("  /clear ").type == "clear"
    assert service.resolve_user_input("/new-session").type == "new_session"
    assert service.resolve_user_input(":q").type == "exit"
    chat_intent = service.resolve_user_input(" hello ")
    assert chat_intent.type == "chat"
    assert chat_intent.message == "hello"


@pytest.mark.asyncio
async def test_service_delegates_chat_and_stream() -> None:
    backend = FakeBackend()
    service = ChatApplicationService(backend)

    await service.initialize()
    text = await service.chat("hi")
    chunks = [chunk async for chunk in service.chat_stream("ok")]
    events = [event async for event in service.stream_chat_events("evt")]
    await service.shutdown()

    assert backend.initialized is True
    assert text == "reply:hi"
    assert chunks == ["ok-1", "ok-2"]
    assert events[0].type == "text"
    assert backend.shutdown_called is True
