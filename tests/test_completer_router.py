from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent, Completion
from prompt_toolkit.document import Document

from msagent.cli.completers.router import CompleterRouter
from msagent.cli.completers.slash import SlashCommandCompleter


async def _collect_completions(completer, text: str) -> list[Completion]:
    document = Document(text=text, cursor_position=len(text))
    complete_event = CompleteEvent(completion_requested=True)
    return [
        completion
        async for completion in completer.get_completions_async(document, complete_event)
    ]


class _StubAsyncCompleter:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def get_completions_async(self, document, complete_event):
        del document, complete_event
        self.calls += 1
        yield Completion(self.text, start_position=0)


@pytest.mark.asyncio
async def test_slash_command_completer_matches_case_insensitive_prefix() -> None:
    completer = SlashCommandCompleter(commands=["/help", "/models", "/quit"])

    completions = await _collect_completions(completer, "/HE")
    completion_texts = [completion.text for completion in completions]

    assert "/help" in completion_texts
    assert "/models" not in completion_texts


@pytest.mark.asyncio
async def test_completer_router_routes_slash_text_to_slash_completer() -> None:
    router = CompleterRouter(commands=["/help"], working_dir=Path.cwd())
    slash = _StubAsyncCompleter("/help")
    reference = _StubAsyncCompleter("@src")
    router.slash_completer = slash
    router.reference_completer = reference

    completions = await _collect_completions(router, "   /he")

    assert [completion.text for completion in completions] == ["/help"]
    assert slash.calls == 1
    assert reference.calls == 0


@pytest.mark.asyncio
async def test_completer_router_routes_non_slash_text_to_reference_completer() -> None:
    router = CompleterRouter(commands=["/help"], working_dir=Path.cwd())
    slash = _StubAsyncCompleter("/help")
    reference = _StubAsyncCompleter("@src")
    router.slash_completer = slash
    router.reference_completer = reference

    completions = await _collect_completions(router, "open @sr")

    assert [completion.text for completion in completions] == ["@src"]
    assert slash.calls == 0
    assert reference.calls == 1


def test_completer_router_sync_interface_returns_empty_iterator() -> None:
    router = CompleterRouter(commands=["/help"], working_dir=Path.cwd())
    document = Document(text="/", cursor_position=1)
    complete_event = CompleteEvent(completion_requested=True)

    assert list(router.get_completions(document, complete_event)) == []
