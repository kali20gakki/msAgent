from types import SimpleNamespace
from pathlib import Path

from msagent.cli.dispatchers.commands import CommandDispatcher


def test_command_dispatcher_removes_resume_and_replay_commands() -> None:
    session = SimpleNamespace(
        context=SimpleNamespace(
            working_dir=Path.cwd(),
            bash_mode=False,
            thread_id="thread-1",
            current_input_tokens=None,
            current_output_tokens=None,
        ),
        prompt=SimpleNamespace(hotkeys={}),
        renderer=SimpleNamespace(
            render_help=lambda *_args, **_kwargs: None,
            render_hotkeys=lambda *_args, **_kwargs: None,
        ),
        update_context=lambda **_kwargs: None,
        running=True,
    )
    dispatcher = CommandDispatcher(session)

    assert "/resume" not in dispatcher.commands
    assert "/replay" not in dispatcher.commands
    assert "/approve" not in dispatcher.commands
    assert "/memory" not in dispatcher.commands
    assert "/compress" not in dispatcher.commands
    assert "/todo" not in dispatcher.commands
    assert "/graph" not in dispatcher.commands
    assert "/offload" in dispatcher.commands
    assert "/threads" in dispatcher.commands


def test_command_dispatcher_offload_delegates_to_compression_handler(monkeypatch) -> None:
    session = SimpleNamespace(
        context=SimpleNamespace(
            working_dir=Path.cwd(),
            bash_mode=False,
            thread_id="thread-1",
            current_input_tokens=None,
            current_output_tokens=None,
        ),
        prompt=SimpleNamespace(hotkeys={}),
        renderer=SimpleNamespace(
            render_help=lambda *_args, **_kwargs: None,
            render_hotkeys=lambda *_args, **_kwargs: None,
        ),
        update_context=lambda **_kwargs: None,
        running=True,
    )
    dispatcher = CommandDispatcher(session)
    calls: list[str] = []

    async def fake_handle() -> None:
        calls.append("offload")

    monkeypatch.setattr(dispatcher.compression_handler, "handle", fake_handle)

    import asyncio

    asyncio.run(dispatcher.cmd_offload([]))

    assert calls == ["offload"]


def test_command_dispatcher_threads_delegates_to_threads_handler(monkeypatch) -> None:
    session = SimpleNamespace(
        context=SimpleNamespace(
            working_dir=Path.cwd(),
            bash_mode=False,
            thread_id="thread-1",
            current_input_tokens=None,
            current_output_tokens=None,
        ),
        prompt=SimpleNamespace(hotkeys={}),
        renderer=SimpleNamespace(
            render_help=lambda *_args, **_kwargs: None,
            render_hotkeys=lambda *_args, **_kwargs: None,
        ),
        update_context=lambda **_kwargs: None,
        running=True,
        message_dispatcher=SimpleNamespace(),
    )
    dispatcher = CommandDispatcher(session)
    calls: list[str] = []

    async def fake_handle() -> None:
        calls.append("threads")

    monkeypatch.setattr(dispatcher.threads_handler, "handle", fake_handle)

    import asyncio

    asyncio.run(dispatcher.cmd_threads([]))

    assert calls == ["threads"]

