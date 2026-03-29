from pathlib import Path
import signal

import pytest

from msagent.cli.core.context import Context
from msagent.cli.core.session import Session
from msagent.cli.ui.prompt import InteractivePrompt
from msagent.configs import ApprovalMode


def _build_context() -> Context:
    return Context(
        agent="msagent",
        model="default",
        thread_id="thread-1",
        working_dir=Path.cwd(),
        approval_mode=ApprovalMode.ACTIVE,
        recursion_limit=80,
    )


def _patch_prompt_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_setup_session(self) -> None:
        self.prompt_session = None

    monkeypatch.setattr(InteractivePrompt, "_setup_session", fake_setup_session)


@pytest.mark.asyncio
async def test_check_updates_background_keeps_sigint_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prompt_setup(monkeypatch)
    session = Session(_build_context())
    session._sigint_registered = True

    monkeypatch.setattr(
        "msagent.cli.core.session.check_for_updates",
        lambda: None,
    )

    await session._check_updates_background()

    assert session._sigint_registered is True


def test_sigint_handler_delegates_to_prompt_when_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_prompt_setup(monkeypatch)
    session = Session(_build_context())
    session._previous_sigint = signal.default_int_handler
    session.current_stream_task = None
    calls: list[str] = []

    session.prompt.handle_external_sigint = lambda: calls.append("prompt") or True

    session._sigint_registered = True
    try:
        handler = None
        session._register_sigint_handler()
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
    finally:
        if handler is not None:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        session._sigint_registered = False

    assert calls == ["prompt"]
