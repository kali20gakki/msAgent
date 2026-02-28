"""Tests for tui module."""

from __future__ import annotations

import asyncio

import pytest

import msagent.tui as tui_module
from msagent.tui import ChatWelcomeBanner, CustomFooter, MSAgentApp, run_tui


def test_custom_footer_render_contains_shortcuts() -> None:
    footer = CustomFooter()
    rendered = footer.render()
    assert "/ for commands" in rendered
    assert "session:" in rendered
    assert "tokens:" in rendered


def test_chat_welcome_banner_compose_shows_server_status_when_connected() -> None:
    no_server_widgets = list(ChatWelcomeBanner().compose())
    assert len(no_server_widgets) == 1

    with_server_widgets = list(ChatWelcomeBanner(mcp_servers=["filesystem"]).compose())
    assert len(with_server_widgets) == 2


def test_chat_welcome_banner_compose_shows_loaded_skills() -> None:
    widgets = list(
        ChatWelcomeBanner(mcp_servers=["filesystem"], loaded_skills=["code-review", "devops"]).compose()
    )
    assert len(widgets) == 3


def test_run_tui_creates_app_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"run": False}

    class FakeApp:
        def run(self) -> None:
            called["run"] = True

    monkeypatch.setattr(tui_module, "MSAgentApp", FakeApp)
    run_tui()
    assert called["run"] is True


@pytest.mark.asyncio
async def test_connection_worker_initializes_and_shuts_down_on_cancel() -> None:
    class FakeAgent:
        def __init__(self) -> None:
            self.is_initialized = False
            self.initialize_called = False
            self.shutdown_called = False

        async def initialize(self) -> bool:
            self.initialize_called = True
            self.is_initialized = True
            return True

        async def shutdown(self) -> None:
            self.shutdown_called = True
            self.is_initialized = False

    app = MSAgentApp()
    app.agent = FakeAgent()

    worker_task = asyncio.create_task(app._connection_worker())
    await asyncio.sleep(0.01)
    worker_task.cancel()
    await worker_task

    assert app.agent.initialize_called is True
    assert app.agent.shutdown_called is True
