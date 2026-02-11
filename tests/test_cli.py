"""Tests for cli module."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import typer

import msagent.cli as cli_module
from msagent.config import AppConfig, MCPConfig


class DummyStatus:
    """Simple context manager used by fake console.status()."""

    def __enter__(self) -> "DummyStatus":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class DummyConsole:
    """Console double that avoids rich/encoding side effects in tests."""

    def __init__(self, inputs: list[str] | None = None) -> None:
        self.inputs = list(inputs or [])
        self.print_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def print(self, *args: Any, **kwargs: Any) -> None:
        self.print_calls.append((args, kwargs))

    def input(self, *_args: Any, **_kwargs: Any) -> str:
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)

    def status(self, *_args: Any, **_kwargs: Any) -> DummyStatus:
        return DummyStatus()


def test_version_callback_raises_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_console = DummyConsole()
    monkeypatch.setattr(cli_module, "console", dummy_console)

    with pytest.raises(typer.Exit):
        cli_module.version_callback(True)

    assert len(dummy_console.print_calls) == 1


def test_chat_command_tui_mode_calls_run_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"run_tui": False}
    monkeypatch.setattr(cli_module, "run_tui", lambda: called.__setitem__("run_tui", True))
    cli_module.chat_command(message=None, stream=True, tui=True)
    assert called["run_tui"] is True


def test_config_command_updates_and_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_console = DummyConsole()
    monkeypatch.setattr(cli_module, "console", dummy_console)

    config = AppConfig()
    saved: dict[str, AppConfig | None] = {"value": None}

    class FakeConfigManager:
        def get_config(self) -> AppConfig:
            return config

        def save_config(self, new_config: AppConfig) -> None:
            saved["value"] = new_config

    monkeypatch.setattr(cli_module, "config_manager", FakeConfigManager())

    cli_module.config_command(
        show=False,
        llm_provider="gemini",
        llm_api_key="api-key",
        llm_base_url="https://example.com/v1",
        llm_model="gemini-model",
    )

    assert saved["value"] is config
    assert config.llm.provider == "gemini"
    assert config.llm.api_key == "api-key"
    assert config.llm.base_url == "https://example.com/v1"
    assert config.llm.model == "gemini-model"


def test_mcp_command_add_requires_name_and_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "console", DummyConsole())

    with pytest.raises(typer.Exit):
        cli_module.mcp_command(action="add", name=None, command=None, args=None)


def test_mcp_command_add_parses_args_and_calls_config_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "console", DummyConsole())
    captured: dict[str, MCPConfig | None] = {"value": None}

    class FakeConfigManager:
        def add_mcp_server(self, config: MCPConfig) -> None:
            captured["value"] = config

    monkeypatch.setattr(cli_module, "config_manager", FakeConfigManager())

    cli_module.mcp_command(
        action="add",
        name="filesystem",
        command="npx",
        args="-y,@modelcontextprotocol/server-filesystem,/tmp",
    )

    assert captured["value"] is not None
    assert captured["value"].name == "filesystem"
    assert captured["value"].command == "npx"
    assert captured["value"].args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]


def test_mcp_command_remove_requires_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_module, "console", DummyConsole())
    with pytest.raises(typer.Exit):
        cli_module.mcp_command(action="remove", name=None, command=None, args=None)


def test_mcp_command_list_without_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_console = DummyConsole()
    monkeypatch.setattr(cli_module, "console", dummy_console)

    class FakeConfigManager:
        def get_config(self) -> AppConfig:
            return AppConfig(mcp_servers=[])

    monkeypatch.setattr(cli_module, "config_manager", FakeConfigManager())
    cli_module.mcp_command(action="list", name=None, command=None, args=None)

    assert len(dummy_console.print_calls) >= 2


def test_chat_command_single_message_non_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_console = DummyConsole()
    monkeypatch.setattr(cli_module, "console", dummy_console)

    class FakeAgent:
        instances: list["FakeAgent"] = []

        def __init__(self) -> None:
            self.error_message = ""
            self.shutdown_called = False
            self.chat_calls: list[str] = []
            FakeAgent.instances.append(self)

        async def initialize(self) -> bool:
            return True

        async def chat(self, message: str) -> str:
            self.chat_calls.append(message)
            return "answer"

        async def chat_stream(self, _message: str):
            yield "stream"

        async def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)

    original_run = asyncio.run
    monkeypatch.setattr(cli_module.asyncio, "run", lambda coro: original_run(coro))

    cli_module.chat_command(message="hello", stream=False, tui=False)

    instance = FakeAgent.instances[-1]
    assert instance.chat_calls == ["hello"]
    assert instance.shutdown_called is True


def test_ask_command_non_stream_uses_chat_and_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_console = DummyConsole()
    monkeypatch.setattr(cli_module, "console", dummy_console)

    class FakeAgent:
        instances: list["FakeAgent"] = []

        def __init__(self) -> None:
            self.error_message = ""
            self.chat_calls: list[str] = []
            self.shutdown_called = False
            FakeAgent.instances.append(self)

        async def initialize(self) -> bool:
            return True

        async def chat(self, question: str) -> str:
            self.chat_calls.append(question)
            return "result"

        async def chat_stream(self, _question: str):
            yield "chunk"

        async def shutdown(self) -> None:
            self.shutdown_called = True

    monkeypatch.setattr(cli_module, "Agent", FakeAgent)

    original_run = asyncio.run
    monkeypatch.setattr(cli_module.asyncio, "run", lambda coro: original_run(coro))

    cli_module.ask_command(question="What is MCP?", stream=False)

    instance = FakeAgent.instances[-1]
    assert instance.chat_calls == ["What is MCP?"]
    assert instance.shutdown_called is True
