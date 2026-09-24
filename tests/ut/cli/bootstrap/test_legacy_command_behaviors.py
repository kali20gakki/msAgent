#!/usr/bin/python3
# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import pytest
import yaml
from rich.console import Console

import msagent.cli.bootstrap.legacy as legacy_module
from msagent.cli.bootstrap.legacy import DEFAULT_SESSION_COMMAND
from msagent.configs.llm import LLMProvider


def test_normalize_argv_routes_to_session_when_arguments_are_empty() -> None:
    assert legacy_module.normalize_argv([]) == [DEFAULT_SESSION_COMMAND]


def test_create_legacy_parser_parses_config_options_when_update_is_requested(
    tmp_path: Path,
) -> None:
    parser = legacy_module.create_legacy_parser()

    args = parser.parse_args(
        [
            "config",
            "--llm-provider",
            "google",
            "--llm-model",
            "gemini-2.5-pro",
            "--llm-max-tokens",
            "4096",
            "--working-dir",
            str(tmp_path),
        ]
    )

    assert args.cli_command == "config"
    assert args.llm_provider == "google"
    assert args.llm_model == "gemini-2.5-pro"
    assert args.llm_max_tokens == 4096
    assert args.working_dir == str(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected_handler"),
    [(None, "chat"), (DEFAULT_SESSION_COMMAND, "chat"), ("config", "config")],
)
async def test_dispatch_legacy_command_calls_expected_handler_when_command_is_supported(
    command: str | None,
    expected_handler: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chat = AsyncMock(return_value=11)
    config = AsyncMock(return_value=12)
    monkeypatch.setattr(legacy_module, "_handle_chat", chat)
    monkeypatch.setattr(legacy_module, "_handle_config", config)
    args = argparse.Namespace(version=False, cli_command=command)

    result = await legacy_module.dispatch_legacy_command(args)

    assert result == (11 if expected_handler == "chat" else 12)
    if expected_handler == "chat":
        chat.assert_awaited_once_with(args)
        config.assert_not_awaited()
    else:
        config.assert_awaited_once_with(args)
        chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_dispatch_legacy_command_reports_error_when_command_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    print_error = Mock()
    monkeypatch.setattr(legacy_module.console, "print_error", print_error)
    monkeypatch.setattr(legacy_module.console, "print", Mock())

    result = await legacy_module.dispatch_legacy_command(argparse.Namespace(version=False, cli_command="removed"))

    assert result == 1
    print_error.assert_called_once_with("Unknown command: removed")


@pytest.mark.asyncio
async def test_handle_chat_forwards_legacy_arguments_when_session_command_is_dispatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = AsyncMock(return_value=3)
    monkeypatch.setattr(legacy_module, "handle_chat_command", handler)
    args = argparse.Namespace(
        message="hello",
        working_dir="work",
        agent="Profiler",
        model="default",
        timer=True,
        approval_mode="active",
        execute_approval_mode=None,
        verbose=True,
        stream=False,
        trace_jsonl="trace.jsonl",
    )

    assert await legacy_module._handle_chat(args) == 3
    forwarded = handler.await_args.args[0]
    assert vars(forwarded) == {
        "message": "hello",
        "working_dir": "work",
        "agent": "Profiler",
        "model": "default",
        "timer": True,
        "server": False,
        "approval_mode": "active",
        "execute_approval_mode": None,
        "verbose": True,
        "stream": False,
        "trace_jsonl": "trace.jsonl",
    }


def _config_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "working_dir": str(tmp_path),
        "show": False,
        "llm_provider": None,
        "llm_api_key": None,
        "llm_max_tokens": None,
        "llm_base_url": None,
        "llm_model": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.asyncio
async def test_handle_config_routes_to_show_when_no_update_option_is_provided(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SimpleNamespace(ensure_config_dir=AsyncMock())
    monkeypatch.setattr(legacy_module.initializer, "get_registry", lambda _path: registry)
    show = AsyncMock(return_value=5)
    monkeypatch.setattr(legacy_module, "_show_config", show)

    result = await legacy_module._handle_config(_config_args(tmp_path))

    assert result == 5
    registry.ensure_config_dir.assert_awaited_once()
    show.assert_awaited_once_with(registry, tmp_path)


@pytest.mark.asyncio
async def test_handle_config_rejects_provider_when_provider_is_unsupported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SimpleNamespace(ensure_config_dir=AsyncMock())
    monkeypatch.setattr(legacy_module.initializer, "get_registry", lambda _path: registry)
    print_error = Mock()
    monkeypatch.setattr(legacy_module.console, "print_error", print_error)
    monkeypatch.setattr(legacy_module.console, "print", Mock())

    result = await legacy_module._handle_config(_config_args(tmp_path, llm_provider="unsupported"))

    assert result == 1
    assert "Unsupported provider: unsupported" in print_error.call_args.args[0]


@pytest.mark.asyncio
async def test_handle_config_persists_model_and_process_key_when_provider_update_is_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "global-config"
    llms_file = config_dir / "config.llms.yml"
    current_llm = SimpleNamespace(
        version="26.1.0",
        provider=LLMProvider.OPENAI,
        model="old-model",
        base_url=None,
        max_tokens=1024,
        temperature=0.2,
        request_timeout_seconds=90.0,
        context_window=128000,
    )
    agent = SimpleNamespace(name="Profiler", llm=current_llm)
    registry = SimpleNamespace(
        ensure_config_dir=AsyncMock(),
        get_agent=AsyncMock(return_value=agent),
        llms_file=llms_file,
        invalidate_cache=Mock(),
    )
    monkeypatch.setattr(legacy_module.initializer, "get_registry", lambda _path: registry)
    monkeypatch.setattr(legacy_module.console, "print_warning", Mock())
    monkeypatch.setattr(legacy_module.console, "print_success", Mock())
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = await legacy_module._handle_config(
        _config_args(
            tmp_path,
            llm_provider=" google ",
            llm_model="gemini-2.5-pro",
            llm_max_tokens=0,
            llm_base_url="https://example.test/v1",
            llm_api_key="temporary-secret",
        )
    )

    assert result == 0
    saved_content = llms_file.read_text(encoding="utf-8")
    saved = yaml.safe_load(saved_content)["llms"][0]
    assert saved["provider"] == "google"
    assert saved["model"] == "gemini-2.5-pro"
    assert saved["max_tokens"] == 0
    assert saved["base_url"] == "https://example.test/v1"
    assert "temporary-secret" not in saved_content
    assert os.environ["GOOGLE_API_KEY"] == "temporary-secret"
    registry.invalidate_cache.assert_called_once()


class _CaptureConsole:
    def __init__(self) -> None:
        self.console = Console(record=True, width=120)

    def print(self, *args: object, **kwargs: object) -> None:
        self.console.print(*args, **kwargs)


@pytest.mark.asyncio
async def test_show_config_renders_provider_key_status_and_mcp_counts_when_servers_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_llm = SimpleNamespace(
        provider=LLMProvider.OPENAI,
        model="default-model",
        api_key_env="OPENAI_API_KEY",
        base_url=None,
        max_tokens=1024,
    )
    workspace_llm = SimpleNamespace(
        provider=LLMProvider.GOOGLE,
        model="gemini-2.5-pro",
        api_key_env="GOOGLE_API_KEY",
        base_url=None,
        max_tokens=0,
    )
    agent = SimpleNamespace(name="Profiler", llm=default_llm)
    servers = {
        "enabled": SimpleNamespace(command="python", url=None, args=["server.py"], enabled=True),
        "disabled": SimpleNamespace(command=None, url="https://mcp.test", args=[], enabled=False),
    }
    registry = SimpleNamespace(
        get_agent=AsyncMock(return_value=agent),
        get_current_model=AsyncMock(return_value="workspace-model"),
        get_llm=AsyncMock(return_value=workspace_llm),
        load_mcp=AsyncMock(return_value=SimpleNamespace(servers=servers)),
        config_dir=tmp_path / "global-config",
    )
    capture = _CaptureConsole()
    monkeypatch.setattr(legacy_module, "console", capture)
    monkeypatch.setenv("GOOGLE_API_KEY", "configured")

    assert await legacy_module._show_config(registry, tmp_path) == 0
    output = capture.console.export_text()

    assert "LLM Provider" in output and "gemini" in output
    assert "API Key" in output and "Set" in output
    assert "Max Tokens" in output and "Auto" in output
    assert "MCP Servers" in output and "Enabled" in output and "Disabled" in output
    assert str(registry.config_dir) in output
    registry.get_current_model.assert_awaited_once_with("Profiler")
    registry.get_llm.assert_awaited_once_with("workspace-model")


@pytest.mark.parametrize(
    ("build_date", "expected"),
    [("2027-01-02", "2027"), ("unknown", "2026"), ("20x6-01-01", "2026")],
)
def test_resolve_copyright_year_returns_expected_year_when_build_date_varies(
    build_date: str,
    expected: str,
) -> None:
    assert legacy_module._resolve_copyright_year(build_date) == expected
