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

import pytest

from msagent.cli.bootstrap import app as bootstrap_app
from msagent.cli.bootstrap.app import create_parser
from msagent.cli.bootstrap.legacy import (
    DEFAULT_SESSION_COMMAND,
    create_session_parser,
    normalize_argv,
    render_config_help,
    render_version_info,
)
from msagent.core.constants import APP_NAME
from msagent.core.storage_layout import StorageLayoutError


def test_create_session_parser_defaults_to_interactive_mode() -> None:
    parser = create_session_parser()
    args = parser.parse_args([])

    assert parser.prog == APP_NAME == "msagent"
    assert args.message is None
    assert args.cli_command == DEFAULT_SESSION_COMMAND
    assert args.resume is False
    assert args.stream is True
    assert args.trace_jsonl is None


def test_create_session_parser_accepts_explicit_agent_selection() -> None:
    parser = create_session_parser()
    args = parser.parse_args(["--agent", "Minos", "--trace-jsonl", "events.jsonl"])

    assert args.agent == "Minos"
    assert args.trace_jsonl == "events.jsonl"
    assert args.message is None


def test_normalize_argv_routes_messages_to_default_session() -> None:
    assert normalize_argv(["hello"]) == [DEFAULT_SESSION_COMMAND, "hello"]
    assert normalize_argv(["--agent", "Minos"]) == [
        DEFAULT_SESSION_COMMAND,
        "--agent",
        "Minos",
    ]
    assert normalize_argv(["config", "--show"]) == ["config", "--show"]
    assert normalize_argv(["--help"]) == ["--help"]
    assert normalize_argv(["-h"]) == ["-h"]
    assert normalize_argv(["--version"]) == ["--version"]
    assert normalize_argv(["-V"]) == ["-V"]


def test_help_only_exposes_public_commands_only() -> None:
    parser = create_parser()
    help_text = parser.format_help()

    assert "config" in help_text
    assert "web" not in help_text
    assert DEFAULT_SESSION_COMMAND not in help_text
    assert "chat" not in help_text
    assert "ask" not in help_text
    assert "mcp" not in help_text


def test_session_parser_no_longer_exposes_resume_flag() -> None:
    parser = create_session_parser()
    help_text = parser.format_help()

    assert "--resume" not in help_text
    assert "-r" not in help_text


@pytest.mark.asyncio
async def test_main_short_circuits_root_help(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"help": 0, "version": 0}

    monkeypatch.setattr(bootstrap_app.sys, "argv", ["msagent", "--help"])
    monkeypatch.setattr(bootstrap_app, "render_root_help", lambda: called.__setitem__("help", called["help"] + 1))
    monkeypatch.setattr(
        bootstrap_app,
        "render_version_info",
        lambda: called.__setitem__("version", called["version"] + 1),
    )

    assert await bootstrap_app.main() == 0
    assert called == {"help": 1, "version": 0}


@pytest.mark.asyncio
async def test_main_short_circuits_config_help(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"root_help": 0, "config_help": 0, "version": 0}

    monkeypatch.setattr(bootstrap_app.sys, "argv", ["msagent", "config", "--help"])
    monkeypatch.setattr(
        bootstrap_app,
        "render_root_help",
        lambda: called.__setitem__("root_help", called["root_help"] + 1),
    )
    monkeypatch.setattr(
        bootstrap_app,
        "render_config_help",
        lambda: called.__setitem__("config_help", called["config_help"] + 1),
    )
    monkeypatch.setattr(
        bootstrap_app,
        "render_version_info",
        lambda: called.__setitem__("version", called["version"] + 1),
    )

    assert await bootstrap_app.main() == 0
    assert called == {"root_help": 0, "config_help": 1, "version": 0}


@pytest.mark.asyncio
async def test_main_short_circuits_root_version(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"help": 0, "version": 0}

    monkeypatch.setattr(bootstrap_app.sys, "argv", ["msagent", "-V"])
    monkeypatch.setattr(
        bootstrap_app,
        "render_root_help",
        lambda: called.__setitem__("help", called["help"] + 1),
    )
    monkeypatch.setattr(
        bootstrap_app,
        "render_version_info",
        lambda: called.__setitem__("version", called["version"] + 1),
    )

    assert await bootstrap_app.main() == 0
    assert called == {"help": 0, "version": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("argv", [["--help"], ["-h"], ["config", "--help"], ["config", "-h"], ["--version"], ["-V"]])
async def test_help_and_version_bypass_storage_guard(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
) -> None:
    monkeypatch.setattr(bootstrap_app.sys, "argv", ["msagent", *argv])
    monkeypatch.setattr(bootstrap_app, "render_root_help", lambda: None)
    monkeypatch.setattr(bootstrap_app, "render_config_help", lambda: None)
    monkeypatch.setattr(bootstrap_app, "render_version_info", lambda: None)
    monkeypatch.setattr(
        bootstrap_app,
        "validate_and_initialize_storage_layout",
        lambda _paths: pytest.fail("storage guard should be bypassed"),
    )

    assert await bootstrap_app.main() == 0


@pytest.mark.asyncio
async def test_storage_guard_runs_before_logging_and_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    errors: list[str] = []

    def reject_layout(_paths) -> None:
        events.append("guard")
        raise StorageLayoutError("legacy layout")

    monkeypatch.setattr(bootstrap_app.sys, "argv", ["msagent", "config", "--show"])
    monkeypatch.setattr(bootstrap_app, "validate_and_initialize_storage_layout", reject_layout)
    monkeypatch.setattr(bootstrap_app, "configure_logging", lambda **_kwargs: events.append("logging"))
    monkeypatch.setattr(bootstrap_app.console, "print_error", errors.append)

    assert await bootstrap_app.main() == 1
    assert events == ["guard"]
    assert errors == ["legacy layout"]


def test_render_version_info_formats_version_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []

    monkeypatch.setattr(
        "msagent.cli.bootstrap.legacy._load_version_info",
        lambda: {
            "version": "26.1.0-alpha.2",
            "commit": "bab2dc9",
            "date": "2026-08-15T16:12:09+08:00",
            "repo": "https://gitcode.com/Ascend/msagent",
        },
    )
    monkeypatch.setattr(
        "msagent.cli.bootstrap.legacy.console.print",
        lambda text, **kwargs: printed.append(text),
    )

    render_version_info()

    assert printed
    assert "msagent 26.1.0-alpha.2 (bab2dc9)" in printed[0]
    assert "Date : 2026-08-15T16:12:09+08:00" in printed[0]
    assert "Repo : https://gitcode.com/Ascend/msagent" in printed[0]


def test_render_config_help_uses_custom_template(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []

    monkeypatch.setattr(
        "msagent.cli.bootstrap.legacy.console.print",
        lambda text, **kwargs: printed.append(text),
    )

    render_config_help()

    assert printed
    assert "Description:" in printed[0]
    assert "Usage:\n  msagent config [options]" in printed[0]
    assert "Optional arguments:" in printed[0]
    assert "Examples:" in printed[0]
    assert "Troubleshooting:" in printed[0]
