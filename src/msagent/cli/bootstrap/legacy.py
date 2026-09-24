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

"""CLI surface that preserves msAgent defaults with a smaller public API."""

from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path

import yaml
from rich.table import Table

from msagent.cli.bootstrap.chat import handle_chat_command
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console
from msagent.configs import ApprovalMode
from msagent.configs.llm import LLMProvider
from msagent.core.constants import APP_NAME

LEGACY_PROVIDER_MAP = {
    "openai": LLMProvider.OPENAI,
    "anthropic": LLMProvider.ANTHROPIC,
    "gemini": LLMProvider.GOOGLE,
    "google": LLMProvider.GOOGLE,
}

DEFAULT_API_ENV_MAP = {
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
    LLMProvider.GOOGLE: "GOOGLE_API_KEY",
}

DEFAULT_SESSION_COMMAND = "__session__"
PUBLIC_COMMANDS = {"config"}
ROOT_ONLY_FLAGS = {"--help", "-h", "--version", "-V"}

AGENT_HELP = (
    "Agent name. Available agents:\n"
    "Profiler  Performance profiling and optimization.\n"
    "Accuracy  Accuracy analysis and debugging.\n"
    "Quantizer Model quantization and adaptation.\n"
    "Modeling  LLM/VLM simulation modeling.\n"
    "Operator  Ascend operator performance tuning.\n"
    "Minos     Documentation UX and code review."
)

SESSION_DESCRIPTION = "Start a chat session with msAgent.\n\nsubcommands:\n  config      Configure msAgent"

ROOT_HELP_TEXT = """Description:
  Start an interactive msagent session, or use subcommands to configure the
  local project settings.

Usage:
  msagent [message] [options]
  msagent config [options]

Optional arguments:
  -h, --help                               Show help message and exit.
  -V, --version                            Show version information and exit.
      --stream                             Stream output.
      --no-stream                          Render the final reply without token streaming.
  -v, --verbose                            Enable verbose logging to console and ~/.msagent/logs.
  -w, --working-dir <DIR>                  Working directory for the session [default: current directory]
  -a, --agent <NAME>                       Agent name. Available: Profiler, Accuracy, Quantizer, Modeling, Operator, Minos.
  -m, --model <NAME>                       LLM model alias.
      --timer                              Enable startup timing.
  -am, --approval-mode {semi-active,active,aggressive}
                                           Tool approval mode [default: active]
      --execute-approval-mode {safe,convenience}
                                           Preset shell execute approval mode for non-interactive runs.
      --trace-jsonl <FILE>                 Write JSONL trace events to this file.

Examples:
  # Start an interactive session in the current directory
  msagent

  # Send a message directly to the default agent
  msagent "analyze profiling"

  # Start a session with a specific agent and model
  msagent --agent Profiler --model deepseek-v4-flash

  # Show current project-local configuration
  msagent config --show

Troubleshooting:
  - "command not found": make sure msagent is installed in the current Python environment.
  - Missing model response: run `msagent config --show` and verify the configured LLM provider and model.
  - Permission or path errors: check that --working-dir points to a writable project directory.
"""

CONFIG_HELP_TEXT = """Description:
  Configure project-local msagent settings, or display the current
  configuration.

Usage:
  msagent config [options]

Optional arguments:
  -h, --help                               Show help message and exit.
  -v, --verbose                            Enable verbose logging to console and ~/.msagent/logs.
  -s, --show                               Show current configuration.
      --llm-provider <NAME>                LLM provider. Available: openai, anthropic, gemini, google.
      --llm-api-key <KEY>                  LLM API key for this process only.
      --llm-max-tokens <INT>               Max output tokens (0 means provider/model default).
      --llm-base-url <URL>                 Custom provider base URL for a compatible service or proxy.
  -m, --llm-model <NAME>                   Model name.
  -w, --working-dir <DIR>                  Project working directory [default: current directory]

Examples:
  # Show current project-local configuration
  msagent config --show

  # Update the default provider and model
  msagent config --llm-provider openai --llm-model gpt-5

  # Use a custom compatible endpoint for the current project
  msagent config --llm-provider openai --llm-base-url http://127.0.0.1:8000/v1

Troubleshooting:
  - Unsupported provider: use one of openai, anthropic, gemini, or google.
  - API key not persisted: `--llm-api-key` only applies to the current process.
  - Config not taking effect: confirm `--working-dir` points to the intended project directory.
"""

VERSION_BANNER = """=================================================================
                   >>>>>   MindStudio   <<<<<
    THE END-TO-END TOOLCHAIN TO UNLEASH HUAWEI ASCEND COMPUTE
================================================================="""

VERSION_INFO_DEFAULTS = {
    "version": "",
    "commit": "unknown",
    "date": "unknown",
    "repo": "https://gitcode.com/Ascend/msagent",
}


def _load_version_info() -> dict[str, str]:
    for path in (
        Path(__file__).resolve().parents[2] / "version.info",
        Path(__file__).resolve().parents[4] / "version.info",
    ):
        if not path.exists():
            continue
        try:
            parser = configparser.ConfigParser()
            parser.read(path, encoding="utf-8")
            section = parser["PACKAGE"]
            return {
                key: section.get(key.capitalize(), default).strip() or default
                for key, default in VERSION_INFO_DEFAULTS.items()
            }
        except Exception:
            continue

    return VERSION_INFO_DEFAULTS.copy()


def render_root_help() -> None:
    console.print(ROOT_HELP_TEXT, end="", markup=False)


def render_config_help() -> None:
    console.print(CONFIG_HELP_TEXT, end="", markup=False)


def _resolve_copyright_year(build_date: str) -> str:
    return build_date[:4] if len(build_date) >= 4 and build_date[:4].isdigit() else "2026"


def render_version_info() -> None:
    version_info = _load_version_info()
    copyright_year = _resolve_copyright_year(version_info["date"])
    version_text = (
        f"{VERSION_BANNER}\n"
        f"{APP_NAME} {version_info.get('version') or 'unknown'} ({version_info['commit']})\n"
        f"Copyright (C) {copyright_year} Huawei Technologies Co., Ltd.\n"
        "License: Mulan PSL v2.\n\n"
        "Build Info:\n"
        f"  Date : {version_info['date']}\n"
        f"  Repo : {version_info['repo']}\n"
    )
    console.print(version_text, end="", markup=False)


def normalize_argv(argv: list[str]) -> list[str]:
    """Route bare invocations to the default interactive session."""
    if not argv:
        return [DEFAULT_SESSION_COMMAND]
    if argv[0] in ROOT_ONLY_FLAGS or argv[0] in PUBLIC_COMMANDS:
        return argv
    return [DEFAULT_SESSION_COMMAND, *argv]


def create_legacy_parser() -> argparse.ArgumentParser:
    """Create a parser with only the retained public commands."""
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description="msAgent - AI Assistant with MCP support",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Show version information and exit",
    )
    subparsers = parser.add_subparsers(dest="cli_command", metavar="{config}")

    config_parser = subparsers.add_parser("config", help="Configure msAgent")
    config_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging to console and ~/.msagent/logs",
    )
    config_parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Show current configuration",
    )
    config_parser.add_argument("--llm-provider", help="LLM provider")
    config_parser.add_argument(
        "--llm-api-key",
        help="LLM API key for this process only",
    )
    config_parser.add_argument(
        "--llm-max-tokens",
        type=int,
        help="Max output tokens (0 means provider/model default)",
    )
    config_parser.add_argument(
        "--llm-base-url",
        help="Custom provider base URL for a compatible service or proxy",
    )
    config_parser.add_argument("--llm-model", "-m", help="Model name")
    config_parser.add_argument(
        "-w",
        "--working-dir",
        default=os.getcwd(),
        help="Project working directory",
    )

    return parser


def create_session_parser() -> argparse.ArgumentParser:
    """Create the internal parser used for the default interactive session."""
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description=SESSION_DESCRIPTION,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.set_defaults(cli_command=DEFAULT_SESSION_COMMAND, version=False, resume=False)
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help='Message to send.\nexample: msagent "analyze profiling"',
    )
    parser.add_argument(
        "--stream",
        dest="stream",
        action="store_true",
        default=True,
        help="Stream output",
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="Render the final reply without token streaming",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging to console and ~/.msagent/logs",
    )
    _add_runtime_options(parser, include_timer=True)
    return parser


def _add_runtime_options(parser: argparse.ArgumentParser, *, include_timer: bool) -> None:
    parser.add_argument(
        "-w",
        "--working-dir",
        default=os.getcwd(),
        help="Working directory for the session (default: current directory)",
    )
    parser.add_argument("-a", "--agent", default=None, help=AGENT_HELP)
    parser.add_argument("-m", "--model", default=None, help="LLM model alias")
    if include_timer:
        parser.add_argument(
            "--timer",
            action="store_true",
            help="Enable startup timing.\nexample: msagent --timer",
        )
    else:
        parser.set_defaults(timer=False)
    parser.add_argument(
        "-am",
        "--approval-mode",
        choices=[mode.value for mode in ApprovalMode],
        default=ApprovalMode.ACTIVE.value,
        help="Tool approval mode (default: active)",
    )
    parser.add_argument(
        "--execute-approval-mode",
        choices=["safe", "convenience"],
        default=None,
        help="Preset shell execute approval mode for non-interactive runs",
    )
    parser.add_argument(
        "--trace-jsonl",
        default=None,
        help="Write JSONL trace events to this file",
    )


async def dispatch_legacy_command(args: argparse.Namespace) -> int:
    """Dispatch a parsed retained command."""
    if args.version:
        render_version_info()
        return 0

    command = args.cli_command or DEFAULT_SESSION_COMMAND
    if command == DEFAULT_SESSION_COMMAND:
        return await _handle_chat(args)
    if command == "config":
        return await _handle_config(args)

    console.print_error(f"Unknown command: {command}")
    console.print("")
    return 1


async def _handle_chat(args: argparse.Namespace) -> int:
    chat_args = argparse.Namespace(
        message=args.message,
        working_dir=args.working_dir,
        agent=args.agent,
        model=args.model,
        timer=args.timer,
        server=False,
        approval_mode=args.approval_mode,
        execute_approval_mode=args.execute_approval_mode,
        verbose=args.verbose,
        stream=args.stream,
        trace_jsonl=args.trace_jsonl,
    )
    return await handle_chat_command(chat_args)


async def _handle_config(args: argparse.Namespace) -> int:
    working_dir = Path(args.working_dir)
    registry = initializer.get_registry(working_dir)
    await registry.ensure_config_dir()

    if args.show or not any(
        [
            args.llm_provider,
            args.llm_api_key,
            args.llm_max_tokens is not None,
            args.llm_base_url,
            args.llm_model,
        ]
    ):
        return await _show_config(registry, working_dir)

    provider = None
    if args.llm_provider:
        provider = LEGACY_PROVIDER_MAP.get(args.llm_provider.lower().strip())
        if provider is None:
            supported = ", ".join(sorted(LEGACY_PROVIDER_MAP))
            console.print_error(f"Unsupported provider: {args.llm_provider}. Supported: {supported}")
            console.print("")
            return 1

    agent_config = await registry.get_agent(None)
    current_llm = agent_config.llm
    effective_provider = provider or current_llm.provider
    llm_data = {
        "version": current_llm.version,
        "provider": effective_provider.value,
        "alias": "default",
        "model": args.llm_model or current_llm.model,
        "api_key_env": DEFAULT_API_ENV_MAP.get(effective_provider),
        "base_url": (args.llm_base_url if args.llm_base_url is not None else current_llm.base_url),
        "max_tokens": (args.llm_max_tokens if args.llm_max_tokens is not None else current_llm.max_tokens),
        "temperature": current_llm.temperature,
        "streaming": True,
        "request_timeout_seconds": current_llm.request_timeout_seconds,
        "context_window": current_llm.context_window,
    }

    llm_config_path = registry.llms_file
    llm_config_path.parent.mkdir(parents=True, exist_ok=True)
    llm_config_path.write_text(
        yaml.safe_dump({"llms": [llm_data]}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    registry.invalidate_cache()

    if args.llm_api_key:
        env_name = llm_data.get("api_key_env") or DEFAULT_API_ENV_MAP.get(provider or current_llm.provider)
        if env_name:
            os.environ[str(env_name)] = args.llm_api_key
            console.print_warning(f"Set {env_name} for this process only; it is not persisted to config.")

    console.print_success("Configuration saved successfully")
    return 0


async def _show_config(registry, working_dir: Path) -> int:
    agent_config = await registry.get_agent(None)
    workspace_model = await registry.get_current_model(agent_config.name)
    llm_config = await registry.get_llm(workspace_model) if workspace_model else agent_config.llm
    mcp_config = await registry.load_mcp()

    provider_label = "gemini" if llm_config.provider == LLMProvider.GOOGLE else llm_config.provider.value
    api_env = llm_config.api_key_env or DEFAULT_API_ENV_MAP.get(llm_config.provider, "")
    api_key_set = bool(os.getenv(api_env, "")) if api_env else False

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Agent", agent_config.name)
    table.add_row("LLM Provider", provider_label)
    table.add_row("Model", llm_config.model)
    table.add_row("API Key", "Set" if api_key_set else "Not set")
    table.add_row("API Key Env", api_env or "Not configured")
    table.add_row("Base URL", llm_config.base_url or "Default")
    table.add_row(
        "Max Tokens",
        "Auto" if llm_config.max_tokens <= 0 else str(llm_config.max_tokens),
    )
    table.add_row(
        "MCP Servers",
        str(len([server for server in mcp_config.servers.values() if server.enabled])),
    )
    console.print(table)

    if mcp_config.servers:
        mcp_table = Table(title="MCP Servers")
        mcp_table.add_column("Name", style="cyan")
        mcp_table.add_column("Command", style="green")
        mcp_table.add_column("Arguments", style="blue")
        mcp_table.add_column("Status", style="yellow")
        for name, server in sorted(mcp_config.servers.items()):
            mcp_table.add_row(
                name,
                server.command or server.url or "",
                " ".join(server.args) if server.args else "None",
                "Enabled" if server.enabled else "Disabled",
            )
        console.print(mcp_table)

    console.print(f"\n[dim]Config dir: {registry.config_dir}[/dim]")
    return 0
