"""CLI surface that preserves msAgent defaults with a smaller public API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from rich.table import Table

from msagent.cli.bootstrap.chat import handle_chat_command
from msagent.cli.bootstrap.web import handle_web_command
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console
from msagent.configs import ApprovalMode
from msagent.configs.llm import LLMProvider
from msagent.core.constants import APP_NAME, CONFIG_LLMS_FILE_NAME

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
PUBLIC_COMMANDS = {"config", "web"}
ROOT_ONLY_FLAGS = {"-h", "--help", "--version"}


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
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit",
    )
    subparsers = parser.add_subparsers(dest="cli_command", metavar="{config,web}")

    config_parser = subparsers.add_parser("config", help="Configure msAgent")
    config_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging to console and .msagent/app.log",
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
        help="Custom provider base URL, or full request URL for custom",
    )
    config_parser.add_argument("--llm-model", "-m", help="Model name")
    config_parser.add_argument(
        "-w",
        "--working-dir",
        default=os.getcwd(),
        help="Working directory for project-local .msagent config",
    )

    web_parser = subparsers.add_parser(
        "web",
        help="Start a LangGraph server for deep-agents-ui",
    )
    web_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging to console and .msagent/app.log",
    )
    web_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the LangGraph dev server",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=2024,
        help="Port for the LangGraph dev server",
    )
    web_parser.add_argument(
        "--ui-port",
        type=int,
        default=3000,
        help="Port for the official deep-agents-ui frontend",
    )
    web_parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Start only the LangGraph API server without the deep-agents-ui frontend",
    )
    web_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the web UI in the default browser after startup",
    )
    web_parser.add_argument(
        "-w",
        "--working-dir",
        default=os.getcwd(),
        help="Working directory for project-local .msagent config",
    )
    web_parser.add_argument("-a", "--agent", default=None, help="Agent name")
    web_parser.add_argument("-m", "--model", default=None, help="LLM model alias")

    return parser


def create_session_parser() -> argparse.ArgumentParser:
    """Create the internal parser used for the default interactive session."""
    parser = argparse.ArgumentParser(prog=APP_NAME)
    parser.set_defaults(cli_command=DEFAULT_SESSION_COMMAND, version=False)
    parser.add_argument("message", nargs="?", default=None, help="Message to send")
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
        help="Enable verbose logging to console and .msagent/app.log",
    )
    _add_runtime_options(parser, include_timer=True)
    return parser


def _add_runtime_options(
    parser: argparse.ArgumentParser, *, include_timer: bool
) -> None:
    parser.add_argument(
        "-w",
        "--working-dir",
        default=os.getcwd(),
        help="Working directory for the session (default: current directory)",
    )
    parser.add_argument("-a", "--agent", default=None, help="Agent name")
    parser.add_argument("-m", "--model", default=None, help="LLM model alias")
    parser.set_defaults(resume=False)
    if include_timer:
        parser.add_argument(
            "--timer",
            action="store_true",
            help="Enable startup timing",
        )
    else:
        parser.set_defaults(timer=False)
    parser.add_argument(
        "-am",
        "--approval-mode",
        choices=[mode.value for mode in ApprovalMode],
        default=ApprovalMode.ACTIVE.value,
        help="Tool approval mode",
    )


async def dispatch_legacy_command(args: argparse.Namespace) -> int:
    """Dispatch a parsed retained command."""
    if args.version:
        from msagent.utils.version import get_version

        console.print(f"[bold cyan]msAgent[/bold cyan] v{get_version()}")
        return 0

    command = args.cli_command or DEFAULT_SESSION_COMMAND
    if command == DEFAULT_SESSION_COMMAND:
        return await _handle_chat(args)
    if command == "config":
        return await _handle_config(args)
    if command == "web":
        return await _handle_web(args)

    console.print_error(f"Unknown command: {command}")
    console.print("")
    return 1


async def _handle_chat(args: argparse.Namespace) -> int:
    chat_args = argparse.Namespace(
        message=args.message,
        working_dir=args.working_dir,
        agent=args.agent,
        model=args.model,
        resume=False,
        timer=args.timer,
        server=False,
        approval_mode=args.approval_mode,
        verbose=args.verbose,
        stream=args.stream,
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
            console.print_error(
                f"Unsupported provider: {args.llm_provider}. Supported: {supported}"
            )
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
        "base_url": (
            args.llm_base_url if args.llm_base_url is not None else current_llm.base_url
        ),
        "max_tokens": (
            args.llm_max_tokens
            if args.llm_max_tokens is not None
            else current_llm.max_tokens
        ),
        "temperature": current_llm.temperature,
        "streaming": True,
        "request_timeout_seconds": current_llm.request_timeout_seconds,
        "context_window": current_llm.context_window,
    }

    llm_config_path = working_dir / CONFIG_LLMS_FILE_NAME
    llm_config_path.write_text(
        yaml.safe_dump({"llms": [llm_data]}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    await registry.update_agent_llm(agent_config.name, "default")
    registry.invalidate_cache()

    if args.llm_api_key:
        env_name = llm_data.get("api_key_env") or DEFAULT_API_ENV_MAP.get(
            provider or current_llm.provider
        )
        if env_name:
            os.environ[str(env_name)] = args.llm_api_key
            console.print_warning(
                f"Set {env_name} for this process only; it is not persisted to config."
            )

    console.print_success("Configuration saved successfully")
    return 0


async def _handle_web(args: argparse.Namespace) -> int:
    web_args = argparse.Namespace(
        host=args.host,
        port=args.port,
        ui_port=args.ui_port,
        no_ui=args.no_ui,
        no_open=args.no_open,
        working_dir=args.working_dir,
        agent=args.agent,
        model=args.model,
        verbose=args.verbose,
    )
    return await handle_web_command(web_args)


async def _show_config(registry, working_dir: Path) -> int:
    agent_config = await registry.get_agent(None)
    llm_config = agent_config.llm
    mcp_config = await registry.load_mcp()

    provider_label = (
        "gemini"
        if llm_config.provider == LLMProvider.GOOGLE
        else llm_config.provider.value
    )
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

    console.print(f"\n[dim]Config dir: {working_dir / '.msagent'}[/dim]")
    return 0
