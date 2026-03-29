from msagent.cli.bootstrap.app import create_parser
from msagent.cli.bootstrap.legacy import (
    DEFAULT_SESSION_COMMAND,
    create_session_parser,
    normalize_argv,
)
from msagent.core.constants import APP_NAME


def test_create_session_parser_defaults_to_interactive_mode() -> None:
    parser = create_session_parser()
    args = parser.parse_args([])

    assert parser.prog == APP_NAME == "msagent"
    assert args.message is None
    assert args.cli_command == DEFAULT_SESSION_COMMAND
    assert args.resume is False
    assert args.stream is True


def test_create_session_parser_accepts_explicit_agent_selection() -> None:
    parser = create_session_parser()
    args = parser.parse_args(["--agent", "Minos"])

    assert args.agent == "Minos"
    assert args.message is None


def test_normalize_argv_routes_messages_to_default_session() -> None:
    assert normalize_argv(["hello"]) == [DEFAULT_SESSION_COMMAND, "hello"]
    assert normalize_argv(["--agent", "Minos"]) == [
        DEFAULT_SESSION_COMMAND,
        "--agent",
        "Minos",
    ]
    assert normalize_argv(["config", "--show"]) == ["config", "--show"]
    assert normalize_argv(["web", "--host", "0.0.0.0"]) == ["web", "--host", "0.0.0.0"]


def test_help_only_exposes_public_commands_only() -> None:
    parser = create_parser()
    help_text = parser.format_help()

    assert "config" in help_text
    assert "web" in help_text
    assert DEFAULT_SESSION_COMMAND not in help_text
    assert "chat" not in help_text
    assert "ask" not in help_text
    assert "mcp" not in help_text


def test_web_parser_exposes_host_and_port() -> None:
    parser = create_parser()
    args = parser.parse_args(
        ["web", "--host", "0.0.0.0", "--port", "3030", "--ui-port", "3001", "--no-open"]
    )

    assert args.cli_command == "web"
    assert args.host == "0.0.0.0"
    assert args.port == 3030
    assert args.ui_port == 3001
    assert args.no_ui is False
    assert args.no_open is True


def test_session_parser_no_longer_exposes_resume_flag() -> None:
    parser = create_session_parser()
    help_text = parser.format_help()

    assert "--resume" not in help_text
    assert "-r" not in help_text
