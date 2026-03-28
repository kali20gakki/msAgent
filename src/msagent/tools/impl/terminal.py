import re
import shlex
from typing import Annotated

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import ToolException
from pydantic import Field

from msagent.agents.context import AgentContext
from msagent.cli.theme import theme
from msagent.core.logging import get_logger
from msagent.middlewares.approval import create_field_transformer
from msagent.utils.bash import execute_bash_command
from msagent.utils.path import resolve_path

logger = get_logger(__name__)

DEFAULT_COMMAND_TIMEOUT_SECONDS = 1800
COMMAND_TIMEOUT_SENTINEL = "Command timed out"
TIMEOUT_OUTPUT_TAIL_LINES = 50


_CHAIN_OPS = re.compile(r"\s*(&&|\|\||;|\|)\s*")
_SUBST_DOLLAR = re.compile(r"\$\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
_SUBST_BACKTICK = re.compile(r"`([^`]+)`")


def _extract_command_parts(command: str) -> list[str]:
    """Extract all command parts including nested $(...) and `...` substitutions."""
    parts = []
    for seg in _CHAIN_OPS.split(command):
        seg = seg.strip()
        if not seg or seg in ("&&", "||", ";", "|"):
            continue
        parts.append(seg)
        for pattern in (_SUBST_DOLLAR, _SUBST_BACKTICK):
            for m in pattern.finditer(seg):
                parts.extend(_extract_command_parts(m.group(1)))
    return parts


def _first_n_words(cmd: str, n: int = 3) -> str:
    """Extract first n words from a command, handling shell quoting."""
    try:
        words = shlex.split(cmd, posix=True)[:n]
    except ValueError:
        words = cmd.split()[:n]
    return " ".join(words)


def _transform_command_for_approval(command: str) -> str:
    """Transform command to first 3 words of each part for pattern matching."""
    parts = [_first_n_words(p) for p in _extract_command_parts(command) if p]
    return " && ".join(parts) if parts else command


def _render_command_args(args: dict, config: dict) -> str:
    """Render command arguments with syntax highlighting."""
    command = args.get("command", "")
    timeout_seconds = args.get("timeout_seconds", DEFAULT_COMMAND_TIMEOUT_SECONDS)

    rendered = f"[{theme.indicator_color}]{command}[/{theme.indicator_color}]"
    if timeout_seconds != DEFAULT_COMMAND_TIMEOUT_SECONDS:
        rendered += (
            f"\n[{theme.muted_text}]timeout_seconds={timeout_seconds}[/{theme.muted_text}]"
        )
    return rendered


def _build_timeout_error_message(
    timeout_seconds: int,
    stdout: str,
    stderr: str,
) -> str:
    """Format a timeout error with any output captured before the process was killed."""
    partial_sections: list[str] = []

    stdout_section = _format_timeout_output_section("stdout", stdout)
    if stdout_section is not None:
        partial_sections.append(stdout_section)

    stderr_detail = stderr.strip()
    if stderr_detail.startswith(COMMAND_TIMEOUT_SENTINEL):
        stderr_detail = stderr_detail[len(COMMAND_TIMEOUT_SENTINEL) :].strip()

    stderr_section = _format_timeout_output_section("stderr", stderr_detail)
    if stderr_section is not None:
        partial_sections.append(stderr_section)

    message = f"Command timed out after {timeout_seconds}s"
    if partial_sections:
        message += "\n\nPartial output before timeout:\n" + "\n\n".join(partial_sections)

    return message


def _tail_output_lines(text: str, max_lines: int = TIMEOUT_OUTPUT_TAIL_LINES) -> tuple[str, int]:
    """Keep only the last max_lines of output and report how many lines were omitted."""
    stripped = text.strip()
    if not stripped:
        return "", 0

    lines = stripped.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines), 0

    tail_lines = lines[-max_lines:]
    return "\n".join(tail_lines), len(lines) - max_lines


def _format_timeout_output_section(label: str, text: str) -> str | None:
    """Format timeout output for one stream, keeping only the most recent lines."""
    tailed_text, omitted_lines = _tail_output_lines(text)
    if not tailed_text:
        return None

    header = f"[{label}]"
    if omitted_lines > 0:
        header += (
            f" last {TIMEOUT_OUTPUT_TAIL_LINES} lines "
            f"(omitted {omitted_lines} earlier lines)"
        )

    return f"{header}\n{tailed_text}"


@tool
async def run_command(
    command: str,
    runtime: ToolRuntime[AgentContext],
    timeout_seconds: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Timeout in seconds for the command. Defaults to 1800 seconds. "
                "Increase this for long-running tasks such as builds, test suites, "
                "downloads, or data processing jobs."
            ),
        ),
    ] = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> str:
    """
    Use this tool to execute terminal commands. Project files should be checked first to understand
    available commands and project structure before running unfamiliar operations.
    For long-running tasks, increase timeout_seconds instead of assuming the command hung.

    Args:
        command: The command to execute
        timeout_seconds: Maximum runtime in seconds. Increase this for long-running tasks.
    """
    context: AgentContext = runtime.context
    status, stdout, stderr = await execute_bash_command(
        ["bash", "-c", command],
        cwd=str(context.working_dir),
        timeout=timeout_seconds,
    )
    if status != 0:
        timed_out = status == -1 and stderr.strip().startswith(COMMAND_TIMEOUT_SENTINEL)
        error_msg = (
            _build_timeout_error_message(timeout_seconds, stdout, stderr)
            if timed_out
            else (
                stderr.strip()
                if stderr.strip()
                else f"Command failed with exit code {status}"
            )
        )
        raise ToolException(error_msg)

    output_parts = []
    if stdout.strip():
        output_parts.append(stdout.strip())
    if stderr.strip():
        output_parts.append(stderr.strip())

    return "\n".join(output_parts) if output_parts else "Command completed successfully"


run_command.metadata = {
    "approval_config": {
        "format_args_fn": create_field_transformer(
            {"command": _transform_command_for_approval}
        ),
        "render_args_fn": _render_command_args,
    }
}


@tool
async def get_directory_structure(
    dir_path: str,
    runtime: ToolRuntime[AgentContext],
) -> ToolMessage:
    """
    Use this tool to get a tree view of a directory structure showing all files and folders, highly recommended before running file operations.

    Args:
        dir_path: Path to the directory (relative to working directory or absolute)
    """
    context: AgentContext = runtime.context
    working_dir = str(context.working_dir)

    resolved_path = resolve_path(working_dir, dir_path)
    absolute_dir_path = str(resolved_path)

    safe_dir = shlex.quote(absolute_dir_path)
    cmd = [
        "bash",
        "-c",
        f"cd {safe_dir} && ls"
    ]
    status, stdout, stderr = await execute_bash_command(cmd, cwd=working_dir)
    if status not in (0, 1):
        raise ToolException(stderr)

    short_content = f"Retrieved directory tree for {absolute_dir_path}"

    return ToolMessage(
        name=get_directory_structure.name,
        content=stdout,
        tool_call_id=runtime.tool_call_id,
        short_content=short_content,
    )


get_directory_structure.metadata = {
    "approval_config": {
        "name_only": True,
        "always_approve": True,
    }
}


TERMINAL_TOOLS = [run_command, get_directory_structure]
