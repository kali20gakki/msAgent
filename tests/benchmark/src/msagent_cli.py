from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

from third_party.codex_cli import (
    AGENT_OUTPUT_SCHEMA,
    JUDGE_OUTPUT_SCHEMA,
    build_agent_prompt,
    build_judge_prompt,
    copy_case_skill,
    copy_input_data,
    display_path,
    safe_command_for_trace,
    safe_path_component,
    trim_event_text,
    trim_text,
)
from schema import BenchmarkCase
from trace import TraceBuilder  # pylint: disable=no-name-in-module


DEFAULT_MSAGENT_AGENT = "Hermes"
DEFAULT_MSAGENT_APPROVAL_MODE = "aggressive"


class MsagentCliUnavailableError(RuntimeError):
    pass


class MsagentCliAgent:
    agent_info = {
        "name": "msagent-cli-agent",
        "runtime": "msagent one-shot",
        "token_usage_mode": "msagent-cli-jsonl",
    }

    def __init__(
        self,
        *,
        workspace: Path,
        artifact_dir: Path,
        model: str | None = None,
        timeout_seconds: int = 900,
        msagent_agent: str | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.msagent_agent = msagent_agent or os.environ.get("MSAGENT_AGENT") or DEFAULT_MSAGENT_AGENT
        self.msagent_command = resolve_msagent_cli_command()
        self.approval_mode = os.environ.get("MSAGENT_APPROVAL_MODE") or DEFAULT_MSAGENT_APPROVAL_MODE

    def run(self, case: BenchmarkCase, input_path: Path) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_text = json.dumps(AGENT_OUTPUT_SCHEMA, indent=2)
        schema_artifact_path = self.artifact_dir / "agent_output.schema.json"
        schema_artifact_path.write_text(schema_text, encoding="utf-8")
        final_artifact_path = self.artifact_dir / f"{case.id}.agent.final.json"
        stdout_path = self.artifact_dir / f"{case.id}.agent.stdout.txt"
        stderr_path = self.artifact_dir / f"{case.id}.agent.stderr.txt"
        trace_jsonl_path = self.artifact_dir / f"{case.id}.agent.msagent.events.jsonl"

        prefix = f"benchmark-builder-msagent-agent-{safe_path_component(case.id)}-"
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            run_workspace = Path(temp_dir).resolve()
            isolated_input_path = run_workspace / "input_data"
            copy_input_data(input_path, isolated_input_path)
            isolated_skill_path = copy_case_skill(case, run_workspace)
            copy_msagent_config(self.workspace, run_workspace)

            prompt = build_msagent_agent_prompt(
                case,
                isolated_input_path,
                visible_input_path=isolated_input_path,
                visible_skill_path=isolated_skill_path,
                schema_text=schema_text,
            )
            cmd = build_msagent_command(
                self.msagent_command,
                working_dir=run_workspace,
                msagent_agent=self.msagent_agent,
                model=self.model,
                approval_mode=self.approval_mode,
                trace_jsonl_path=trace_jsonl_path,
                prompt=prompt,
            )

            started = perf_counter()
            completed = subprocess.run(
                cmd,
                cwd=run_workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=build_msagent_env(run_workspace),
            )
            duration_ms = round((perf_counter() - started) * 1000)
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            msagent_events = read_msagent_trace_jsonl(trace_jsonl_path)
            token_usage = msagent_token_usage(msagent_events)
            session_duration_ms = msagent_session_duration_ms(msagent_events)

            if completed.returncode != 0:
                raise RuntimeError(
                    "msagent failed for agent run "
                    f"{case.id} with exit code {completed.returncode}: {completed.stderr[-4000:]}"
                )
            final_answer = read_msagent_structured_final(
                completed.stdout,
                required_keys={"answer"},
                msagent_events=msagent_events,
            )
            final_artifact_path.write_text(
                json.dumps(final_answer, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            trace = TraceBuilder(
                case_id=case.id,
                prompt=case.prompt,
                agent={
                    **self.agent_info,
                    "model": self.model,
                    "cli_command": self.msagent_command,
                    "msagent_agent": self.msagent_agent,
                    "approval_mode": self.approval_mode,
                    "isolation": "temp-workspace-copied-input",
                },
            )
            trace.add(
                "agent_run",
                command=safe_command_for_trace(cmd),
                cwd=str(run_workspace),
                input_data_path="input_data",
                input_data_isolation="copied",
                skill_path="skill" if isolated_skill_path is not None else None,
                skill_isolation="copied" if isolated_skill_path is not None else None,
                stdout_path=display_path(stdout_path, self.workspace),
                stderr_path=display_path(stderr_path, self.workspace),
                msagent_trace_path=display_path(trace_jsonl_path, self.workspace),
                msagent_trace_event_count=len(msagent_events),
                msagent_session_duration_ms=session_duration_ms,
                stderr_tail=completed.stderr[-2000:],
                duration_ms=duration_ms,
            )
            trace_events = normalize_msagent_trace_events(msagent_events)
            if not trace_events:
                trace_events = normalize_msagent_stdout(completed.stdout)
            for event in trace_events:
                trace.add(**event)
            trace.final_answer(final_answer)
            trace.finish(token_usage)
            trace.duration_ms = duration_ms
            return trace.to_dict()


class MsagentCliJudge:
    judge_info = {
        "name": "msagent-cli-judge",
        "runtime": "msagent one-shot",
        "token_usage_mode": "msagent-cli-jsonl",
    }

    def __init__(
        self,
        *,
        workspace: Path,
        artifact_dir: Path,
        model: str | None = None,
        timeout_seconds: int = 900,
        msagent_agent: str | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.artifact_dir = artifact_dir.resolve()
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.msagent_agent = msagent_agent or os.environ.get("MSAGENT_AGENT") or DEFAULT_MSAGENT_AGENT
        self.msagent_command = resolve_msagent_cli_command()
        self.approval_mode = os.environ.get("MSAGENT_APPROVAL_MODE") or DEFAULT_MSAGENT_APPROVAL_MODE

    def judge(
        self,
        case: BenchmarkCase,
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_text = json.dumps(JUDGE_OUTPUT_SCHEMA, indent=2)
        schema_artifact_path = self.artifact_dir / "judge_output.schema.json"
        schema_artifact_path.write_text(schema_text, encoding="utf-8")
        final_artifact_path = self.artifact_dir / f"{case.id}.judge.final.json"
        stdout_path = self.artifact_dir / f"{case.id}.judge.stdout.txt"
        stderr_path = self.artifact_dir / f"{case.id}.judge.stderr.txt"
        trace_jsonl_path = self.artifact_dir / f"{case.id}.judge.msagent.events.jsonl"

        prompt = build_msagent_judge_prompt(
            case,
            trace,
            schema_text=schema_text,
        )
        prefix = f"benchmark-builder-msagent-judge-{safe_path_component(case.id)}-"
        with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
            run_workspace = Path(temp_dir).resolve()
            copy_msagent_config(self.workspace, run_workspace)
            cmd = build_msagent_command(
                self.msagent_command,
                working_dir=run_workspace,
                msagent_agent=self.msagent_agent,
                model=self.model,
                approval_mode=self.approval_mode,
                trace_jsonl_path=trace_jsonl_path,
                prompt=prompt,
            )

            started = perf_counter()
            completed = subprocess.run(
                cmd,
                cwd=run_workspace,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=build_msagent_env(run_workspace),
            )
            duration_ms = round((perf_counter() - started) * 1000)
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            msagent_events = read_msagent_trace_jsonl(trace_jsonl_path)
            token_usage = msagent_token_usage(msagent_events)
            session_duration_ms = msagent_session_duration_ms(msagent_events)

            if completed.returncode != 0:
                raise RuntimeError(
                    "msagent failed for judge run "
                    f"{case.id} with exit code {completed.returncode}: {completed.stderr[-4000:]}"
                )
            judge_result = read_msagent_structured_final(
                completed.stdout,
                required_keys={"rubric_score"},
                msagent_events=msagent_events,
            )
            final_artifact_path.write_text(
                json.dumps(judge_result, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            return {
                "case_id": case.id,
                "judge": {
                    **self.judge_info,
                    "model": self.model,
                    "cli_command": self.msagent_command,
                    "msagent_agent": self.msagent_agent,
                    "approval_mode": self.approval_mode,
                    "isolation": "temp-workspace-no-repo",
                },
                **judge_result,
                "duration_ms": duration_ms,
                "msagent_session_duration_ms": session_duration_ms,
                "token_usage": token_usage,
                "stdout_path": display_path(stdout_path, self.workspace),
                "stderr_path": display_path(stderr_path, self.workspace),
                "msagent_trace_path": display_path(trace_jsonl_path, self.workspace),
                "msagent_trace_event_count": len(msagent_events),
            }


def resolve_msagent_cli_command() -> list[str]:
    configured = os.environ.get("MSAGENT_CLI")
    if configured:
        configured_parts = shlex.split(configured)
        if not configured_parts:
            raise MsagentCliUnavailableError("MSAGENT_CLI is set but empty.")
        executable = configured_parts[0]
        resolved = resolve_executable(executable)
        if resolved:
            return [str(resolved), *configured_parts[1:]]
        raise MsagentCliUnavailableError(
            f"MSAGENT_CLI executable was not found: {executable}. Use an absolute path or add it to PATH."
        )

    msagent_path = resolve_executable("msagent")
    if msagent_path:
        return [str(msagent_path)]
    raise MsagentCliUnavailableError(
        "msAgent CLI was not found. Install mindstudio-agent or set MSAGENT_CLI, "
        'for example MSAGENT_CLI="uv --project /path/to/msagent run msagent".'
    )


def resolve_executable(executable: str) -> Path | None:
    executable_path = Path(executable).expanduser()
    if executable_path.exists():
        return executable_path.resolve()

    resolved = shutil.which(executable)
    if resolved:
        return Path(resolved).resolve()

    for directory in user_executable_search_paths():
        candidate = directory / executable
        if candidate.exists():
            return candidate.resolve()
    return None


def user_executable_search_paths() -> list[Path]:
    home = Path.home()
    paths = [home / ".local" / "bin"]
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            paths.append(Path(appdata) / "Python" / "Scripts")
    else:
        paths.extend(
            [
                home / ".cargo" / "bin",
                home / ".rye" / "shims",
                Path("/opt/homebrew/bin"),
                Path("/usr/local/bin"),
            ]
        )
    return paths


def build_msagent_command(
    msagent_command: list[str],
    *,
    working_dir: Path,
    msagent_agent: str,
    model: str | None,
    approval_mode: str,
    trace_jsonl_path: Path | None,
    prompt: str,
) -> list[str]:
    cmd = [
        *msagent_command,
        "--no-stream",
        "--working-dir",
        str(working_dir),
        "--agent",
        msagent_agent,
        "--approval-mode",
        approval_mode,
    ]
    if trace_jsonl_path is not None:
        cmd.extend(["--trace-jsonl", str(trace_jsonl_path)])
    if model:
        cmd.extend(["--model", model])
    cmd.append(prompt)
    return cmd


def build_msagent_env(working_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = resolve_home_dir()
    user = os.environ.get("USER") or Path(home).name
    env["HOME"] = home
    env["USER"] = user
    env["LOGNAME"] = os.environ.get("LOGNAME") or user
    env["PWD"] = str(working_dir)
    env.setdefault("NO_COLOR", "1")
    env.setdefault("CLICOLOR", "0")
    env.setdefault("TERM", "dumb")
    env.setdefault("COLUMNS", "20000")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def resolve_home_dir() -> str:
    home = os.environ.get("HOME")
    if home:
        return home
    return str(Path.home())


def copy_msagent_config(source_workspace: Path, run_workspace: Path) -> None:
    source_config = source_workspace / ".msagent"
    if not source_config.is_dir() or source_config.is_symlink():
        return

    destination_config = run_workspace / ".msagent"
    ignored_dirs = {".git", "logs", "__pycache__"}
    ignored_files = {".history"}
    ignored_prefixes = ("config.checkpoints.db",)
    for root, dirs, files in os.walk(source_config, followlinks=False):
        root_path = Path(root)
        relative_root = root_path.relative_to(source_config)
        target_root = destination_config / relative_root
        target_root.mkdir(parents=True, exist_ok=True)

        dirs[:] = [
            dirname for dirname in dirs if dirname not in ignored_dirs and not (root_path / dirname).is_symlink()
        ]
        for filename in files:
            if filename in ignored_files or filename.startswith(ignored_prefixes):
                continue
            source_file = root_path / filename
            if source_file.is_symlink():
                continue
            shutil.copy2(source_file, target_root / filename)
    clear_copied_agent_defaults(destination_config / "agents")
    inject_stdio_mcp_environment(destination_config / "config.mcp.json")


def clear_copied_agent_defaults(agents_dir: Path) -> None:
    if not agents_dir.is_dir():
        return
    for agent_file in agents_dir.glob("*.yml"):
        text = agent_file.read_text(encoding="utf-8")
        updated = re.sub(r"(?m)^(\s*default:\s*)true\s*$", r"\1false", text)
        if updated != text:
            agent_file.write_text(updated, encoding="utf-8")


def inject_stdio_mcp_environment(config_path: Path) -> None:
    if not config_path.exists():
        return

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    servers = config.get("mcpServers")
    if not isinstance(servers, dict):
        return

    inherited_env = msagent_stdio_mcp_env()
    changed = False
    for server in servers.values():
        if not isinstance(server, dict) or server.get("transport") != "stdio":
            continue
        server_env = server.setdefault("env", {})
        if not isinstance(server_env, dict):
            server_env = {}
            server["env"] = server_env
        for key, value in inherited_env.items():
            if key not in server_env and value:
                server_env[key] = value
                changed = True

    if changed:
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def msagent_stdio_mcp_env() -> dict[str, str]:
    home = resolve_home_dir()
    user = os.environ.get("USER") or Path(home).name
    values = {
        "HOME": home,
        "USER": user,
        "LOGNAME": os.environ.get("LOGNAME") or user,
        "PATH": os.environ.get("PATH", ""),
        "SHELL": os.environ.get("SHELL", ""),
        "TERM": os.environ.get("TERM", "dumb"),
    }
    return {key: value for key, value in values.items() if value}


_TILDE_NEEDING_GUARD = re.compile(r"~(?=[^\s/])")


def sanitize_msagent_prompt(text: str) -> str:
    """Replace tildes that msagent would treat as bogus home references.

    msagent's reference resolver runs ``Path(token).expanduser()`` on every
    whitespace-separated token. A token like ``~25`` (common in natural-language
    approximations such as ``~25 ms``) raises ``RuntimeError`` and aborts the
    whole turn. Only ``~/`` is a real home reference in our prompts, so we
    rewrite every other tilde to the math operator ``∼`` (U+223C), which the
    LLM still reads as "approximately".
    """
    return _TILDE_NEEDING_GUARD.sub("∼", text)


def build_msagent_agent_prompt(
    case: BenchmarkCase,
    input_path: Path,
    *,
    visible_input_path: Path | str | None,
    visible_skill_path: Path | str | None,
    schema_text: str,
) -> str:
    base_prompt = build_agent_prompt(
        case,
        input_path,
        visible_input_path=visible_input_path,
        visible_skill_path=visible_skill_path,
    )
    return sanitize_msagent_prompt(f"""{base_prompt}

Output contract for msagent one-shot mode:
- Your final response must be a single JSON object.
- Do not wrap the JSON in Markdown fences.
- Do not include any text before or after the JSON object.
- The JSON object must match this schema:
{schema_text}
""")


def build_msagent_judge_prompt(
    case: BenchmarkCase,
    trace: dict[str, Any],
    *,
    schema_text: str,
) -> str:
    base_prompt = build_judge_prompt(case, trace)
    return sanitize_msagent_prompt(f"""{base_prompt}

Output contract for msagent one-shot mode:
- Your final response must be a single JSON object.
- Do not wrap the JSON in Markdown fences.
- Do not include any text before or after the JSON object.
- The JSON object must match this schema:
{schema_text}
""")


def read_msagent_structured_final(
    stdout: str,
    *,
    required_keys: set[str],
    msagent_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fallback: dict[str, Any] | None = None
    if msagent_events:
        for event in reversed(msagent_events):
            if event.get("type") != "assistant_message":
                continue
            content = event.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            candidates = extract_json_objects(content)
            for candidate in reversed(candidates):
                if required_keys <= set(candidate):
                    return candidate
            if candidates and fallback is None:
                fallback = candidates[-1]

    stdout_candidates = extract_json_objects(strip_ansi(stdout))
    for candidate in reversed(stdout_candidates):
        if required_keys <= set(candidate):
            return candidate
    if stdout_candidates:
        return stdout_candidates[-1]
    if fallback is not None:
        return fallback
    raise RuntimeError(f"Could not parse structured msagent final output: {stdout[-1000:]}")


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)


def normalize_msagent_stdout(stdout: str) -> list[dict[str, Any]]:
    text = strip_ansi(stdout).strip()
    if not text:
        return []

    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        tool_call = parse_msagent_tool_call_line(line)
        if tool_call is not None:
            events.append(tool_call)

    events.append(
        {
            "event_type": "agent_event",
            "raw_type": "msagent.stdout",
            "summary": trim_text(text, 4000),
        }
    )
    return events


def read_msagent_trace_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def normalize_msagent_trace_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type == "tool_call":
            normalized.append(
                {
                    "event_type": "tool_call",
                    "raw_type": "msagent.trace.tool_call",
                    "tool": str(event.get("tool") or "unknown"),
                    "item_id": event.get("item_id"),
                    "input": event.get("input", {}),
                    "summary": trim_event_text(event),
                }
            )
        elif event_type == "tool_result":
            payload = {
                "event_type": "tool_result",
                "raw_type": "msagent.trace.tool_result",
                "tool": str(event.get("tool") or "tool"),
                "item_id": event.get("item_id"),
                "output": event.get("output", {}),
                "summary": trim_event_text(event),
            }
            duration_ms = event.get("duration_ms")
            if isinstance(duration_ms, (int, float)):
                payload["duration_ms"] = round(float(duration_ms))
            normalized.append(payload)
        elif event_type == "token_usage":
            normalized.append(
                {
                    "event_type": "token_usage",
                    "raw_type": "msagent.trace.token_usage",
                    "usage": event.get("usage", {}),
                    "cumulative": event.get("cumulative", {}),
                    "summary": trim_event_text(event),
                }
            )
        elif event_type == "session_finished":
            normalized.append(
                {
                    "event_type": "agent_event",
                    "raw_type": "msagent.trace.session_finished",
                    "exit_code": event.get("exit_code"),
                    "duration_ms": event.get("duration_ms"),
                    "token_usage": event.get("token_usage", {}),
                    "summary": trim_event_text(event),
                }
            )
        elif event_type in {"session_started", "assistant_message", "error"}:
            normalized.append(
                {
                    "event_type": "agent_event",
                    "raw_type": f"msagent.trace.{event_type}",
                    "summary": trim_event_text(event),
                }
            )
    return normalized


def parse_msagent_tool_call_line(line: str) -> dict[str, Any] | None:
    cleaned = line.strip()
    match = re.search(r"Use tool\s+([A-Za-z0-9_.:-]+)", cleaned)
    if not match:
        return None
    return {
        "event_type": "tool_call",
        "raw_type": "msagent.rendered_tool_call",
        "tool": match.group(1),
        "input": {},
        "summary": trim_event_text({"line": cleaned}),
    }


def msagent_session_duration_ms(events: list[dict[str, Any]]) -> int | None:
    for event in reversed(events):
        if event.get("type") != "session_finished":
            continue
        duration_ms = event.get("duration_ms")
        if isinstance(duration_ms, (int, float)):
            return round(float(duration_ms))
    return None


def msagent_token_usage(events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if events:
        for event in reversed(events):
            if event.get("type") != "session_finished":
                continue
            usage = normalize_msagent_token_usage(event.get("token_usage"))
            if usage is not None:
                return usage

        for event in reversed(events):
            if event.get("type") != "token_usage":
                continue
            usage = normalize_msagent_token_usage(event.get("cumulative"))
            if usage is not None:
                return usage

    return {
        "available": False,
        "source": "msagent-cli-no-usage-event",
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def normalize_msagent_token_usage(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    input_tokens = int(value.get("input_tokens") or 0)
    output_tokens = int(value.get("output_tokens") or 0)
    total_tokens = int(value.get("total_tokens") or input_tokens + output_tokens)
    available = bool(value.get("available", total_tokens > 0))
    source = str(value.get("source") or "msagent-cli-jsonl")
    return {
        "available": available,
        "source": source,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
