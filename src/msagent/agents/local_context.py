"""Local environment context collection for prompt injection."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

_CACHE_TTL_SECONDS = 30.0
_MAX_SCRIPT_COMMANDS = 8
_MAX_TOP_LEVEL_ENTRIES = 24
_MAX_CONTEXT_CHARS = 4000

_CONTEXT_CACHE: dict[str, tuple[float, str]] = {}
LOCAL_CONTEXT_PROMPT_PLACEHOLDER = """Use this local workspace snapshot to ground your answers in the current environment:
<local-context>
{local_environment_context}
</local-context>"""


def build_local_environment_context(
    working_dir: Path,
    *,
    now: datetime | None = None,
) -> str:
    """Build a concise markdown snapshot of local runtime/project environment."""
    resolved_dir = working_dir.resolve()
    cache_key = str(resolved_dir)
    timestamp = time.monotonic()
    cached = _CONTEXT_CACHE.get(cache_key)
    if cached and timestamp - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    dt = now or datetime.now().astimezone()
    lines: list[str] = []
    lines.extend(_build_runtime_section(resolved_dir, dt))
    lines.extend(_build_project_section(resolved_dir))
    lines.extend(_build_git_section(resolved_dir))
    lines.extend(_build_top_level_section(resolved_dir))

    context = "\n".join(lines).strip()
    if len(context) > _MAX_CONTEXT_CHARS:
        context = f"{context[: _MAX_CONTEXT_CHARS - 24].rstrip()}\n... (truncated)"

    _CONTEXT_CACHE[cache_key] = (timestamp, context)
    return context


def ensure_local_context_prompt(prompt: str) -> str:
    """Ensure prompt contains the local-environment placeholder block."""
    if "{local_environment_context}" in prompt:
        return prompt
    return f"{prompt.rstrip()}\n\n{LOCAL_CONTEXT_PROMPT_PLACEHOLDER}".strip()


def _build_runtime_section(working_dir: Path, now: datetime) -> list[str]:
    shell = os.environ.get("SHELL") or os.environ.get("COMSPEC") or "unknown"
    tool_candidates = [
        "git",
        "python",
        "pytest",
        "node",
        "npm",
        "pnpm",
        "yarn",
        "uv",
        "go",
        "cargo",
        "make",
    ]
    available_tools = [name for name in tool_candidates if shutil.which(name)]
    tools_text = ", ".join(available_tools) if available_tools else "none detected"
    return [
        "## Local Runtime Snapshot",
        f"- Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Working directory: `{working_dir}`",
        f"- OS: `{platform.system()} {platform.release()}`",
        f"- Python: `{platform.python_version()}`",
        f"- Shell: `{shell}`",
        f"- Available dev tools: {tools_text}",
        "",
    ]


def _build_project_section(working_dir: Path) -> list[str]:
    try:
        file_names = {path.name.lower() for path in working_dir.iterdir()}
    except Exception:
        return [
            "## Project Signals",
            "- Unable to inspect project files in current working directory",
            "",
        ]
    languages: list[str] = []
    package_managers: list[str] = []
    suggestions: list[str] = []

    if {"pyproject.toml", "requirements.txt", "requirements-dev.txt"} & file_names:
        languages.append("Python")
        suggestions.extend(["pytest", "python -m pytest"])
    if "uv.lock" in file_names:
        package_managers.append("uv")
        suggestions.extend(["uv run pytest", "uv run python -m pytest"])
    if "package.json" in file_names:
        languages.append("JavaScript/TypeScript")
        package_managers.append("npm")
        suggestions.extend(_load_npm_script_commands(working_dir / "package.json"))
    if "pnpm-lock.yaml" in file_names:
        package_managers.append("pnpm")
    if "yarn.lock" in file_names:
        package_managers.append("yarn")
    if {"go.mod", "go.sum"} & file_names:
        languages.append("Go")
        package_managers.append("go modules")
        suggestions.append("go test ./...")
    if {"cargo.toml", "cargo.lock"} & file_names:
        languages.append("Rust")
        package_managers.append("cargo")
        suggestions.append("cargo test")
    if {"pom.xml", "build.gradle", "build.gradle.kts"} & file_names:
        languages.append("Java/Kotlin")
        suggestions.append("./gradlew test")
    if "makefile" in file_names:
        suggestions.append("make test")

    language_text = ", ".join(dict.fromkeys(languages)) if languages else "unknown"
    pm_text = (
        ", ".join(dict.fromkeys(package_managers))
        if package_managers
        else "none detected"
    )
    unique_suggestions = list(dict.fromkeys(suggestions))
    section = [
        "## Project Signals",
        f"- Detected languages: {language_text}",
        f"- Detected package/build managers: {pm_text}",
    ]
    if unique_suggestions:
        section.append("- Candidate local test/validation commands:")
        for cmd in unique_suggestions[:_MAX_SCRIPT_COMMANDS]:
            section.append(f"  - `{cmd}`")
    section.append("")
    return section


def _build_git_section(working_dir: Path) -> list[str]:
    if not shutil.which("git"):
        return ["## Git Snapshot", "- `git` command not available", ""]

    inside_repo = _run_git(working_dir, "rev-parse", "--is-inside-work-tree")
    if inside_repo != "true":
        return ["## Git Snapshot", "- Not a git repository", ""]

    branch = _run_git(working_dir, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = _run_git(working_dir, "status", "--short")
    changed_files = len([line for line in status.splitlines() if line.strip()]) if status else 0

    return [
        "## Git Snapshot",
        f"- Branch: `{branch}`",
        f"- Working tree changed files: {changed_files}",
        "",
    ]


def _build_top_level_section(working_dir: Path) -> list[str]:
    entries: list[str] = []
    skip_names = {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
    }

    try:
        candidates = sorted(
            working_dir.iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
    except Exception:
        return [
            "## Top-Level Workspace Layout",
            "- Unable to enumerate workspace entries",
            "",
        ]

    for path in candidates:
        if path.name in skip_names:
            continue
        prefix = "[D]" if path.is_dir() else "[F]"
        entries.append(f"- {prefix} `{path.name}`")
        if len(entries) >= _MAX_TOP_LEVEL_ENTRIES:
            break

    if not entries:
        entries.append("- (empty)")

    return ["## Top-Level Workspace Layout", *entries, ""]


def _load_npm_script_commands(package_json_path: Path) -> list[str]:
    try:
        payload = json.loads(package_json_path.read_text(encoding="utf-8"))
    except Exception:
        return ["npm test"]

    scripts = payload.get("scripts")
    if not isinstance(scripts, dict):
        return ["npm test"]

    return [f"npm run {name}" for name in scripts.keys()][: _MAX_SCRIPT_COMMANDS]


def _run_git(working_dir: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(working_dir), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None
    return completed.stdout.strip()
