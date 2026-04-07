"""Compatibility helpers for deepagents memory file handling."""

from __future__ import annotations

from pathlib import Path

from msagent.core.constants import CONFIG_MEMORY_FILE_NAME


def get_memory_file_path(working_dir: Path | None = None) -> Path:
    """Return the canonical memory file path used by deepagents memory middleware."""
    if working_dir is None:
        working_dir = Path.cwd()

    return working_dir / CONFIG_MEMORY_FILE_NAME


def ensure_memory_file(working_dir: Path | None = None) -> Path:
    """Ensure the memory file exists and return its path."""
    memory_path = get_memory_file_path(working_dir)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    if not memory_path.exists():
        memory_path.write_text("", encoding="utf-8")
    return memory_path


def read_memory_file(working_dir: Path | None = None) -> str:
    """Read user memory content from the canonical memory file."""
    memory_path = get_memory_file_path(working_dir)

    if not memory_path.exists():
        return ""

    try:
        return memory_path.read_text(encoding="utf-8")
    except Exception:
        return ""
