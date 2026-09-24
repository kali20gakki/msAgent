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

"""Validation and initialization for the global msAgent storage layout."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from msagent.core.paths import AppPaths

STORAGE_LAYOUT_VERSION = 2

LEGACY_HOME_ENTRIES = (
    "config.llms.yml",
    "config.mcp.json",
    "config.approval.json",
    "config.agents.yml",
    "config.subagents.yml",
    "config.checkpointers.yml",
    "config.checkpoints.db",
    "langgraph.json",
    ".history",
    "memory.md",
    "agents/",
    "subagents/",
    "llms/",
    "checkpointers/",
)

_MANAGED_DIRECTORIES = (
    "config",
    "prompts",
    "skills",
    "state",
    "state/projects",
    "cache",
    "cache/mcp",
    "oauth",
    "oauth/mcp",
    "logs",
)


class StorageLayoutError(RuntimeError):
    """Raised when the global storage directory cannot be used safely."""


def validate_and_initialize_storage_layout(paths: AppPaths) -> None:
    """Validate the configured home and initialize storage layout version 2."""
    _inspect_storage_layout(paths)
    # Inspect again immediately before the first write to narrow the race with
    # another process or an external tool changing the directory.
    metadata_version = _inspect_storage_layout(paths)
    _ensure_layout_directories(paths)
    if metadata_version is not None:
        return

    if _path_present(paths.metadata_file):
        _read_metadata_version(paths.metadata_file, {paths.metadata_file.name})
        return
    _write_metadata_atomically(paths.metadata_file)


def _inspect_storage_layout(paths: AppPaths) -> int | None:
    entries = _read_home_entries(paths.home)
    metadata_version = _read_metadata_version(paths.metadata_file, entries)
    legacy_entries = _find_legacy_entries(entries)
    new_entries = _find_new_entries(paths, entries, metadata_version)
    if legacy_entries:
        if new_entries:
            raise StorageLayoutError(_mixed_layout_message(paths.home, legacy_entries))
        raise StorageLayoutError(_legacy_layout_message(paths.home))

    _validate_managed_directories(paths.home)
    return metadata_version


def _ensure_layout_directories(paths: AppPaths) -> None:
    try:
        paths.ensure()
    except OSError as exc:
        raise StorageLayoutError(f"Cannot initialize msAgent storage at {paths.home}: {exc}") from exc


def _read_home_entries(home: Path) -> set[str]:
    if not _path_present(home):
        return set()
    if home.is_symlink() or not home.is_dir():
        raise StorageLayoutError(f"msAgent home must be a directory: {home}")
    try:
        return {entry.name for entry in home.iterdir()}
    except OSError as exc:
        raise StorageLayoutError(f"Cannot inspect msAgent home {home}: {exc}") from exc


def _read_metadata_version(metadata_file: Path, entries: set[str]) -> int | None:
    if metadata_file.name not in entries:
        return None
    if metadata_file.is_symlink() or not metadata_file.is_file():
        raise StorageLayoutError(f"msAgent storage metadata must be a regular file: {metadata_file}")
    try:
        payload = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StorageLayoutError(
            f"Cannot read valid msAgent storage metadata from {metadata_file}: {exc}. "
            "Back up the file and repair or remove the storage directory."
        ) from exc
    if not isinstance(payload, dict):
        raise StorageLayoutError(f"msAgent storage metadata must contain a JSON object: {metadata_file}")

    version = payload.get("storage_layout_version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise StorageLayoutError(f"Invalid storage_layout_version in {metadata_file}; expected a positive integer.")
    if version < STORAGE_LAYOUT_VERSION:
        raise StorageLayoutError(
            f"Storage layout version {version} at {metadata_file.parent} is no longer supported. "
            "Back up and move the directory, then run msAgent again."
        )
    if version > STORAGE_LAYOUT_VERSION:
        raise StorageLayoutError(
            f"Storage layout version {version} at {metadata_file.parent} is newer than this msAgent "
            "version supports. Upgrade msAgent before using this directory."
        )
    return version


def _find_legacy_entries(entries: set[str]) -> tuple[str, ...]:
    return tuple(label for label in LEGACY_HOME_ENTRIES if label.rstrip("/") in entries)


def _find_new_entries(paths: AppPaths, entries: set[str], metadata_version: int | None) -> tuple[str, ...]:
    found: list[str] = []
    if metadata_version == STORAGE_LAYOUT_VERSION:
        found.append("metadata.json")
    if "config" in entries and paths.config_dir.is_dir():
        found.append("config/")
    if paths.projects_dir.is_dir():
        found.append("state/projects/")
    return tuple(found)


def _validate_managed_directories(home: Path) -> None:
    for relative_path in _MANAGED_DIRECTORIES:
        path = home / relative_path
        if not _path_present(path):
            continue
        if path.is_symlink() or not path.is_dir():
            raise StorageLayoutError(f"msAgent storage path must be a regular directory: {path}")


def _write_metadata_atomically(metadata_file: Path) -> None:
    temp_file = metadata_file.with_name(f".{metadata_file.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = {"storage_layout_version": STORAGE_LAYOUT_VERSION}
    try:
        descriptor = os.open(temp_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temp_file.replace(metadata_file)
    except OSError as exc:
        raise StorageLayoutError(f"Cannot write msAgent storage metadata {metadata_file}: {exc}") from exc
    finally:
        try:
            temp_file.unlink(missing_ok=True)
        except OSError:
            pass


def _path_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _legacy_layout_message(home: Path) -> str:
    return (
        "msAgent 启动已中止：检测到不兼容的旧版配置目录。\n\n"
        "您已升级 msAgent，但以下目录仍为旧版本结构：\n\n"
        f"  {home}\n\n"
        "新版改为统一使用全局目录 ~/.msagent，配置只需设置一次，即可在多个项目间共享；"
        "同时各工作目录的运行状态独立保存，互不干扰。\n\n"
        "由于新旧目录结构不兼容，无法直接复用，请手动删除该目录后重新启动 msAgent。"
        "启动时会自动创建新版目录结构。\n\n"
        "⚠️  删除前请注意：\n"
        f"  • 将清除：{home} 下的所有配置、状态、历史及检查点\n"
        f"  • 如需保留数据，请提前备份\n"
        f"  • {home}/skills 可能包含您自己添加的 Skill，请单独备份\n"
        f"  • 新版内置 Skill 由安装包提供，无需备份或恢复\n\n"
        "本次启动未对该目录做任何更改。"
    )


def _mixed_layout_message(
    home: Path,
    legacy_entries: tuple[str, ...],
) -> str:
    legacy_summary = ", ".join(legacy_entries)
    skills_note = _skills_backup_note(home)

    return (
        "msAgent 启动已中止：检测到新旧版本配置混用。\n\n"
        "您已升级 msAgent，但以下目录中同时存在新版结构和旧版残留：\n\n"
        f"  {home}\n\n"
        "检测到的旧版残留：\n\n"
        f"  {legacy_summary}\n\n"
        "新版改为统一使用全局目录 ~/.msagent，配置只需设置一次，即可在多个项目间共享；"
        "同时各工作目录的运行状态独立保存，互不干扰。\n\n"
        "由于新旧目录结构不兼容，无法安全混用，请手动删除该目录后重新启动 msAgent。"
        "启动时会自动创建新版目录结构。\n\n"
        "⚠️  删除前请注意：\n"
        f"  • 会清除：新版配置、项目状态、对话历史、检查点\n"
        f"  • 如需保留数据，请提前备份\n"
        f"{skills_note}\n"
        "本次启动未对该目录做任何更改。"
    )


def _skills_backup_note(home: Path) -> str:
    return (
        f"  • {home}/skills 可能包含您自己添加的 Skill，请单独备份\n  • 新版内置 Skill 由安装包提供，无需备份或恢复\n"
    )
