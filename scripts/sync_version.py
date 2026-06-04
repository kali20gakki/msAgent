#!/usr/bin/env python3
"""Sync package version across pyproject.toml and uv.lock."""

from __future__ import annotations

import configparser
import os
import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version


PACKAGE_NAME = "mindstudio-agent"


def _die(message: str) -> None:
    raise SystemExit(message)


def _validate(version: str) -> str:
    try:
        Version(version)
    except InvalidVersion as exc:
        raise SystemExit(
            f"Invalid package version {version!r}. "
            "Expected a valid Python package version such as '26.0.0' or '26.0.0rc1'."
        ) from exc
    return version


def _resolve_version(repo_root: Path) -> str:
    if version := os.getenv("WHL_VERSION", "").strip():
        return _validate(version)

    parser = configparser.ConfigParser()
    version_info_path = repo_root / "version.info"
    parser.read(version_info_path, encoding="utf-8")

    try:
        version = parser["PACKAGE"]["Version"].strip()
    except KeyError as exc:
        raise SystemExit(f"Failed to parse {version_info_path}") from exc

    if not version:
        _die("Resolved version is empty")
    return _validate(version)


def _replace_once(path: Path, pattern: str, replacement: str) -> None:
    content = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
    if count != 1:
        _die(f"Could not update version in {path}")
    if updated != content:
        path.write_text(updated, encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    version = _resolve_version(repo_root)

    _replace_once(
        repo_root / "pyproject.toml",
        r'^(version\s*=\s*")([^"]+)(")$',
        rf"\g<1>{version}\g<3>",
    )
    _replace_once(
        repo_root / "uv.lock",
        rf'^(\[\[package\]\]\nname = "{re.escape(PACKAGE_NAME)}"\nversion = ")([^"]+)(")$',
        rf"\g<1>{version}\g<3>",
    )

    print(f"Synced package version to {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
