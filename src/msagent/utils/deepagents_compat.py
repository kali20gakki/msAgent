"""Compatibility helpers for deepagents integration."""

from __future__ import annotations

import ntpath
from collections.abc import Sequence

from deepagents.backends import utils as _backend_utils

from msagent.utils.path import is_windows_absolute_path

_original_validate_path = _backend_utils.validate_path


def validate_deepagents_path(
    path: str,
    *,
    allowed_prefixes: Sequence[str] | None = None,
) -> str:
    """Accept native Windows absolute paths while preserving deepagents rules."""
    if is_windows_absolute_path(path) and allowed_prefixes is None:
        return ntpath.normpath(path)

    return _original_validate_path(path, allowed_prefixes=allowed_prefixes)


def patch_deepagents_windows_absolute_paths() -> None:
    """Patch deepagents filesystem path validation for Windows hosts."""
    from deepagents.backends import utils as backend_utils
    from deepagents.middleware import filesystem as filesystem_middleware

    if backend_utils.validate_path is not validate_deepagents_path:
        backend_utils.validate_path = validate_deepagents_path
    if filesystem_middleware.validate_path is not validate_deepagents_path:
        filesystem_middleware.validate_path = validate_deepagents_path
