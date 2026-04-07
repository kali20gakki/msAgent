from __future__ import annotations

from deepagents.backends import utils as backend_utils
from deepagents.middleware import filesystem as filesystem_middleware

from msagent.utils.deepagents_compat import (
    patch_deepagents_windows_absolute_paths,
    validate_deepagents_path,
)


def test_validate_deepagents_path_accepts_windows_absolute_paths() -> None:
    path = r"C:\ProfileData\trace\result.json"

    assert validate_deepagents_path(path) == path


def test_patch_deepagents_windows_absolute_paths_updates_deepagents_modules() -> None:
    patch_deepagents_windows_absolute_paths()

    path = r"C:\ProfileData\trace\result.json"

    assert backend_utils.validate_path(path) == path
    assert filesystem_middleware.validate_path(path) == path
