from __future__ import annotations

from pathlib import Path

from msagent.utils.path import (
    is_absolute_path_like,
    is_windows_absolute_path,
    resolve_path,
)


def test_is_windows_absolute_path_detects_drive_and_unc() -> None:
    assert is_windows_absolute_path(r"C:\Users\alice\data.txt") is True
    assert is_windows_absolute_path("D:/work/report.md") is True
    assert is_windows_absolute_path(r"\\server\share\data.csv") is True

    assert is_windows_absolute_path("relative/path.txt") is False
    assert is_windows_absolute_path("/tmp/file.txt") is False


def test_is_absolute_path_like_supports_windows_style() -> None:
    assert is_absolute_path_like(r"C:\tmp\foo.txt") is True
    assert is_absolute_path_like("E:/tmp/foo.txt") is True
    assert is_absolute_path_like("/tmp/foo.txt") is True
    assert is_absolute_path_like("foo/bar.txt") is False


def test_resolve_path_keeps_windows_absolute_path_outside_working_dir(
    tmp_path: Path,
) -> None:
    windows_path = r"C:\Users\alice\project\main.py"
    resolved = resolve_path(str(tmp_path), windows_path)
    normalized = str(resolved).replace("/", "\\")

    assert normalized.lower().startswith(r"c:\users\alice\project\main.py")
    assert str(tmp_path).lower() not in normalized.lower()
