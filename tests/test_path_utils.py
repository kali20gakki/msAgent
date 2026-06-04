#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pytest

from msagent.utils.path import (
    SymlinkEscapeError,
    expand_pattern,
    is_absolute_path_like,
    is_path_within,
    is_symlink_escape,
    is_windows_absolute_path,
    matches_hidden,
    pattern_to_regex,
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


def test_resolve_path_handles_root_home_and_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello", encoding="utf-8")

    assert resolve_path(str(tmp_path), "/") == tmp_path.resolve()
    assert resolve_path(str(tmp_path), str(sample)) == sample.resolve()
    assert resolve_path(str(tmp_path), "~").is_absolute()

    monkeypatch.setattr("msagent.utils.path.is_symlink_escape", lambda *_args, **_kwargs: True)
    with pytest.raises(SymlinkEscapeError):
        resolve_path(str(tmp_path), "sample.txt")


def test_is_path_within_and_is_symlink_escape_cover_existing_missing_and_failures(
    tmp_path: Path,
) -> None:
    inside = tmp_path / "inside"
    inside.mkdir()
    nested = inside / "child.txt"
    nested.write_text("ok", encoding="utf-8")

    assert is_path_within(nested, [tmp_path]) is True
    assert is_path_within(tmp_path / "missing.txt", [tmp_path]) is True

    class _FakePath:
        def __init__(self, *, is_symlink: bool, resolve_result: Path | None = None, error: Exception | None = None):
            self._is_symlink = is_symlink
            self._resolve_result = resolve_result
            self._error = error

        def is_symlink(self) -> bool:
            return self._is_symlink

        def resolve(self) -> Path:
            if self._error is not None:
                raise self._error
            assert self._resolve_result is not None
            return self._resolve_result

    assert is_symlink_escape(_FakePath(is_symlink=False), [tmp_path]) is False
    assert is_symlink_escape(_FakePath(is_symlink=True, error=OSError("boom")), [tmp_path]) is True
    assert is_symlink_escape(_FakePath(is_symlink=True, resolve_result=nested), [tmp_path]) is False
    assert is_symlink_escape(_FakePath(is_symlink=True, resolve_result=Path("C:/outside")), [tmp_path]) is True


def test_expand_pattern_supports_globs_and_nonexistent_literals(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    alpha = tmp_path / "src" / "alpha.py"
    beta = tmp_path / "src" / "beta.py"
    alpha.write_text("", encoding="utf-8")
    beta.write_text("", encoding="utf-8")

    globbed = expand_pattern("src/*.py", tmp_path)
    assert set(path.name for path in globbed) == {"alpha.py", "beta.py"}
    assert expand_pattern("missing.txt", tmp_path) == []
    assert expand_pattern("missing.txt", tmp_path, include_nonexistent=True) == [tmp_path / "missing.txt"]


def test_pattern_to_regex_and_matches_hidden_cover_literal_glob_and_posix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hidden_dir = tmp_path / ".secret"
    hidden_dir.mkdir()
    hidden_file = hidden_dir / "token.txt"
    hidden_file.write_text("x", encoding="utf-8")

    regex = pattern_to_regex("**/*.txt")
    assert regex is not None and "\\.txt" in regex
    posix_regex = pattern_to_regex("**/*.txt", posix=True)
    assert posix_regex is not None and "(?:" not in posix_regex

    class _FakeRegex:
        def __init__(self, pattern: str) -> None:
            self.pattern = pattern

    class _FakeGitIgnoreSpecPattern:
        def __init__(self, _pattern: str) -> None:
            self.regex = _FakeRegex("^(?:tmp)")

    class _FakePath:
        def __init__(self, value: str) -> None:
            self.value = value

        def expanduser(self) -> "_FakePath":
            return self

        def __str__(self) -> str:
            return self.value

    with monkeypatch.context() as ctx:
        ctx.setattr("msagent.utils.path.Path", _FakePath)
        ctx.setattr("msagent.utils.path.GitIgnoreSpecPattern", _FakeGitIgnoreSpecPattern)
        assert pattern_to_regex("/tmp/*.txt") == "^/(?:tmp)"

    class _FakeNoRegexGitIgnoreSpecPattern:
        def __init__(self, _pattern: str) -> None:
            self.regex = None

    with monkeypatch.context() as ctx:
        ctx.setattr("msagent.utils.path.GitIgnoreSpecPattern", _FakeNoRegexGitIgnoreSpecPattern)
        assert pattern_to_regex("README.md") is None

    assert matches_hidden(hidden_file, ["**/.secret/*"], tmp_path) is True
    assert matches_hidden(hidden_file, [".secret"], tmp_path) is True
    assert matches_hidden(tmp_path / "visible.txt", [".secret"], tmp_path) is False
