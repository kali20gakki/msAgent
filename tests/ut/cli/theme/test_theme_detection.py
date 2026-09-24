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

from __future__ import annotations

from types import SimpleNamespace

import pytest

import msagent.cli.theme.detect as detect_module


class _InputStream:
    def __init__(self, response: str, *, is_tty: bool = True) -> None:
        self.response = iter(response)
        self.is_tty = is_tty

    def isatty(self) -> bool:
        return self.is_tty

    def fileno(self) -> int:
        return 17

    def read(self, size: int) -> str:
        assert size == 1
        return next(self.response)


class _OutputStream:
    def __init__(self, *, is_tty: bool = True) -> None:
        self.is_tty = is_tty
        self.writes: list[str] = []
        self.flush_count = 0

    def isatty(self) -> bool:
        return self.is_tty

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        self.flush_count += 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0;7", "light"), ("0;15", "light"), ("15;0", "dark")],
)
def test_detect_via_colorfgbg_returns_theme_when_background_index_is_valid(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
    expected: str,
) -> None:
    monkeypatch.setenv("COLORFGBG", value)

    assert detect_module._detect_via_colorfgbg() == expected


@pytest.mark.parametrize("value", ["", "7", "0;white"])
def test_detect_via_colorfgbg_returns_none_when_value_is_missing_or_invalid(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("COLORFGBG", value)

    assert detect_module._detect_via_colorfgbg() is None


def test_detect_terminal_theme_uses_normalized_override_when_override_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSAGENT_BACKGROUND_THEME", " Light ")
    monkeypatch.setattr(
        detect_module,
        "_detect_via_osc11",
        lambda: (_ for _ in ()).throw(AssertionError("fallback must not run")),
    )

    assert detect_module.detect_terminal_theme() == "light"


def test_detect_terminal_theme_uses_osc11_when_override_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MSAGENT_BACKGROUND_THEME", "automatic")
    monkeypatch.setattr(detect_module, "_detect_via_osc11", lambda: "light")
    monkeypatch.setattr(
        detect_module,
        "_detect_via_colorfgbg",
        lambda: (_ for _ in ()).throw(AssertionError("later fallback must not run")),
    )

    assert detect_module.detect_terminal_theme() == "light"


def test_detect_terminal_theme_uses_colorfgbg_when_osc11_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MSAGENT_BACKGROUND_THEME", raising=False)
    monkeypatch.setattr(detect_module, "_detect_via_osc11", lambda: None)
    monkeypatch.setattr(detect_module, "_detect_via_colorfgbg", lambda: "light")

    assert detect_module.detect_terminal_theme() == "light"


def test_detect_terminal_theme_defaults_to_dark_when_all_sources_are_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MSAGENT_BACKGROUND_THEME", raising=False)
    monkeypatch.setattr(detect_module, "_detect_via_osc11", lambda: None)
    monkeypatch.setattr(detect_module, "_detect_via_colorfgbg", lambda: None)

    assert detect_module.detect_terminal_theme() == "dark"


def test_detect_via_osc11_returns_none_when_stream_is_not_tty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(detect_module.sys, "stdin", _InputStream("", is_tty=False))
    monkeypatch.setattr(detect_module.sys, "stdout", _OutputStream())

    assert detect_module._detect_via_osc11() is None


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("\x1b]11;rgb:ffff/ffff/ffff\\", "light"),
        ("\x1b]11;rgb:0000/0000/0000\a", "dark"),
    ],
)
def test_detect_via_osc11_returns_luminance_theme_when_terminal_responds(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
    expected: str,
) -> None:
    input_stream = _InputStream(response)
    output_stream = _OutputStream()
    restore_calls: list[tuple[int, int, object]] = []
    termios = SimpleNamespace(
        error=OSError,
        TCSADRAIN=1,
        tcgetattr=lambda fd: ["settings", fd],
        tcsetattr=lambda fd, when, settings: restore_calls.append((fd, when, settings)),
    )
    raw_calls: list[int] = []
    tty = SimpleNamespace(setraw=lambda fd: raw_calls.append(fd))

    monkeypatch.setitem(__import__("sys").modules, "termios", termios)
    monkeypatch.setitem(__import__("sys").modules, "tty", tty)
    monkeypatch.setattr(detect_module.sys, "stdin", input_stream)
    monkeypatch.setattr(detect_module.sys, "stdout", output_stream)
    monkeypatch.setattr(
        detect_module.select,
        "select",
        lambda readers, _writers, _errors, _timeout: (
            readers if response else [],
            [],
            [],
        ),
    )

    result = detect_module._detect_via_osc11()

    assert result == expected
    assert output_stream.writes == ["\x1b]11;?\x1b\\"]
    assert output_stream.flush_count == 1
    assert raw_calls == [17]
    assert restore_calls == [(17, 1, ["settings", 17])]
