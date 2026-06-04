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

from msagent.utils.matching import (
    find_fuzzy_match,
    find_progressive_match,
    format_match_error,
    normalize_whitespace,
)


def test_normalize_whitespace_converts_line_endings_and_dedents() -> None:
    text = "    alpha  \r\n      beta\t \r\n"

    assert normalize_whitespace(text) == "alpha\n  beta\n"


def test_find_fuzzy_match_returns_best_window_and_line_number() -> None:
    content = "alpha\nbeta value\ngamma\n"
    result = find_fuzzy_match(content, "beta value\ngamma", threshold=0.5)

    assert result is not None
    assert result[0] == "beta value\ngamma"
    assert result[2] == 2


def test_find_fuzzy_match_returns_none_when_threshold_not_met() -> None:
    assert find_fuzzy_match("alpha\nbeta", "totally different", threshold=0.99) is None


def test_find_progressive_match_handles_exact_and_normalized_matches() -> None:
    content = "start\n  alpha\n    beta\nend"

    assert find_progressive_match(content, "alpha\n    beta") == (True, 8, 22)
    assert find_progressive_match(content, "missing text") == (False, -1, -1)


def test_format_match_error_reports_closest_match_with_preview() -> None:
    message = format_match_error(
        file_path="sample.py",
        edit_num=3,
        search_content="alpha\nbeta\ngamma\ndelta",
        file_content="zero\nalpha\nbeta\ngimma\ndelta\nend",
        preview_len=8,
    )

    assert "Old content not found in file: sample.py" in message
    assert "Edit #3 failed." in message
    assert "Closest match" in message
    assert "Expected:\nalpha\nbe..." in message
    assert "Found:\nalpha\nbe..." in message


def test_format_match_error_reports_no_similar_content_when_fuzzy_lookup_fails(monkeypatch) -> None:
    monkeypatch.setattr("msagent.utils.matching.find_fuzzy_match", lambda *_args, **_kwargs: None)

    message = format_match_error(
        file_path="sample.py",
        edit_num=1,
        search_content="needle",
        file_content="haystack",
    )

    assert "No similar content found." in message
    assert "Searching for:\nneedle" in message
