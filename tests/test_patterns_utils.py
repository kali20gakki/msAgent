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

from msagent.utils.patterns import (
    matches_patterns,
    mcp_server_matcher,
    three_part_matcher,
    two_part_matcher,
)


def test_matches_patterns_requires_positive_match_and_rejects_negatives() -> None:
    def matcher(pattern: str) -> bool:
        return pattern in {"cli:*", "cli:tools", "mcp:server:*"}

    assert matches_patterns(["cli:*"], matcher) is True
    assert matches_patterns(["cli:*", "!cli:tools"], matcher) is False
    assert matches_patterns(["!cli:*"], matcher) is False


def test_two_part_matcher_supports_globs_and_reports_invalid_patterns() -> None:
    invalid: list[str] = []
    matcher = two_part_matcher(name="run_command", module="cli.handlers", on_invalid=invalid.append)

    assert matcher("cli.*:run_*") is True
    assert matcher("invalid-pattern") is False
    assert invalid == ["invalid-pattern"]


def test_three_part_matcher_supports_globs_and_reports_invalid_patterns() -> None:
    invalid: list[str] = []
    matcher = three_part_matcher(
        name="run_command",
        module="cli.handlers",
        category="tool",
        on_invalid=invalid.append,
    )

    assert matcher("tool:cli.*:run_*") is True
    assert matcher("tool:cli.handlers") is False
    assert invalid == ["tool:cli.handlers"]


def test_mcp_server_matcher_validates_category_and_wildcard_tool_name() -> None:
    invalid: list[str] = []
    matcher = mcp_server_matcher("profiler", "mcp", on_invalid=invalid.append)

    assert matcher("mcp") is False
    assert matcher("tool:profiler:*") is False
    assert matcher("mcp:profiler:run") is False
    assert matcher("mcp:prof*: *".replace(" ", "")) is True
    assert invalid == ["mcp:profiler:run"]
