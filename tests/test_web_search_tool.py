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

from unittest.mock import AsyncMock, Mock
import pytest
from langchain_core.tools import ToolException

import msagent.tools.web_search as web_search_module
from msagent.tools.web_search import (
    WebSearchInput,
    WebSearchNoResultsError,
    _run_duckduckgo_search,
    web_search,
)


def test_web_search_input_normalizes_query() -> None:
    payload = WebSearchInput(query="  langchain deepagents  ")
    assert payload.query == "langchain deepagents"


@pytest.mark.asyncio
async def test_run_duckduckgo_search_returns_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = Mock()
    tool.ainvoke = AsyncMock(return_value="deepagents search result")
    monkeypatch.setattr(web_search_module, "_get_duckduckgo_search_tool", Mock(return_value=tool))

    result = await _run_duckduckgo_search("deepagents")

    assert result == "deepagents search result"


@pytest.mark.asyncio
async def test_web_search_returns_plain_text_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search_module,
        "_run_duckduckgo_search",
        AsyncMock(return_value="deepagents search result"),
    )

    result = await web_search.coroutine(query="deepagents")

    assert result == "deepagents search result"


@pytest.mark.asyncio
async def test_web_search_reports_no_results_as_business_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search_module,
        "_run_duckduckgo_search",
        AsyncMock(side_effect=WebSearchNoResultsError("DuckDuckGo search returned no results")),
    )

    result = await web_search.coroutine(query="deepagents")

    assert "Unable to complete web search for query: deepagents" in result
    assert "reason=DuckDuckGo search returned no results" in result
    assert "do not retry web_search again for this request" in result


@pytest.mark.asyncio
async def test_web_search_reraises_execution_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search_module,
        "_run_duckduckgo_search",
        AsyncMock(side_effect=ToolException("DuckDuckGo search timed out after 10s")),
    )

    with pytest.raises(ToolException, match="DuckDuckGo search timed out after 10s"):
        await web_search.coroutine(query="deepagents")
