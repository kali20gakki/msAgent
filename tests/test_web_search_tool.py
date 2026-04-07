from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.tools import ToolException

import msagent.tools.web_search as web_search_module
from msagent.tools.web_search import (
    WebSearchInput,
    _extract_results,
    _fetch_duckduckgo_html,
    _filter_results,
    _normalize_result_url,
    _search_results_with_provider,
    _search_with_tavily,
    web_search,
)


def test_web_search_input_normalizes_domains() -> None:
    payload = WebSearchInput(
        query="  langchain deepagents  ",
        allowed_domains=["https://www.github.com", "github.com", "docs.python.org"],
        blocked_domains=["www.example.com", "example.com"],
    )

    assert payload.query == "langchain deepagents"
    assert payload.allowed_domains == ["github.com", "docs.python.org"]
    assert payload.blocked_domains == ["example.com"]


def test_extract_results_parses_and_deduplicates_html() -> None:
    html = """
    <a class="result__a" href="//example.com/one"> First Result </a>
    <a class="result__a" href="https://example.com/one">Duplicate Result</a>
    <a class="result__a" href="https://docs.python.org/3/">Python Docs</a>
    """

    results = _extract_results(html)

    assert results == [
        {"title": "First Result", "url": "https://example.com/one"},
        {"title": "Python Docs", "url": "https://docs.python.org/3/"},
    ]


def test_filter_results_honors_allowed_and_blocked_domains() -> None:
    results = [
        {"title": "GitHub", "url": "https://github.com/langchain-ai/deepagents"},
        {"title": "Python", "url": "https://docs.python.org/3/"},
        {"title": "Blocked", "url": "https://sub.example.com/page"},
    ]

    filtered = _filter_results(
        results,
        allowed_domains={"github.com", "example.com"},
        blocked_domains={"example.com"},
    )

    assert filtered == [
        {"title": "GitHub", "url": "https://github.com/langchain-ai/deepagents"}
    ]


def test_normalize_result_url_decodes_duckduckgo_redirect() -> None:
    url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Flangchain-ai%2Fdeepagents"

    assert _normalize_result_url(url) == "https://github.com/langchain-ai/deepagents"


@pytest.mark.asyncio
async def test_fetch_duckduckgo_html_translates_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx.Response(503, request=httpx.Request("GET", "https://html.duckduckgo.com/html/"))
    transport = httpx.MockTransport(lambda request: response)
    original_async_client = web_search_module.httpx.AsyncClient

    def _client(*args, **kwargs):
        return original_async_client(transport=transport)

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", _client)

    with pytest.raises(ToolException, match="DuckDuckGo web search request failed"):
        await _fetch_duckduckgo_html("deepagents")


@pytest.mark.asyncio
async def test_search_with_tavily_formats_and_deduplicates_results(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx.Response(
        200,
        json={
            "results": [
                {"title": "deepagents", "url": "https://github.com/langchain-ai/deepagents"},
                {"title": "duplicate", "url": "https://github.com/langchain-ai/deepagents"},
                {"title": "", "url": "https://docs.python.org/3/"},
            ]
        },
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )
    transport = httpx.MockTransport(lambda request: response)
    original_async_client = web_search_module.httpx.AsyncClient

    def _client(*args, **kwargs):
        return original_async_client(transport=transport)

    monkeypatch.setattr(web_search_module.httpx, "AsyncClient", _client)

    results = await _search_with_tavily(
        query="deepagents",
        api_key="test-key",
        allowed_domains=set(),
        blocked_domains=set(),
        limit=5,
    )

    assert results == [
        {"title": "deepagents", "url": "https://github.com/langchain-ai/deepagents"},
        {"title": "https://docs.python.org/3/", "url": "https://docs.python.org/3/"},
    ]


@pytest.mark.asyncio
async def test_search_results_with_provider_uses_tavily_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        web_search_module,
        "_search_with_tavily",
        AsyncMock(return_value=[{"title": "deepagents", "url": "https://github.com/langchain-ai/deepagents"}]),
    )
    monkeypatch.setattr(
        web_search_module,
        "_fetch_duckduckgo_html",
        AsyncMock(side_effect=AssertionError("DuckDuckGo fallback should not be used")),
    )

    results, provider = await _search_results_with_provider(
        query="deepagents",
        allowed_domains=set(),
        blocked_domains=set(),
        limit=5,
    )

    assert provider == "Tavily"
    assert results == [{"title": "deepagents", "url": "https://github.com/langchain-ai/deepagents"}]


@pytest.mark.asyncio
async def test_search_results_with_provider_falls_back_without_tavily_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(
        web_search_module,
        "_fetch_duckduckgo_html",
        AsyncMock(return_value='''<a class="result__a" href="https://github.com/langchain-ai/deepagents">deepagents</a>'''),
    )

    results, provider = await _search_results_with_provider(
        query="deepagents",
        allowed_domains=set(),
        blocked_domains=set(),
        limit=5,
    )

    assert provider == "DuckDuckGo HTML fallback"
    assert results == [{"title": "deepagents", "url": "https://github.com/langchain-ai/deepagents"}]


@pytest.mark.asyncio
async def test_search_results_with_provider_falls_back_when_tavily_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setattr(
        web_search_module,
        "_search_with_tavily",
        AsyncMock(side_effect=ToolException("boom")),
    )
    monkeypatch.setattr(
        web_search_module,
        "_fetch_duckduckgo_html",
        AsyncMock(return_value='''<a class="result__a" href="https://docs.python.org/3/">Python Docs</a>'''),
    )

    results, provider = await _search_results_with_provider(
        query="deepagents",
        allowed_domains=set(),
        blocked_domains=set(),
        limit=5,
    )

    assert provider == "DuckDuckGo HTML fallback"
    assert results == [{"title": "Python Docs", "url": "https://docs.python.org/3/"}]


@pytest.mark.asyncio
async def test_web_search_formats_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search_module,
        "_search_results_with_provider",
        AsyncMock(
            return_value=(
                [{"title": "deepagents", "url": "https://github.com/langchain-ai/deepagents"}],
                "Tavily",
            )
        ),
    )

    result = await web_search.coroutine(query="deepagents")

    assert "Web search results for: deepagents" in result
    assert "Provider: Tavily" in result
    assert "1. deepagents" in result
    assert "URL: https://github.com/langchain-ai/deepagents" in result


@pytest.mark.asyncio
async def test_web_search_returns_no_results_message_when_filtered_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        web_search_module,
        "_search_results_with_provider",
        AsyncMock(return_value=([], "DuckDuckGo HTML fallback")),
    )

    result = await web_search.coroutine(
        query="deepagents",
        allowed_domains=["python.org"],
    )

    assert result == "No web results found for query: deepagents (allowed=python.org)"
