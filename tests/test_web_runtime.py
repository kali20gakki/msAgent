from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from msagent.web import runtime


def test_load_web_graph_caches_graph(monkeypatch) -> None:
    runtime._reset_cached_web_graph_state()
    calls = {"count": 0}

    async def _fake_cleanup() -> None:
        return None

    def _fake_run_graph_creation_sync(options):
        del options
        calls["count"] += 1
        return SimpleNamespace(name="fake-graph"), _fake_cleanup

    monkeypatch.setattr(runtime, "_run_graph_creation_sync", _fake_run_graph_creation_sync)

    graph_one = runtime.load_web_graph()
    graph_two = runtime.load_web_graph()

    assert graph_one is graph_two
    assert calls["count"] == 1


def test_graph_factory_returns_cached_graph(monkeypatch) -> None:
    runtime._reset_cached_web_graph_state()
    sentinel = SimpleNamespace(name="awaited-graph")

    def _fake_load_web_graph():
        return sentinel

    monkeypatch.setattr("msagent.web.graph.load_web_graph", _fake_load_web_graph)

    from msagent.web.graph import graph

    assert graph() is sentinel


def test_cleanup_graph_runs_cached_cleanup() -> None:
    runtime._reset_cached_web_graph_state()
    calls = {"count": 0}

    async def _fake_cleanup() -> None:
        calls["count"] += 1

    runtime._CACHE.cleanup = _fake_cleanup

    runtime._cleanup_graph()

    assert calls["count"] == 1


def test_cleanup_graph_falls_back_to_new_event_loop(monkeypatch) -> None:
    runtime._reset_cached_web_graph_state()
    calls = {"count": 0}
    original_asyncio_run = asyncio.run

    async def _fake_cleanup() -> None:
        calls["count"] += 1

    class _FakeLoop:
        def __init__(self) -> None:
            self.closed = False
            self.awaited = None

        def run_until_complete(self, awaitable):
            self.awaited = awaitable
            return original_asyncio_run(awaitable)

        def close(self) -> None:
            self.closed = True

    fake_loop = _FakeLoop()

    runtime._CACHE.cleanup = _fake_cleanup
    monkeypatch.setattr(runtime.asyncio, "run", lambda awaitable: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(runtime.asyncio, "new_event_loop", lambda: fake_loop)

    runtime._cleanup_graph()

    assert calls["count"] == 1
    assert fake_loop.awaited is not None
    assert fake_loop.closed is True


def test_run_graph_creation_sync_reraises_creation_error(monkeypatch, tmp_path) -> None:
    expected = RuntimeError("boom")

    async def _fake_create_web_graph(options):
        del options
        raise expected

    monkeypatch.setattr(runtime, "create_web_graph", _fake_create_web_graph)

    with pytest.raises(RuntimeError, match="boom") as exc_info:
        runtime._run_graph_creation_sync(
            runtime.WebGraphOptions(working_dir=tmp_path),
        )

    assert exc_info.value is expected
