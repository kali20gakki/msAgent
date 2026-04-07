"""Runtime helpers for the LangGraph web entrypoint."""

from __future__ import annotations

import asyncio
import atexit
import os
import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from msagent.cli.bootstrap.initializer import initializer

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from msagent.testing.fake_graph import FakeGraph

    WebGraph = CompiledStateGraph | FakeGraph


ENV_WORKING_DIR = "MSAGENT_WEB_WORKING_DIR"
ENV_AGENT = "MSAGENT_WEB_AGENT"
ENV_MODEL = "MSAGENT_WEB_MODEL"
WebGraphCleanup = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class WebGraphOptions:
    """Resolved options used to build the LangGraph-exported graph."""

    working_dir: Path
    agent: str | None = None
    model: str | None = None


@dataclass(slots=True)
class _WebGraphCache:
    """Mutable cache for the exported web graph and its cleanup hook."""

    graph: "WebGraph | None" = None
    cleanup: WebGraphCleanup | None = None
    cleanup_registered: bool = False


@dataclass(slots=True)
class _GraphCreationOutcome:
    """Thread-safe handoff object for synchronous graph construction."""

    graph: "WebGraph | None" = None
    cleanup: WebGraphCleanup | None = None
    error: BaseException | None = None


_CACHE = _WebGraphCache()


def resolve_web_graph_options(env: dict[str, str] | None = None) -> WebGraphOptions:
    """Resolve graph construction options from environment variables."""
    values = env or os.environ
    working_dir_raw = values.get(ENV_WORKING_DIR, "").strip()
    working_dir = Path(working_dir_raw or os.getcwd()).resolve()

    agent = values.get(ENV_AGENT, "").strip() or None
    model = values.get(ENV_MODEL, "").strip() or None
    return WebGraphOptions(
        working_dir=working_dir,
        agent=agent,
        model=model,
    )


async def create_web_graph(
    options: WebGraphOptions,
) -> tuple["WebGraph", WebGraphCleanup]:
    """Create the graph used by the LangGraph server."""
    return await initializer.create_graph(
        agent=options.agent,
        model=options.model,
        working_dir=options.working_dir,
    )


def load_web_graph() -> "WebGraph":
    """Create the graph once and reuse it for LangGraph imports."""
    if _CACHE.graph is not None:
        return _CACHE.graph

    options = resolve_web_graph_options()
    graph, cleanup = _run_graph_creation_sync(options)
    _CACHE.graph = graph
    _CACHE.cleanup = cleanup

    if not _CACHE.cleanup_registered:
        atexit.register(_cleanup_graph)
        _CACHE.cleanup_registered = True

    return graph


def _cleanup_graph() -> None:
    """Close long-lived resources created for the exported graph."""
    cleanup = _CACHE.cleanup
    if cleanup is None:
        return

    cleanup_coro = cast(Coroutine[object, object, None], cleanup())
    try:
        asyncio.run(cleanup_coro)
    except RuntimeError:
        cleanup_coro.close()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(cleanup())
        finally:
            loop.close()


def _reset_cached_web_graph_state() -> None:
    """Reset cached graph state for tests."""
    _CACHE.graph = None
    _CACHE.cleanup = None
    _CACHE.cleanup_registered = False


def _run_graph_creation_sync(
    options: WebGraphOptions,
) -> tuple["WebGraph", WebGraphCleanup]:
    """Build the graph on a dedicated thread so import-time callers stay loop-safe."""
    outcome = _GraphCreationOutcome()

    def _runner() -> None:
        try:
            outcome.graph, outcome.cleanup = asyncio.run(create_web_graph(options))
        except BaseException as exc:  # pragma: no cover - re-raised below
            outcome.error = exc

    thread = threading.Thread(target=_runner, name="msagent-web-graph-init")
    thread.start()
    thread.join()

    if outcome.error is not None:
        raise outcome.error
    if outcome.graph is None or outcome.cleanup is None:
        raise RuntimeError("Graph creation finished without returning a graph and cleanup handler.")

    return outcome.graph, outcome.cleanup
