from __future__ import annotations

from types import SimpleNamespace

import pytest

import msagent.agents.factory as factory_module
from msagent.agents.factory import AgentFactory


class _DummyLLMFactory:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, _config, **kwargs):
        self.calls.append(kwargs)
        return "dummy-model"


class _DummyMCPClient:
    module_map: dict[str, str] = {}

    async def tools(self):
        return []


class _DummyGraph:
    pass


def _patch_deepagent_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyBackend:
        def __init__(self, root_dir: str):
            self.root_dir = root_dir

    monkeypatch.setattr(factory_module, "LocalShellBackend", _DummyBackend)
    monkeypatch.setattr(
        factory_module,
        "create_deep_agent",
        lambda **_kwargs: _DummyGraph(),
    )


@pytest.mark.asyncio
async def test_agent_factory_create_populates_runtime_tools_without_name_error(
    monkeypatch,
) -> None:
    _patch_deepagent_entrypoints(monkeypatch)
    llm_factory = _DummyLLMFactory()

    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=None,
    )

    graph = await AgentFactory(llm_factory=llm_factory).create(
        config=config,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    assert hasattr(graph, "_llm_tools")
    assert hasattr(graph, "_tools_in_catalog")
    assert hasattr(graph, "_agent_backend")
    tool_names = {tool.name for tool in graph._llm_tools}
    assert {"fetch_tools", "get_tool", "run_tool", "fetch_skills", "get_skill", "web_search"} <= tool_names
    assert "write_todos" not in tool_names


@pytest.mark.asyncio
async def test_agent_factory_create_uses_explicit_working_dir_for_backends(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_deepagent_entrypoints(monkeypatch)

    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=None,
    )

    graph = await AgentFactory(llm_factory=_DummyLLMFactory()).create(
        config=config,
        working_dir=tmp_path,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    assert graph._agent_backend.default.root_dir == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_agent_factory_create_patches_deepagents_windows_path_validation(
    monkeypatch,
) -> None:
    _patch_deepagent_entrypoints(monkeypatch)
    called = False

    def _fake_patch() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(
        factory_module,
        "patch_deepagents_windows_absolute_paths",
        _fake_patch,
    )

    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=None,
    )

    await AgentFactory(llm_factory=_DummyLLMFactory()).create(
        config=config,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    assert called is True


@pytest.mark.asyncio
async def test_agent_factory_create_filters_tools_by_patterns_for_impl_and_mcp(
    monkeypatch,
) -> None:
    _patch_deepagent_entrypoints(monkeypatch)
    llm_factory = _DummyLLMFactory()

    class _PatternMCPClient:
        module_map = {
            "alpha_ping": "mcp:alpha",
            "beta_info": "mcp:beta",
        }

        async def tools(self):
            return [
                SimpleNamespace(name="alpha_ping"),
                SimpleNamespace(name="beta_info"),
            ]

    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=SimpleNamespace(
            patterns=["impl:deepagents:get_skill", "mcp:alpha:*"],
            execution_timeout_seconds=None,
        ),
    )

    graph = await AgentFactory(llm_factory=llm_factory).create(
        config=config,
        mcp_client=_PatternMCPClient(),
        llm_config=SimpleNamespace(),
    )

    tool_names = {tool.name for tool in graph._llm_tools}
    assert tool_names == {"get_skill", "alpha_ping"}


@pytest.mark.asyncio
async def test_agent_factory_create_supports_negative_tool_patterns(
    monkeypatch,
) -> None:
    _patch_deepagent_entrypoints(monkeypatch)
    llm_factory = _DummyLLMFactory()

    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=SimpleNamespace(
            patterns=[
                "impl:deepagents:*",
                "!impl:deepagents:run_tool",
                "!impl:deepagents:fetch_*",
                "!impl:deepagents:web_search",
            ],
            execution_timeout_seconds=None,
        ),
    )

    graph = await AgentFactory(llm_factory=llm_factory).create(
        config=config,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    tool_names = {tool.name for tool in graph._llm_tools}
    assert "run_tool" not in tool_names
    assert "fetch_tools" not in tool_names
    assert "web_search" not in tool_names
    assert "get_skill" in tool_names


def test_agent_factory_filters_deepagents_default_tools_from_model_request() -> None:
    factory = AgentFactory(llm_factory=_DummyLLMFactory())
    positive, negative = factory._compile_tool_patterns(
        ["impl:deepagents:get_skill", "mcp:msprof-mcp:*"]
    )

    tools = [
        SimpleNamespace(name="execute"),
        SimpleNamespace(name="task"),
        SimpleNamespace(name="get_skill"),
        SimpleNamespace(name="msprof-mcp_ping"),
    ]

    filtered = factory._filter_tool_objects_by_patterns(
        tools=tools,
        positive_patterns=positive,
        negative_patterns=negative,
        mcp_module_map={"msprof-mcp_ping": "mcp:msprof-mcp"},
        mcp_servers={"msprof-mcp"},
    )

    assert {tool.name for tool in filtered} == {"get_skill", "msprof-mcp_ping"}


@pytest.mark.asyncio
async def test_agent_factory_maps_retry_config_to_deepagents_primitives(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyBackend:
        def __init__(self, root_dir: str):
            self.root_dir = root_dir

    def _fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return _DummyGraph()

    monkeypatch.setattr(factory_module, "LocalShellBackend", _DummyBackend)
    monkeypatch.setattr(factory_module, "create_deep_agent", _fake_create_deep_agent)

    llm_factory = _DummyLLMFactory()
    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=None,
        retry=SimpleNamespace(
            enabled=True,
            model=SimpleNamespace(
                enabled=True,
                max_retries=4,
                timeout=77.0,
            ),
            tool=SimpleNamespace(
                enabled=True,
                max_retries=4,
                tools=["alpha_ping"],
                retry_on=["TimeoutError", "ConnectionError"],
                on_failure="error",
                backoff_factor=3.0,
                initial_delay=2.5,
                max_delay=30.0,
                jitter=False,
            ),
        ),
    )

    await AgentFactory(llm_factory=llm_factory).create(
        config=config,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    assert llm_factory.calls[-1]["max_retries"] == 4
    assert llm_factory.calls[-1]["timeout_seconds"] == 77.0

    middleware = captured["middleware"]
    assert isinstance(middleware, list)
    tool_retry = next(
        item for item in middleware if item.__class__.__name__ == "ToolRetryMiddleware"
    )
    assert tool_retry.max_retries == 4
    assert tool_retry.tools == []
    assert tool_retry._tool_filter == ["alpha_ping"]
    assert tool_retry.retry_on == (TimeoutError, ConnectionError)
    assert tool_retry.on_failure == "error"
    assert tool_retry.backoff_factor == 3.0
    assert tool_retry.initial_delay == 2.5
    assert tool_retry.max_delay == 30.0
    assert tool_retry.jitter is False


@pytest.mark.asyncio
async def test_agent_factory_disables_retry_when_flag_off(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyBackend:
        def __init__(self, root_dir: str):
            self.root_dir = root_dir

    def _fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return _DummyGraph()

    monkeypatch.setattr(factory_module, "LocalShellBackend", _DummyBackend)
    monkeypatch.setattr(factory_module, "create_deep_agent", _fake_create_deep_agent)

    llm_factory = _DummyLLMFactory()
    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=None,
        retry=SimpleNamespace(
            enabled=False,
            model=SimpleNamespace(
                enabled=True,
                max_retries=9,
                timeout=200.0,
            ),
            tool=SimpleNamespace(
                enabled=True,
                max_retries=9,
                tools=None,
                retry_on=None,
                on_failure="continue",
                backoff_factor=2.0,
                initial_delay=1.0,
                max_delay=10.0,
                jitter=True,
            ),
        ),
    )

    await AgentFactory(llm_factory=llm_factory).create(
        config=config,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    assert llm_factory.calls[-1]["max_retries"] == 0
    assert llm_factory.calls[-1]["timeout_seconds"] is None
    middleware = captured["middleware"]
    assert isinstance(middleware, list)
    assert all(item.__class__.__name__ != "ToolRetryMiddleware" for item in middleware)


@pytest.mark.asyncio
async def test_agent_factory_adds_tool_result_eviction_middleware_when_output_limit_configured(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyBackend:
        def __init__(self, root_dir: str):
            self.root_dir = root_dir

    def _fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return _DummyGraph()

    monkeypatch.setattr(factory_module, "LocalShellBackend", _DummyBackend)
    monkeypatch.setattr(factory_module, "create_deep_agent", _fake_create_deep_agent)

    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=SimpleNamespace(
            patterns=["impl:deepagents:*"],
            execution_timeout_seconds=None,
            output_max_tokens=1234,
        ),
    )

    await AgentFactory(llm_factory=_DummyLLMFactory()).create(
        config=config,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    middleware = captured["middleware"]
    assert isinstance(middleware, list)
    assert any(
        item.__class__.__name__ == "ToolResultEvictionMiddleware"
        for item in middleware
    )


def test_build_composite_backend_persists_conversation_history_under_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _DummyLocalBackend:
        def __init__(self, root_dir: str):
            self.root_dir = root_dir

    class _DummyFilesystemBackend:
        def __init__(self, *, root_dir, virtual_mode):
            self.root_dir = root_dir
            self.virtual_mode = virtual_mode

    class _DummyCompositeBackend:
        def __init__(self, *, default, routes):
            self.default = default
            self.routes = routes

    monkeypatch.setattr(factory_module, "LocalShellBackend", _DummyLocalBackend)
    monkeypatch.setattr(factory_module, "FilesystemBackend", _DummyFilesystemBackend)
    monkeypatch.setattr(factory_module, "CompositeBackend", _DummyCompositeBackend)

    backend = AgentFactory._build_composite_backend(tmp_path)
    conversation_history_backend = backend.routes["/conversation_history/"]

    assert conversation_history_backend.root_dir == (
        tmp_path / factory_module.CONFIG_CONVERSATION_HISTORY_DIR
    )
    assert conversation_history_backend.virtual_mode is True


@pytest.mark.asyncio
async def test_agent_factory_injects_local_environment_placeholder_into_system_prompt(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyBackend:
        def __init__(self, root_dir: str):
            self.root_dir = root_dir

    def _fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return _DummyGraph()

    monkeypatch.setattr(factory_module, "LocalShellBackend", _DummyBackend)
    monkeypatch.setattr(factory_module, "create_deep_agent", _fake_create_deep_agent)

    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=None,
    )

    await AgentFactory(llm_factory=_DummyLLMFactory()).create(
        config=config,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    system_prompt = str(captured.get("system_prompt"))
    assert "test prompt" in system_prompt
    assert "{local_environment_context}" in system_prompt
