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

from types import SimpleNamespace

import pytest
from langchain_core.messages import SystemMessage

import msagent.agents.factory as factory_module
from msagent.agents.factory import AgentFactory, _SystemMessageMiddleware


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


def _make_model_request(
    *,
    system_message: SystemMessage,
    template_vars: dict[str, object] | None,
) -> SimpleNamespace:
    request = SimpleNamespace(
        system_message=system_message,
        runtime=SimpleNamespace(context=SimpleNamespace(template_vars=template_vars or {})),
    )

    def _override(**overrides):
        data = {
            "system_message": request.system_message,
            "runtime": request.runtime,
        }
        data.update(overrides)
        return SimpleNamespace(**data, override=_override)

    request.override = _override
    return request


def _patch_deepagent_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyBackend:
        def __init__(self, root_dir: str, **kwargs):
            self.root_dir = root_dir
            self.kwargs = kwargs

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
    assert {
        "fetch_tools",
        "get_tool",
        "run_tool",
        "fetch_skills",
        "get_skill",
        "web_search",
    } <= tool_names
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
    positive, negative = factory._compile_tool_patterns(["impl:deepagents:get_skill", "mcp:msprof-mcp:*"])

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


def test_system_message_middleware_renders_known_vars_and_preserves_unknown_placeholders() -> None:
    request = _make_model_request(
        system_message=SystemMessage(
            content="cwd={working_dir}; local={local_environment_context}; worker={worker}; rank={Rank_ID}"
        ),
        template_vars={
            "working_dir": "/tmp/project",
            "local_environment_context": "GPU=Ascend",
        },
    )

    updated = _SystemMessageMiddleware._render_request_system_message(request)

    assert updated is not request
    assert str(updated.system_message.content) == (
        "cwd=/tmp/project; local=GPU=Ascend; worker={worker}; rank={Rank_ID}"
    )


def test_system_message_middleware_leaves_request_unchanged_without_template_vars() -> None:
    request = _make_model_request(
        system_message=SystemMessage(content="cwd={working_dir}; worker={worker}"),
        template_vars={},
    )

    updated = _SystemMessageMiddleware._render_request_system_message(request)

    assert updated is request
    assert str(updated.system_message.content) == "cwd={working_dir}; worker={worker}"


def test_system_message_middleware_renders_text_blocks_only() -> None:
    content = [
        {
            "type": "text",
            "text": "cwd={working_dir}",
            "metadata": {"template": "{working_dir}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/{working_dir}.png"},
        },
    ]
    request = _make_model_request(
        system_message=SystemMessage(content=content),
        template_vars={"working_dir": "/workspace"},
    )

    updated = _SystemMessageMiddleware._render_request_system_message(request)

    assert updated is not request
    assert updated.system_message.content == [
        {
            "type": "text",
            "text": "cwd=/workspace",
            "metadata": {"template": "{working_dir}"},
        },
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/{working_dir}.png"},
        },
    ]


def test_system_message_middleware_preserves_system_message_metadata() -> None:
    request = _make_model_request(
        system_message=SystemMessage(
            content="cwd={working_dir}",
            additional_kwargs={"source": "runtime"},
            response_metadata={"trace_id": "abc"},
            name="system-name",
            id="system-id",
        ),
        template_vars={"working_dir": "/workspace"},
    )

    updated = _SystemMessageMiddleware._render_request_system_message(request)

    assert str(updated.system_message.content) == "cwd=/workspace"
    assert updated.system_message.additional_kwargs == {"source": "runtime"}
    assert updated.system_message.response_metadata == {"trace_id": "abc"}
    assert updated.system_message.name == "system-name"
    assert updated.system_message.id == "system-id"


@pytest.mark.asyncio
async def test_system_message_middleware_awrap_model_call_applies_rendering() -> None:
    middleware = _SystemMessageMiddleware()
    request = _make_model_request(
        system_message=SystemMessage(content="local={local_environment_context}; rank={Rank_ID}"),
        template_vars={"local_environment_context": "NPU=910B"},
    )

    captured = {}

    async def _handler(updated_request):
        captured["request"] = updated_request
        return "ok"

    result = await middleware.awrap_model_call(request, _handler)

    assert result == "ok"
    assert str(captured["request"].system_message.content) == "local=NPU=910B; rank={Rank_ID}"


@pytest.mark.asyncio
async def test_agent_factory_maps_retry_config_to_deepagents_primitives(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyBackend:
        def __init__(self, root_dir: str, **kwargs):
            self.root_dir = root_dir
            self.kwargs = kwargs

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
    tool_retry = next(item for item in middleware if item.__class__.__name__ == "ToolRetryMiddleware")
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
        def __init__(self, root_dir: str, **kwargs):
            self.root_dir = root_dir
            self.kwargs = kwargs

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
        def __init__(self, root_dir: str, **kwargs):
            self.root_dir = root_dir
            self.kwargs = kwargs

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
    assert any(item.__class__.__name__ == "ToolResultEvictionMiddleware" for item in middleware)


def test_build_composite_backend_persists_conversation_history_under_workdir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _DummyLocalBackend:
        def __init__(self, root_dir: str, **kwargs):
            self.root_dir = root_dir
            self.kwargs = kwargs

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

    assert conversation_history_backend.root_dir == (tmp_path / factory_module.CONFIG_CONVERSATION_HISTORY_DIR)
    assert conversation_history_backend.virtual_mode is True


def test_build_composite_backend_inherits_parent_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    class _DummyLocalBackend:
        def __init__(self, root_dir: str, **kwargs):
            captured["root_dir"] = root_dir
            captured.update(kwargs)

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

    AgentFactory._build_composite_backend(tmp_path)

    assert captured["root_dir"] == str(tmp_path)
    assert captured["inherit_env"] is True


@pytest.mark.asyncio
async def test_agent_factory_injects_local_environment_placeholder_into_system_prompt(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyBackend:
        def __init__(self, root_dir: str, **kwargs):
            self.root_dir = root_dir
            self.kwargs = kwargs

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


@pytest.mark.asyncio
async def test_should_prefer_search_mcp_requires_valid_tavily_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory_module, "_TAVILY_KEY_VALIDATION_CACHE", {})
    monkeypatch.setattr(
        AgentFactory,
        "_probe_tavily_api_key",
        staticmethod(lambda api_key: _return_true(api_key)),
    )
    mcp_client = SimpleNamespace(
        config=SimpleNamespace(
            servers={
                "tavily-mcp": SimpleNamespace(
                    enabled=True,
                    env={"TAVILY_API_KEY": "tvly-valid-key"},
                )
            }
        )
    )

    assert (
        await AgentFactory._should_prefer_search_mcp(
            mcp_client,
            mcp_tools=[SimpleNamespace(name="tavily_search")],
            mcp_module_map={"tavily_search": "mcp:tavily-mcp"},
        )
        is True
    )


@pytest.mark.asyncio
async def test_should_prefer_search_mcp_keeps_builtin_web_search_for_invalid_tavily_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory_module, "_TAVILY_KEY_VALIDATION_CACHE", {})
    monkeypatch.setattr(
        AgentFactory,
        "_probe_tavily_api_key",
        staticmethod(lambda api_key: _return_false(api_key)),
    )
    mcp_client = SimpleNamespace(
        config=SimpleNamespace(
            servers={
                "tavily-mcp": SimpleNamespace(
                    enabled=True,
                    env={"TAVILY_API_KEY": "tvly-invalid-key"},
                )
            }
        )
    )

    assert (
        await AgentFactory._should_prefer_search_mcp(
            mcp_client,
            mcp_tools=[SimpleNamespace(name="tavily_search")],
            mcp_module_map={"tavily_search": "mcp:tavily-mcp"},
        )
        is False
    )


@pytest.mark.asyncio
async def test_should_prefer_search_mcp_requires_actual_search_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory_module, "_TAVILY_KEY_VALIDATION_CACHE", {})
    mcp_client = SimpleNamespace(
        config=SimpleNamespace(
            servers={
                "brave-search": SimpleNamespace(
                    enabled=True,
                    env={},
                )
            }
        )
    )

    assert (
        await AgentFactory._should_prefer_search_mcp(
            mcp_client,
            mcp_tools=[SimpleNamespace(name="brave_fetch")],
            mcp_module_map={"brave_fetch": "mcp:brave-search"},
        )
        is False
    )


@pytest.mark.asyncio
async def test_should_prefer_search_mcp_prefers_non_tavily_search_tool_when_available() -> None:
    mcp_client = SimpleNamespace(
        config=SimpleNamespace(
            servers={
                "brave-search": SimpleNamespace(
                    enabled=True,
                    env={},
                )
            }
        )
    )

    assert (
        await AgentFactory._should_prefer_search_mcp(
            mcp_client,
            mcp_tools=[SimpleNamespace(name="brave_search")],
            mcp_module_map={"brave_search": "mcp:brave-search"},
        )
        is True
    )


@pytest.mark.asyncio
async def test_resolve_tavily_api_key_reads_env_placeholders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-env")

    key = AgentFactory._resolve_tavily_api_key(SimpleNamespace(env={"TAVILY_API_KEY": "${TAVILY_API_KEY}"}))

    assert key == "tvly-from-env"


@pytest.mark.asyncio
async def test_probe_tavily_api_key_uses_usage_endpoint_and_caches_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(factory_module, "_TAVILY_KEY_VALIDATION_CACHE", {})
    calls: list[tuple[str, dict[str, str]]] = []

    class _DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url: str, headers: dict[str, str]):
            calls.append((url, headers))
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(factory_module.httpx, "AsyncClient", lambda **kwargs: _DummyClient())

    assert await AgentFactory._probe_tavily_api_key("tvly-valid") is True
    assert await AgentFactory._probe_tavily_api_key("tvly-valid") is True
    assert calls == [("https://api.tavily.com/usage", {"Authorization": "Bearer tvly-valid"})]


async def _return_true(_api_key: str) -> bool:
    return True


async def _return_false(_api_key: str) -> bool:
    return False


async def test_agent_factory_passes_resolved_subagents_to_create_deep_agent(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _DummyBackend:
        def __init__(self, root_dir: str, **kwargs):
            self.root_dir = root_dir
            self.kwargs = kwargs

    def _fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return _DummyGraph()

    monkeypatch.setattr(factory_module, "LocalShellBackend", _DummyBackend)
    monkeypatch.setattr(factory_module, "create_deep_agent", _fake_create_deep_agent)

    from msagent.configs import SubAgentConfig
    from msagent.configs.llm import LLMConfig, LLMProvider

    llm = LLMConfig.model_construct(
        provider=LLMProvider.OPENAI,
        model="gpt-4",
        max_tokens=1024,
        temperature=0.0,
    )
    explorer = SubAgentConfig.model_construct(
        name="explorer",
        description="explores code",
        prompt="You explore.",
        llm=llm,
    )
    general_purpose = SubAgentConfig.model_construct(
        name="general-purpose",
        description="gp",
        prompt="gp prompt",
        llm=llm,
    )

    config = SimpleNamespace(
        name="msagent",
        prompt="test prompt",
        llm=SimpleNamespace(),
        tools=None,
        subagents=[general_purpose, explorer],
        retry=None,
    )

    await AgentFactory(llm_factory=_DummyLLMFactory()).create(
        config=config,
        mcp_client=_DummyMCPClient(),
        llm_config=SimpleNamespace(),
    )

    subs = captured.get("subagents")
    assert subs is not None
    assert len(subs) == 1
    assert subs[0]["name"] == "explorer"
    assert "You explore." in str(subs[0]["system_prompt"])
