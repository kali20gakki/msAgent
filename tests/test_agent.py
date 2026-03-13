"""Tests for agent module."""

from __future__ import annotations

import asyncio
import tempfile
from typing import Any
from pathlib import Path

import pytest

import msagent.agent as agent_module
from msagent.agent import Agent
from msagent.config import AppConfig, MCPConfig


class FakeMCPManager:
    """Test double for global MCP manager."""

    def __init__(self) -> None:
        self.added_servers: list[str] = []
        self.connected_servers: list[str] = []
        self.tools: list[dict[str, Any]] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.disconnect_called = False
        self.call_result = "tool-result"

    async def add_server(self, config: MCPConfig) -> bool:
        self.added_servers.append(config.name)
        return True

    def get_connected_servers(self) -> list[str]:
        return self.connected_servers

    def get_all_tools(self) -> list[dict[str, Any]]:
        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self.tool_calls.append((tool_name, arguments))
        return self.call_result

    async def disconnect_all(self) -> None:
        self.disconnect_called = True


class FakeLLMClient:
    """Test double for LLM client."""

    def __init__(self) -> None:
        self.chat_response = "assistant-response"
        self.stream_chunks = ["chunk-1", "chunk-2"]
        self.stream_events: list[dict[str, Any]] | None = None
        self.chat_calls = 0
        self.last_usage: dict[str, int] | None = None
        self.context_tokens = 0

    async def chat(self, messages: list[Any], tools: list[dict] | None = None) -> str:
        self.chat_calls += 1
        return self.chat_response

    async def chat_stream(self, messages: list[Any], tools: list[dict] | None = None):
        for chunk in self.stream_chunks:
            yield chunk
            await asyncio.sleep(0)

    async def chat_stream_events(self, messages: list[Any], tools: list[dict] | None = None):
        if self.stream_events is not None:
            for event in self.stream_events:
                yield event
                await asyncio.sleep(0)
            return
        for chunk in self.stream_chunks:
            yield {"type": "text", "content": chunk}
            await asyncio.sleep(0)

    def count_tokens(self, messages: list[Any], tools: list[dict] | None = None) -> int:
        return self.context_tokens


def make_config(api_key: str = "test-key", mcp_servers: list[MCPConfig] | None = None) -> AppConfig:
    config = AppConfig()
    config.llm.api_key = api_key
    config.mcp_servers = mcp_servers or []
    return config


@pytest.mark.asyncio
async def test_initialize_fails_when_llm_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    config = make_config(api_key="")
    agent = Agent(config)

    initialized = await agent.initialize()

    assert initialized is False
    assert "LLM not configured" in agent.error_message


@pytest.mark.asyncio
async def test_initialize_loads_only_enabled_mcp_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    fake_llm = FakeLLMClient()
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)
    monkeypatch.setattr(
        agent_module,
        "create_llm_client",
        lambda _cfg,
        skills=None,
        memory=None,
        recursion_limit=80,
        workspace_root=None,
        tool_invoker=None,
        backend_mode="filesystem": fake_llm,
    )

    config = make_config(
        api_key="configured",
        mcp_servers=[
            MCPConfig(name="enabled", command="python", enabled=True),
            MCPConfig(name="disabled", command="python", enabled=False),
        ],
    )
    agent = Agent(config)

    initialized = await agent.initialize()

    assert initialized is True
    assert agent.is_initialized is True
    assert agent.llm_client is fake_llm
    assert fake_mcp.added_servers == ["enabled"]


@pytest.mark.asyncio
async def test_initialize_passes_deepagents_settings_to_llm_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_mcp = FakeMCPManager()
    fake_llm = FakeLLMClient()
    captured: dict[str, Any] = {}
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills").mkdir()

    def _fake_create_llm_client(
        _cfg,
        skills=None,
        memory=None,
        recursion_limit=80,
        workspace_root=None,
        tool_invoker=None,
        backend_mode="filesystem",
    ):
        captured["skills"] = skills
        captured["memory"] = memory
        captured["recursion_limit"] = recursion_limit
        captured["workspace_root"] = workspace_root
        captured["tool_invoker"] = tool_invoker
        captured["backend_mode"] = backend_mode
        return fake_llm

    monkeypatch.setattr(agent_module, "create_llm_client", _fake_create_llm_client)

    config = make_config(api_key="configured", mcp_servers=[])
    config.deepagents.skills = ["/skills/user/", "/skills/project/"]
    config.deepagents.memory = ["/memory/AGENTS.md"]
    agent = Agent(config)

    initialized = await agent.initialize()

    assert initialized is True
    assert captured["skills"] == [
        "/skills",
        "/skills/user",
        "/skills/project",
    ]
    assert captured["memory"] == ["/memory/AGENTS.md"]
    assert captured["recursion_limit"] == config.deepagents.recursion_limit
    assert captured["workspace_root"] == tmp_path
    assert callable(captured["tool_invoker"])
    assert captured["backend_mode"] == "filesystem"


def test_resolve_skill_sources_does_not_include_packaged_skills_automatically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    packaged_skills_dir = tmp_path / "packaged-skills"
    packaged_skills_dir.mkdir()
    monkeypatch.setattr(Agent, "_PACKAGE_SKILLS_DIR", packaged_skills_dir)

    agent = Agent(make_config(api_key="configured"))

    assert agent._resolve_skill_sources() == []


def test_resolve_skill_sources_normalizes_workspace_absolute_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    custom_skills_dir = tmp_path / "custom-skills"
    custom_skills_dir.mkdir()

    config = make_config(api_key="configured")
    config.deepagents.skills = [str(custom_skills_dir.resolve())]
    agent = Agent(config)

    assert agent._resolve_skill_sources() == ["/custom-skills"]


def test_get_system_prompt_includes_connected_servers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.connected_servers = ["alpha", "beta"]
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    prompt = agent.get_system_prompt()

    assert "alpha, beta" in prompt
    assert "msAgent" in prompt


def test_get_system_prompt_adds_local_shell_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._deepagents_backend_mode = Agent._LOCAL_SHELL_MODE

    prompt = agent.get_system_prompt()

    assert "LocalShellBackend" in prompt
    assert "execute" in prompt


def test_get_status_returns_frontend_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.connected_servers = ["filesystem"]
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    class FakeClient:
        def __init__(self) -> None:
            self.last_usage = {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
            }
            self.counted_tokens = 42

        def count_tokens(self, messages: list[Any], tools: list[dict] | None = None) -> int:
            return self.counted_tokens

    agent = Agent(make_config())
    agent._initialized = True
    agent._loaded_skills = ["code-review"]
    agent.llm_client = FakeClient()
    agent._session_usage_totals = {
        "prompt_tokens": 30,
        "completion_tokens": 10,
        "total_tokens": 40,
    }
    agent._context_tokens = 42

    status = agent.get_status()

    assert status.is_initialized is True
    assert status.backend_mode == "filesystem"
    assert status.connected_servers == ("filesystem",)
    assert status.loaded_skills == ("code-review",)
    assert status.usage is not None
    assert status.usage.total_tokens == 16
    assert status.cumulative_usage is not None
    assert status.cumulative_usage.total_tokens == 40
    assert status.context_tokens == 42


@pytest.mark.asyncio
async def test_chat_returns_error_if_not_initialized() -> None:
    agent = Agent(make_config())
    result = await agent.chat("hello")
    assert "Agent not initialized" in result


@pytest.mark.asyncio
async def test_chat_without_tools_uses_plain_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.tools = []
    fake_llm = FakeLLMClient()
    fake_llm.chat_response = "plain-response"
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True
    agent.llm_client = fake_llm

    result = await agent.chat("hi")

    assert result == "plain-response"
    assert [message.role for message in agent.messages] == ["user", "assistant"]
    assert agent.messages[-1].content == "plain-response"


@pytest.mark.asyncio
async def test_chat_updates_cumulative_usage_and_context_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPManager()
    fake_llm = FakeLLMClient()
    fake_llm.last_usage = {
        "prompt_tokens": 12,
        "completion_tokens": 4,
        "total_tokens": 16,
    }
    fake_llm.context_tokens = 77
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True
    agent.llm_client = fake_llm

    result = await agent.chat("hi")
    status = agent.get_status()

    assert result == "assistant-response"
    assert status.cumulative_usage is not None
    assert status.cumulative_usage.prompt_tokens == 12
    assert status.cumulative_usage.completion_tokens == 4
    assert status.cumulative_usage.total_tokens == 16
    assert status.context_tokens == 77


@pytest.mark.asyncio
async def test_chat_with_tools_passes_tools_through_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.tools = [{"type": "function", "function": {"name": "calc__sum"}}]
    fake_llm = FakeLLMClient()
    fake_llm.chat_response = "The answer is 42"
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True
    agent.llm_client = fake_llm

    result = await agent.chat("what is 40 + 2?")

    assert result == "The answer is 42"
    assert fake_mcp.tool_calls == []
    assert [message.role for message in agent.messages] == ["user", "assistant"]
    assert agent.messages[-1].content == "The answer is 42"


@pytest.mark.asyncio
async def test_chat_stream_with_tools_yields_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.tools = [{"type": "function", "function": {"name": "calc__sum"}}]
    fake_llm = FakeLLMClient()
    fake_llm.stream_chunks = ["one ", "two"]
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True
    agent.llm_client = fake_llm

    chunks = [chunk async for chunk in agent.chat_stream("sum 1 and 9")]

    assert "".join(chunks[:2]) == "one two"
    assert agent.messages[-1].role == "assistant"
    assert agent.messages[-1].content == "one two"


@pytest.mark.asyncio
async def test_stream_chat_events_does_not_fallback_after_tool_only_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mcp = FakeMCPManager()
    fake_mcp.tools = [
        {"type": "function", "function": {"name": "msprof-mcp__msprof_analyze_advisor"}}
    ]
    fake_llm = FakeLLMClient()
    fake_llm.stream_events = [
        {
            "type": "tool_start",
            "name": "msprof-mcp__msprof_analyze_advisor",
            "input": {"mode": "all"},
        },
        {
            "type": "tool_end",
            "name": "msprof-mcp__msprof_analyze_advisor",
            "output": "tool-output",
        },
    ]
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True
    agent.llm_client = fake_llm

    events = [event async for event in agent.stream_chat_events("run advisor")]

    assert [event.type for event in events] == ["tool_call", "tool_result", "done"]
    assert fake_llm.chat_calls == 0
    assert [message.role for message in agent.messages] == ["user"]
    assert [message.content for message in agent.messages] == ["run advisor"]


def test_clear_history_and_get_history_copy() -> None:
    agent = Agent(make_config())
    agent.messages = [agent_module.Message("user", "hello")]

    history = agent.get_history()
    history.append(agent_module.Message("assistant", "world"))

    assert len(agent.messages) == 1
    agent.clear_history()
    assert agent.messages == []


def test_start_new_session_increments_session_and_clears_context() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.last_usage: dict[str, int] | None = {"prompt_tokens": 10, "total_tokens": 10}
            self.counted_tokens = 0

        def count_tokens(self, messages: list[Any], tools: list[dict] | None = None) -> int:
            return self.counted_tokens

    agent = Agent(make_config())
    agent.llm_client = FakeClient()
    agent.messages = [agent_module.Message("user", "hello")]
    agent._session_usage_totals = {
        "prompt_tokens": 10,
        "completion_tokens": 0,
        "total_tokens": 10,
    }
    agent._context_tokens = 25

    session_number = agent.start_new_session()

    assert session_number == 2
    assert agent.session_number == 2
    assert agent.messages == []
    assert agent.llm_client.last_usage is None
    assert agent.get_status().cumulative_usage is not None
    assert agent.get_status().cumulative_usage.total_tokens == 0
    assert agent.get_status().context_tokens == 0


@pytest.mark.asyncio
async def test_shutdown_disconnects_mcp_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mcp = FakeMCPManager()
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)

    agent = Agent(make_config())
    agent._initialized = True

    await agent.shutdown()

    assert fake_mcp.disconnect_called is True
    assert agent.is_initialized is False


@pytest.mark.asyncio
async def test_initialize_warns_when_local_shell_backend_enabled(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_mcp = FakeMCPManager()
    fake_llm = FakeLLMClient()
    monkeypatch.setattr(agent_module, "mcp_manager", fake_mcp)
    monkeypatch.setenv("MSAGENT_ENABLE_LOCAL_SHELL", "1")
    monkeypatch.setattr(
        agent_module,
        "create_llm_client",
        lambda _cfg,
        skills=None,
        memory=None,
        recursion_limit=80,
        workspace_root=None,
        tool_invoker=None,
        backend_mode="filesystem": fake_llm,
    )

    agent = Agent(make_config(api_key="configured", mcp_servers=[]))

    initialized = await agent.initialize()

    captured = capsys.readouterr()
    assert initialized is True
    assert "LocalShellBackend" in captured.err
    assert agent._deepagents_backend_mode == Agent._LOCAL_SHELL_MODE


def test_switch_deepagents_backend_updates_mode_before_initialization() -> None:
    agent = Agent(make_config())

    message = agent.switch_deepagents_backend("local_shell")

    assert agent.get_status().backend_mode == "local_shell"
    assert "LocalShellBackend" in message


@pytest.mark.asyncio
async def test_switch_deepagents_backend_rebuilds_llm_client_when_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = Agent(make_config(api_key="configured", mcp_servers=[]))
    created_modes: list[str] = []

    def _fake_create_llm_client(
        _cfg,
        skills=None,
        memory=None,
        recursion_limit=80,
        workspace_root=None,
        tool_invoker=None,
        backend_mode="filesystem",
    ):
        created_modes.append(backend_mode)
        return FakeLLMClient()

    monkeypatch.setattr(agent_module, "create_llm_client", _fake_create_llm_client)
    agent._initialized = True
    agent._loaded_skill_sources = ["/skills/project/"]
    agent.llm_client = FakeLLMClient()

    message = agent.switch_deepagents_backend("local_shell")

    assert created_modes == ["local_shell"]
    assert agent.get_status().backend_mode == "local_shell"
    assert "LocalShellBackend" in message


def test_switch_deepagents_backend_rejects_invalid_mode() -> None:
    agent = Agent(make_config())

    message = agent.switch_deepagents_backend("unknown")

    assert "不支持的 deepagents backend" in message
    assert agent.get_status().backend_mode == "filesystem"


def test_find_local_files_supports_partial_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "msagent").mkdir()
    (tmp_path / "src" / "msagent" / "tui.py").write_text("print('tui')", encoding="utf-8")
    (tmp_path / "src" / "msagent" / "agent.py").write_text("print('agent')", encoding="utf-8")
    (tmp_path / "docs" / "guide.md").write_text("# guide", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    agent = Agent(make_config())
    candidates = agent.find_local_files("msa/tu", limit=5)

    assert "src/msagent/tui.py" in candidates


def test_find_local_files_empty_query_uses_quick_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('ok')", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    agent = Agent(make_config())
    monkeypatch.setattr(
        agent,
        "_list_workspace_files",
        lambda: (_ for _ in ()).throw(AssertionError("should not call full index")),
    )

    candidates = agent.find_local_files("", limit=5)

    assert "README.md" in candidates


def test_find_local_files_exact_relative_path_avoids_full_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "agent.py").write_text("pass", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    agent = Agent(make_config())
    monkeypatch.setattr(
        agent,
        "_list_workspace_files",
        lambda: (_ for _ in ()).throw(AssertionError("should not call full index")),
    )

    candidates = agent.find_local_files("src/agent.py", limit=5)

    assert candidates == ["src/agent.py"]


def test_inject_file_context_from_at_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "src").mkdir()
    target = tmp_path / "src" / "agent.py"
    target.write_text("def main():\n    return 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    agent = Agent(make_config())
    enriched = agent._inject_file_context("请查看 @src/agent.py 并总结")

    assert "[Attached file context]" in enriched
    assert '<file path="src/agent.py">' in enriched


def test_find_local_files_supports_absolute_path_query() -> None:
    root = Path(tempfile.gettempdir()).resolve()
    candidates = Agent(make_config()).find_local_files(root.as_posix(), limit=20)
    assert any(path.startswith(root.as_posix()) for path in candidates)
