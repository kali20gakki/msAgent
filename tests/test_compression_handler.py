from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from msagent.cli.handlers import compress as compress_module
from msagent.configs import ApprovalMode
from msagent.utils.offload import ConversationOffloadResult


class _FakeGraph:
    def __init__(self) -> None:
        self._agent_backend = object()
        self.updated: list[tuple[object, dict]] = []

    async def aget_state(self, _config):
        return SimpleNamespace(
            values={
                "messages": [HumanMessage(content="hello"), HumanMessage(content="world")],
                "_summarization_event": None,
            }
        )

    async def aupdate_state(self, config, update: dict) -> None:
        self.updated.append((config, update))


@pytest.mark.asyncio
async def test_compression_handler_updates_current_thread_with_offload_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(
        context=SimpleNamespace(
            agent="msagent",
            thread_id="thread-1",
            working_dir=tmp_path,
            approval_mode=ApprovalMode.ACTIVE,
            tool_output_max_tokens=None,
            current_input_tokens=500,
            current_output_tokens=100,
        ),
        graph=_FakeGraph(),
        update_context=lambda **kwargs: session.context.__dict__.update(kwargs),
    )

    agent_config = SimpleNamespace(
        llm=SimpleNamespace(),
        compression=SimpleNamespace(
            prompt=None,
            llm=None,
            messages_to_keep=1,
        ),
    )
    agents_config = SimpleNamespace(get_agent_config=lambda _name: agent_config)

    async def fake_load_agents_config(_working_dir):
        return agents_config

    async def fake_load_user_memory(_working_dir):
        return ""

    async def fake_perform_conversation_offload(**kwargs):
        assert kwargs["thread_id"] == "thread-1"
        return ConversationOffloadResult(
            new_event={
                "cutoff_index": 1,
                "summary_message": HumanMessage(content="summary"),
                "file_path": "/conversation_history/thread-1.md",
            },
            messages_offloaded=1,
            messages_kept=1,
            tokens_before=200,
            tokens_after=80,
            pct_decrease=60,
            offload_warning=None,
        )

    printed_success: list[str] = []

    monkeypatch.setattr(
        compress_module.initializer,
        "load_agents_config",
        fake_load_agents_config,
    )
    monkeypatch.setattr(
        compress_module.initializer,
        "load_user_memory",
        fake_load_user_memory,
    )
    monkeypatch.setattr(
        compress_module.initializer.llm_factory,
        "create",
        lambda _config: SimpleNamespace(),
    )
    monkeypatch.setattr(
        compress_module,
        "perform_conversation_offload",
        fake_perform_conversation_offload,
    )
    monkeypatch.setattr(compress_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(compress_module.console, "print_warning", lambda *_args: None)
    monkeypatch.setattr(compress_module.console, "print_success", printed_success.append)
    monkeypatch.setattr(compress_module.console, "print", lambda *_args, **_kwargs: None)

    handler = compress_module.CompressionHandler(session)
    await handler.handle()

    assert session.context.thread_id == "thread-1"
    assert session.context.current_input_tokens == 80
    assert session.context.current_output_tokens == 0
    assert len(session.graph.updated) == 1
    assert session.graph.updated[0][1] == {
        "_summarization_event": {
            "cutoff_index": 1,
            "summary_message": HumanMessage(content="summary"),
            "file_path": "/conversation_history/thread-1.md",
        }
    }
    assert printed_success
