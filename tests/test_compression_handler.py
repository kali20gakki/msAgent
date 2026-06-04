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


@pytest.mark.asyncio
async def test_compression_handler_reports_agent_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(compress_module.console, "print_error", errors.append)
    monkeypatch.setattr(compress_module.console, "print", lambda *_args, **_kwargs: None)

    session = SimpleNamespace(
        context=SimpleNamespace(
            agent="nonexistent",
            thread_id="thread-1",
            working_dir=tmp_path,
            approval_mode=ApprovalMode.ACTIVE,
            tool_output_max_tokens=None,
        ),
        graph=_FakeGraph(),
    )

    async def fake_load_agents_config(_working_dir):
        return SimpleNamespace(get_agent_config=lambda _name: None)

    monkeypatch.setattr(compress_module.initializer, "load_agents_config", fake_load_agents_config)

    handler = compress_module.CompressionHandler(session)
    await handler.handle()

    assert "not found" in errors[0]


@pytest.mark.asyncio
async def test_compression_handler_reports_no_graph_when_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(compress_module.console, "print_error", errors.append)
    monkeypatch.setattr(compress_module.console, "print", lambda *_args, **_kwargs: None)

    session = SimpleNamespace(
        context=SimpleNamespace(
            agent="msagent",
            thread_id="thread-1",
            working_dir=tmp_path,
            approval_mode=ApprovalMode.ACTIVE,
            tool_output_max_tokens=None,
        ),
        graph=None,
    )

    agent_config = SimpleNamespace(
        llm=SimpleNamespace(),
        compression=SimpleNamespace(prompt=None, llm=None, messages_to_keep=1),
    )
    agents_config = SimpleNamespace(get_agent_config=lambda _name: agent_config)

    async def fake_load_agents_config(_working_dir):
        return agents_config

    monkeypatch.setattr(compress_module.initializer, "load_agents_config", fake_load_agents_config)
    monkeypatch.setattr(compress_module.initializer.llm_factory, "create", lambda _config: SimpleNamespace())

    handler = compress_module.CompressionHandler(session)
    await handler.handle()

    assert any("not ready for compression" in e for e in errors)


@pytest.mark.asyncio
async def test_compression_handler_reports_no_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []

    class _EmptyGraph:
        _agent_backend = object()

        async def aget_state(self, _config):
            return SimpleNamespace(values={"messages": []})

    session = SimpleNamespace(
        context=SimpleNamespace(
            agent="msagent",
            thread_id="thread-1",
            working_dir=tmp_path,
            approval_mode=ApprovalMode.ACTIVE,
            tool_output_max_tokens=None,
        ),
        graph=_EmptyGraph(),
    )

    agent_config = SimpleNamespace(
        llm=SimpleNamespace(),
        compression=SimpleNamespace(prompt=None, llm=None, messages_to_keep=1),
    )
    agents_config = SimpleNamespace(get_agent_config=lambda _name: agent_config)

    async def fake_load_agents_config(_working_dir):
        return agents_config

    monkeypatch.setattr(compress_module.initializer, "load_agents_config", fake_load_agents_config)
    monkeypatch.setattr(compress_module.initializer.llm_factory, "create", lambda _config: SimpleNamespace())
    monkeypatch.setattr(compress_module.console, "print_error", errors.append)
    monkeypatch.setattr(compress_module.console, "print", lambda *_args, **_kwargs: None)

    handler = compress_module.CompressionHandler(session)
    await handler.handle()

    assert any("No conversation history" in e for e in errors)


@pytest.mark.asyncio
async def test_compression_handler_reports_backend_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []

    class _NoBackendGraph:
        _agent_backend = None

        async def aget_state(self, _config):
            return SimpleNamespace(values={"messages": [HumanMessage(content="hello")]})

    session = SimpleNamespace(
        context=SimpleNamespace(
            agent="msagent",
            thread_id="thread-1",
            working_dir=tmp_path,
            approval_mode=ApprovalMode.ACTIVE,
            tool_output_max_tokens=None,
        ),
        graph=_NoBackendGraph(),
    )

    agent_config = SimpleNamespace(
        llm=SimpleNamespace(),
        compression=SimpleNamespace(prompt=None, llm=None, messages_to_keep=1),
    )
    agents_config = SimpleNamespace(get_agent_config=lambda _name: agent_config)

    async def fake_load_agents_config(_working_dir):
        return agents_config

    monkeypatch.setattr(compress_module.initializer, "load_agents_config", fake_load_agents_config)
    monkeypatch.setattr(compress_module.initializer.llm_factory, "create", lambda _config: SimpleNamespace())
    monkeypatch.setattr(compress_module.console, "print_error", errors.append)
    monkeypatch.setattr(compress_module.console, "print", lambda *_args, **_kwargs: None)

    handler = compress_module.CompressionHandler(session)
    await handler.handle()

    assert any("unavailable for compression" in e for e in errors)


@pytest.mark.asyncio
async def test_compression_handler_warns_when_already_within_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    warnings: list[str] = []

    class _FakeGraphWithBackend:
        _agent_backend = object()

        async def aget_state(self, _config):
            return SimpleNamespace(values={"messages": [HumanMessage(content="hello")]})

    session = SimpleNamespace(
        context=SimpleNamespace(
            agent="msagent",
            thread_id="thread-1",
            working_dir=tmp_path,
            approval_mode=ApprovalMode.ACTIVE,
            tool_output_max_tokens=None,
        ),
        graph=_FakeGraphWithBackend(),
    )

    agent_config = SimpleNamespace(
        llm=SimpleNamespace(),
        compression=SimpleNamespace(prompt=None, llm=None, messages_to_keep=1),
    )
    agents_config = SimpleNamespace(get_agent_config=lambda _name: agent_config)

    async def fake_load_agents_config(_working_dir):
        return agents_config

    async def fake_perform_conversation_offload(**kwargs):
        return None

    monkeypatch.setattr(compress_module.initializer, "load_agents_config", fake_load_agents_config)
    monkeypatch.setattr(compress_module.initializer.llm_factory, "create", lambda _config: SimpleNamespace())
    monkeypatch.setattr(compress_module, "perform_conversation_offload", fake_perform_conversation_offload)
    monkeypatch.setattr(compress_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(compress_module.console, "print_warning", warnings.append)
    monkeypatch.setattr(compress_module.console, "print", lambda *_args, **_kwargs: None)

    handler = compress_module.CompressionHandler(session)
    await handler.handle()

    assert any("already within the configured retention window" in w for w in warnings)
