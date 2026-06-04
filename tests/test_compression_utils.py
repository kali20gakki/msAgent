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
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from msagent.utils import compression as compression_module


def test_calculate_message_tokens_uses_llm_tokenizer_when_available() -> None:
    class _FakeLLM:
        def __init__(self) -> None:
            self.received = None

        def get_num_tokens_from_messages(self, messages):
            self.received = messages
            return 12

    llm = _FakeLLM()
    messages = [HumanMessage(content="hello"), AIMessage(content="world")]

    assert compression_module.calculate_message_tokens(messages, llm) == 12
    assert [message.content for message in llm.received] == ["hello", "world"]


def test_calculate_message_tokens_falls_back_to_tiktoken_and_character_estimate(monkeypatch) -> None:
    class _BrokenLLM:
        def get_num_tokens_from_messages(self, _messages):
            raise NotImplementedError

    messages = [HumanMessage(content="hello world")]
    monkeypatch.setattr(
        compression_module.tiktoken,
        "get_encoding",
        lambda _name: SimpleNamespace(encode=lambda text: [1] * len(text.split())),
    )
    assert compression_module.calculate_message_tokens(messages, _BrokenLLM()) == 2

    monkeypatch.setattr(
        compression_module.tiktoken,
        "get_encoding",
        lambda _name: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert compression_module.calculate_message_tokens(messages, _BrokenLLM()) == len("hello world") // 4


def test_should_auto_compress_respects_context_window_and_threshold() -> None:
    assert compression_module.should_auto_compress(80, None, 0.5) is False
    assert compression_module.should_auto_compress(80, 0, 0.5) is False
    assert compression_module.should_auto_compress(40, 100, 0.5) is False
    assert compression_module.should_auto_compress(50, 100, 0.5) is True


@pytest.mark.asyncio
async def test_compress_messages_handles_empty_system_only_and_keep_tail(monkeypatch) -> None:
    async def fake_summarize(messages, _llm, prompt=None, prompt_vars=None):
        return AIMessage(content=f"summary:{len(messages)}:{prompt}:{prompt_vars['tag']}")

    monkeypatch.setattr(compression_module, "_summarize_messages", fake_summarize)

    assert await compression_module.compress_messages([], SimpleNamespace()) == []

    system_only = [SystemMessage(content="rules")]
    assert await compression_module.compress_messages(system_only, SimpleNamespace()) == system_only

    messages = [
        SystemMessage(content="rules"),
        HumanMessage(content="u1"),
        AIMessage(content="a1"),
        HumanMessage(content="u2"),
    ]
    compressed = await compression_module.compress_messages(
        messages,
        SimpleNamespace(),
        messages_to_keep=1,
        prompt="Prompt {conversation}",
        prompt_vars={"tag": "x"},
    )

    assert compressed[0].content == "rules"
    assert compressed[1].content == "summary:2:Prompt {conversation}:x"
    assert compressed[2].content == "u2"

    keep_all_non_system = await compression_module.compress_messages(
        [SystemMessage(content="rules"), HumanMessage(content="u1")],
        SimpleNamespace(),
        messages_to_keep=5,
    )
    assert [message.content for message in keep_all_non_system] == ["rules", "u1"]


@pytest.mark.asyncio
async def test_summarize_messages_renders_prompt_and_marks_summary_name() -> None:
    captured_messages = []

    class _FakeCompressionLLM:
        async def ainvoke(self, messages):
            captured_messages.extend(messages)
            return AIMessage(content="condensed result")

    summary = await compression_module._summarize_messages(
        [HumanMessage(content="hello"), AIMessage(content="world")],
        _FakeCompressionLLM(),
        prompt="Focus on {topic}",
        prompt_vars={"topic": "testing"},
    )

    assert summary.name == "compression_summary"
    assert "Previous conversation summary" in summary.text
    assert "Conversation:\nHuman: hello" in captured_messages[0].content
    assert "Focus on testing" in captured_messages[0].content
