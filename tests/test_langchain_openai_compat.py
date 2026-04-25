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

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai.chat_models import base as openai_base

from msagent.utils.langchain_openai_compat import patch_chat_openai_reasoning_content_support


def test_patch_replays_reasoning_content_in_chat_completions_requests() -> None:
    patch_chat_openai_reasoning_content_support()

    payload = openai_base._convert_message_to_dict(
        AIMessage(
            content="",
            additional_kwargs={
                "reasoning_content": "need to inspect the workspace first",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "ls", "arguments": "{\"path\":\".\"}"},
                    }
                ],
            },
        )
    )

    assert payload["reasoning_content"] == "need to inspect the workspace first"
    assert payload["content"] is None
    assert payload["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "ls", "arguments": "{\"path\":\".\"}"},
        }
    ]


def test_patch_extracts_reasoning_content_from_chat_completion_responses() -> None:
    patch_chat_openai_reasoning_content_support()

    message = openai_base._convert_dict_to_message(
        {
            "role": "assistant",
            "content": "I found the issue.",
            "reasoning_content": "first inspect the log, then replay the tool loop",
        }
    )

    assert isinstance(message, AIMessage)
    assert message.content == "I found the issue."
    assert message.additional_kwargs["reasoning_content"] == ("first inspect the log, then replay the tool loop")


def test_patch_extracts_reasoning_content_from_streaming_deltas() -> None:
    patch_chat_openai_reasoning_content_support()

    chunk = openai_base._convert_delta_to_message_chunk(
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "step 1 -> step 2 -> call tool",
        },
        AIMessageChunk,
    )

    assert isinstance(chunk, AIMessageChunk)
    assert chunk.additional_kwargs["reasoning_content"] == "step 1 -> step 2 -> call tool"
