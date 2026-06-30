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

"""Local fake backend graph used by CLI entrypoint E2E tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from msagent.tools.internal.todo import TODO_PANEL_MARKER, format_todos
from msagent.utils.render import TOOL_TIMING_RESPONSE_METADATA_KEY


def _extract_prompt(input_data: dict[str, Any] | Any) -> str:
    if isinstance(input_data, dict):
        messages = input_data.get("messages", [])
        if messages:
            last = messages[-1]
            if isinstance(last, HumanMessage):
                if isinstance(last.content, str):
                    return last.content
                return str(last.content)
            return str(getattr(last, "content", ""))
    return str(input_data)


@dataclass
class FakeGraph:
    """Minimal graph interface compatible with MessageDispatcher expectations."""

    async def astream(
        self,
        input_data: dict[str, Any] | Any,
        config: Any = None,
        *,
        context: Any = None,
        stream_mode: list[str] | None = None,
        subgraphs: bool = True,
    ):
        del config, context, stream_mode, subgraphs
        prompt = _extract_prompt(input_data).lower()

        if "todo" in prompt:
            todo_call_id = "call-todo-1"
            yield (
                (),
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                id="fake-todo-ai-1",
                                content="Updating todo list.",
                                usage_metadata={
                                    "input_tokens": 40,
                                    "output_tokens": 8,
                                    "total_tokens": 48,
                                },
                                tool_calls=[
                                    {
                                        "name": "write_todos",
                                        "args": {
                                            "todos": [
                                                {
                                                    "content": "Review profile bottleneck",
                                                    "status": "in_progress",
                                                },
                                                {
                                                    "content": "Draft optimization report",
                                                    "status": "pending",
                                                },
                                            ]
                                        },
                                        "id": todo_call_id,
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        ]
                    }
                },
            )
            rendered = format_todos(
                [
                    {"content": "Review profile bottleneck", "status": "in_progress"},
                    {"content": "Draft optimization report", "status": "pending"},
                ],
                max_items=50,
                max_completed=50,
                show_completed_indicator=False,
            )
            tool_message = ToolMessage(
                content='[{"content":"Review profile bottleneck","status":"in_progress"}]',
                tool_call_id=todo_call_id,
                name="write_todos",
                response_metadata={
                    TOOL_TIMING_RESPONSE_METADATA_KEY: {"duration_seconds": 0.25},
                },
            )
            setattr(tool_message, "short_content", f"{TODO_PANEL_MARKER}{rendered}")
            yield ((), "updates", {"agent": {"messages": [tool_message]}})
            yield (
                (),
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                id="fake-todo-ai-2",
                                content="Todo list updated.",
                                usage_metadata={
                                    "input_tokens": 45,
                                    "output_tokens": 4,
                                    "total_tokens": 49,
                                },
                            )
                        ]
                    }
                },
            )
            return

        if "tool" in prompt:
            call_id = "call-tool-1"
            yield (
                (),
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                id="fake-tool-ai-1",
                                content="I will execute one tool call.",
                                usage_metadata={
                                    "input_tokens": 50,
                                    "output_tokens": 9,
                                    "total_tokens": 59,
                                },
                                tool_calls=[
                                    {
                                        "name": "run_command",
                                        "args": {"command": "echo fake-tool-output"},
                                        "id": call_id,
                                        "type": "tool_call",
                                    }
                                ],
                            )
                        ]
                    }
                },
            )
            yield (
                (),
                "updates",
                {
                    "agent": {
                        "messages": [
                            ToolMessage(
                                content="fake-tool-output",
                                tool_call_id=call_id,
                                name="run_command",
                                response_metadata={
                                    TOOL_TIMING_RESPONSE_METADATA_KEY: {"duration_seconds": 0.5},
                                },
                            )
                        ]
                    }
                },
            )
            yield (
                (),
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                id="fake-tool-ai-2",
                                content="Tool call finished.",
                                usage_metadata={
                                    "input_tokens": 55,
                                    "output_tokens": 5,
                                    "total_tokens": 60,
                                },
                            )
                        ]
                    }
                },
            )
            return

        chunk = AIMessageChunk(
            id="fake-hello-ai-1",
            content="Hello from msagent fake backend.",
            usage_metadata={
                "input_tokens": 20,
                "output_tokens": 6,
                "total_tokens": 26,
            },
        )
        yield ((), "messages", (chunk, {}))
        yield (
            (),
            "updates",
            {
                "agent": {
                    "messages": [
                        AIMessage(
                            id="fake-hello-ai-1",
                            content="Hello from msagent fake backend.",
                            usage_metadata={
                                "input_tokens": 20,
                                "output_tokens": 6,
                                "total_tokens": 26,
                            },
                        )
                    ]
                }
            },
        )

    async def ainvoke(
        self,
        input_data: dict[str, Any] | Any,
        config: Any = None,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        del config, context
        prompt = _extract_prompt(input_data).lower()
        if "tool" in prompt:
            call_id = "call-tool-1"
            return {
                "messages": [
                    AIMessage(
                        id="fake-tool-ai-1",
                        content="I will execute one tool call.",
                        usage_metadata={
                            "input_tokens": 50,
                            "output_tokens": 9,
                            "total_tokens": 59,
                        },
                        tool_calls=[
                            {
                                "name": "run_command",
                                "args": {"command": "echo fake-tool-output"},
                                "id": call_id,
                                "type": "tool_call",
                            }
                        ],
                    ),
                    ToolMessage(
                        content="fake-tool-output",
                        tool_call_id=call_id,
                        name="run_command",
                        response_metadata={
                            TOOL_TIMING_RESPONSE_METADATA_KEY: {"duration_seconds": 0.5},
                        },
                    ),
                    AIMessage(
                        id="fake-tool-ai-2",
                        content="Tool call finished.",
                        usage_metadata={
                            "input_tokens": 55,
                            "output_tokens": 5,
                            "total_tokens": 60,
                        },
                    ),
                ]
            }
        if "todo" in prompt:
            return {
                "messages": [
                    AIMessage(
                        id="fake-todo-ai-2",
                        content="Todo list updated.",
                        usage_metadata={
                            "input_tokens": 45,
                            "output_tokens": 4,
                            "total_tokens": 49,
                        },
                    )
                ]
            }
        return {
            "messages": [
                AIMessage(
                    id="fake-hello-ai-1",
                    content="Hello from msagent fake backend.",
                    usage_metadata={
                        "input_tokens": 20,
                        "output_tokens": 6,
                        "total_tokens": 26,
                    },
                )
            ]
        }
