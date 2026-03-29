"""Local fake backend graph used by CLI entrypoint E2E tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from msagent.tools.internal.todo import TODO_PANEL_MARKER, format_todos


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
                                content="Updating todo list.",
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
            )
            setattr(tool_message, "short_content", f"{TODO_PANEL_MARKER}{rendered}")
            yield ((), "updates", {"agent": {"messages": [tool_message]}})
            yield (
                (),
                "updates",
                {"agent": {"messages": [AIMessage(content="Todo list updated.")]}}
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
                                content="I will execute one tool call.",
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
                            )
                        ]
                    }
                },
            )
            yield (
                (),
                "updates",
                {"agent": {"messages": [AIMessage(content="Tool call finished.")]}}
            )
            return

        chunk = AIMessageChunk(content="Hello from msagent fake backend.")
        yield ((), "messages", (chunk, {}))
        yield (
            (),
            "updates",
            {"agent": {"messages": [AIMessage(content="Hello from msagent fake backend.")]}}
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
            return {"messages": [AIMessage(content="Tool call finished.")]}
        if "todo" in prompt:
            return {"messages": [AIMessage(content="Todo list updated.")]}
        return {"messages": [AIMessage(content="Hello from msagent fake backend.")]}

