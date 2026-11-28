# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Stream mapping between LangGraph graph runs and ACP session updates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from msagent.agents.context import AgentContext
from msagent.cli.acp.approval import AcpApprovalBackend
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.builders import MessageContentBuilder
from msagent.cli.dispatchers.messages import (
    MessageDispatcher,
    build_agent_context,
    resolve_prior_agent_prompt,
)
from msagent.cli.handlers.interrupts import InterruptHandler
from msagent.cli.ui.renderer import Renderer
from msagent.core.logging import get_logger

logger = get_logger(__name__)


class AcpTurnRunner:
    """Run one graph turn and forward progress updates to the ACP client."""

    def __init__(self, *, conn: Any, session_id: str, session: Any, graph: Any) -> None:
        self._conn = conn
        self._session_id = session_id
        self._session = session
        self._graph = graph
        self._interrupt_handler = InterruptHandler(
            session=session,
            backend=AcpApprovalBackend(
                conn=conn,
                session_id=session_id,
                resolve_tool_call_id=self._resolve_tool_call_id,
            ),
        )
        self._started_tool_calls: set[str] = set()
        self._tool_call_names: dict[str, str] = {}

    async def run(self, user_text: str) -> None:
        """Execute one user turn end to end."""
        context = self._session.context
        await initializer.refresh_cached_skills(agent=context.agent, working_dir=context.working_dir)

        message_builder = MessageContentBuilder(Path(context.working_dir))
        message_content, image_refs = message_builder.build(user_text)
        human_message = HumanMessage(
            content=message_content,
            short_content=user_text,
            additional_kwargs={"reference_mapping": image_refs},
        )

        agent_context = await self._build_agent_context()
        graph_config = RunnableConfig(
            configurable={"thread_id": context.thread_id},
            recursion_limit=context.recursion_limit,
        )

        run_id = str(uuid4())
        prior_prompt = await self._resolve_prior_agent_prompt(graph_config)
        self._session.subagent_audit.begin_run(run_id, user_message=user_text, prompt=prior_prompt)

        agent_context.retry_notice_handler = self._log_retry_notice
        try:
            await self._stream_turn({"messages": [human_message]}, graph_config, agent_context)
        finally:
            agent_context.retry_notice_handler = None

    async def _stream_turn(
        self,
        graph_input: dict[str, Any] | Command,
        config: RunnableConfig,
        agent_context: AgentContext,
    ) -> None:
        """Stream graph chunks until the turn completes, resuming through interrupts."""
        current_input: dict[str, Any] | Command = graph_input
        while True:
            interrupted = False
            async for chunk in self._graph.astream(
                current_input,
                config,
                context=agent_context,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                interrupts = MessageDispatcher.extract_interrupts(chunk)
                if interrupts:
                    resume_value = await self._interrupt_handler.handle(interrupts)
                    agent_context.approval_mode = self._session.context.approval_mode
                    if isinstance(resume_value, dict):
                        current_input = Command(resume=resume_value)
                    else:
                        current_input = Command(resume={interrupts[0].id: resume_value})
                    interrupted = True
                    break

                _namespace, mode, data = chunk
                if mode == "messages":
                    message_chunk, _metadata = data
                    if isinstance(message_chunk, AIMessageChunk):
                        await self._emit_message_chunk(message_chunk)
                elif mode == "updates":
                    await self._emit_tool_completions(data)

            if not interrupted:
                break

    async def _emit_message_chunk(self, message_chunk: AIMessageChunk) -> None:
        """Forward assistant text deltas and new tool calls to the client."""
        reasoning_delta = self._extract_reasoning_delta(message_chunk)
        if reasoning_delta:
            await self._send_session_update(AgentThoughtChunk(content=TextContentBlock(text=reasoning_delta)))

        text_delta = MessageDispatcher.extract_chunk_content(message_chunk)
        if text_delta:
            await self._send_session_update(AgentMessageChunk(content=TextContentBlock(text=text_delta)))

        for preview in MessageDispatcher.extract_tool_call_previews(message_chunk):
            if preview.call_id is None or preview.call_id in self._started_tool_calls:
                continue
            if preview.name in MessageDispatcher.HIDDEN_ACTIVITY_TOOLS:
                continue
            self._started_tool_calls.add(preview.call_id)
            self._tool_call_names[preview.call_id] = preview.name
            await self._send_session_update(
                ToolCallStart(
                    tool_call_id=preview.call_id,
                    title=preview.name,
                    name=preview.name,
                    kind="other",
                    status="pending",
                )
            )

    @staticmethod
    def _extract_reasoning_delta(message_chunk: AIMessageChunk) -> str:
        """Collect streamed reasoning/thinking text from a chunk."""
        parts: list[str] = []
        additional = getattr(message_chunk, "additional_kwargs", None) or {}
        reasoning_content = additional.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            parts.append(reasoning_content)

        metadata_thinking = Renderer.extract_thinking_from_metadata(cast(AIMessage, message_chunk))
        if isinstance(metadata_thinking, str) and metadata_thinking:
            parts.append(metadata_thinking)

        content = message_chunk.content
        if isinstance(content, list):
            _texts, thinking_parts = Renderer.extract_thinking_and_text_from_blocks(content)
            parts.extend(part for part in thinking_parts if part)
        return "".join(parts)

    async def _emit_tool_completions(self, data: Any) -> None:
        """Mark tool calls completed once their ToolMessage lands in state updates."""
        if not isinstance(data, dict):
            return
        for node_data in data.values():
            if not isinstance(node_data, dict):
                continue
            messages_value = MessageDispatcher.unwrap_overwrite(node_data.get("messages"))
            if isinstance(messages_value, tuple):
                messages_value = list(messages_value)
            if not isinstance(messages_value, list):
                continue
            for message in messages_value:
                if not isinstance(message, ToolMessage) or not message.tool_call_id:
                    continue
                if message.tool_call_id not in self._started_tool_calls:
                    continue
                self._tool_call_names.pop(message.tool_call_id, None)
                await self._send_session_update(
                    ToolCallProgress(
                        tool_call_id=message.tool_call_id,
                        status="failed" if message.status == "error" else "completed",
                    )
                )

    async def abort_pending_tool_calls(self) -> None:
        """Report tool calls still pending when a turn is cancelled as failed."""
        pending = list(self._started_tool_calls)
        self._started_tool_calls.clear()
        self._tool_call_names.clear()
        for tool_call_id in pending:
            await self._send_session_update(ToolCallProgress(tool_call_id=tool_call_id, status="failed"))

    def _resolve_tool_call_id(self, tool_name: str) -> str | None:
        """Return the pending tool call id for an approval prompt, if one is known."""
        for call_id, name in self._tool_call_names.items():
            if name == tool_name:
                return call_id
        return None

    async def _send_session_update(self, update: Any) -> None:
        await self._conn.session_update(self._session_id, update=update)

    async def _build_agent_context(self) -> AgentContext:
        """Build runtime context injected into prompt templates and middleware."""
        return await build_agent_context(self._session.context)

    async def _resolve_prior_agent_prompt(self, graph_config: RunnableConfig) -> str | None:
        """Read the latest assistant message from checkpointed graph state."""
        return await resolve_prior_agent_prompt(self._graph, graph_config)

    @staticmethod
    def _log_retry_notice(notice: Any) -> None:
        logger.warning(
            "Model retry scheduled: attempt=%s, max_retries=%s",
            getattr(notice, "attempt", "?"),
            getattr(notice, "max_retries", "?"),
        )


async def replay_session_history(*, conn: Any, session_id: str, messages: list[Any]) -> None:
    """Replay checkpointed messages so a reloaded session shows its history."""
    tool_status: dict[str, str] = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id:
            tool_status[message.tool_call_id] = "failed" if message.status == "error" else "completed"

    for index, message in enumerate(messages):
        message_id = getattr(message, "id", None) or f"replay-{index}"
        if isinstance(message, HumanMessage):
            text = _message_text(message)
            if text:
                await conn.session_update(
                    session_id,
                    update=UserMessageChunk(content=TextContentBlock(text=text), message_id=message_id),
                )
        elif isinstance(message, AIMessage):
            text = _message_text(message)
            if text:
                await conn.session_update(
                    session_id,
                    update=AgentMessageChunk(content=TextContentBlock(text=text), message_id=message_id),
                )
            for preview in MessageDispatcher.extract_tool_call_previews(message):
                if preview.call_id is None or preview.name in MessageDispatcher.HIDDEN_ACTIVITY_TOOLS:
                    continue
                await conn.session_update(
                    session_id,
                    update=ToolCallStart(
                        tool_call_id=preview.call_id,
                        title=preview.name,
                        name=preview.name,
                        kind="other",
                        status=cast(Any, tool_status.get(preview.call_id, "failed")),
                    ),
                )


def _message_text(message: Any) -> str:
    """Extract plain text blocks from a stored message."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    texts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(str(block.get("text", "")))
    return "".join(texts)
