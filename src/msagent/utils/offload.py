#!/usr/bin/python3
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

"""Conversation offload helpers for context compaction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from deepagents.middleware.summarization import (
    SummarizationEvent,
    SummarizationMiddleware,
)
from langchain_core.messages import AnyMessage, HumanMessage, get_buffer_string
from msagent.agents.local_context import build_local_environment_context, ensure_local_context_prompt
from msagent.core.constants import OS_VERSION, PLATFORM
from msagent.utils.render import render_templates

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationOffloadResult:
    """Details about a completed conversation offload."""

    new_event: SummarizationEvent
    messages_offloaded: int
    messages_kept: int
    tokens_before: int
    tokens_after: int
    pct_decrease: int
    offload_warning: str | None


def _event_cutoff(event: SummarizationEvent | None) -> int:
    """Extract the prior cutoff index from a summarization event, defaulting to 0.

    ``cutoff_index`` is declared as an ``int``, but malformed persisted state can
    hold a ``bool`` (which subclasses ``int``). A bare ``isinstance(value, int)``
    would silently treat ``True`` as ``1`` and shift the degeneracy guard off by
    one, so we reject ``bool`` first and return ``0`` for any non-``int`` value.
    """
    if event is None:
        return 0
    cutoff = event.get("cutoff_index")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int):
        return 0
    return cutoff


def render_compression_summary_prompt(
    prompt: str | list[str] | None,
    working_dir: Path,
) -> str | None:
    """Render the compression prompt with environment variables and message placeholder."""
    if not prompt:
        return None
    prepared = ensure_local_context_prompt(cast(str, prompt))
    now = datetime.now(timezone.utc).astimezone()
    # ``render_templates`` is all-or-nothing: an unknown placeholder makes
    # ``str.format`` fall back to the raw template. Pass the conversation
    # placeholder through as a literal so the local-environment placeholders
    # are rendered while ``{conversation}`` survives for the summarization
    # middleware, which substitutes ``{messages}`` with the real transcript.
    context = {
        "working_dir": str(working_dir),
        "platform": PLATFORM,
        "os_version": OS_VERSION,
        "current_date_time_zoned": now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "local_environment_context": build_local_environment_context(working_dir, now=now),
        "conversation": "{conversation}",
    }
    rendered = cast(str, render_templates(prepared, context))
    # The prompt templates reference ``{conversation}``, while the summarization
    # middleware substitutes ``{messages}``. Rewrite the placeholder here so the
    # compiled prompt stays aligned with what DeepAgents/LangChain expects.
    return rendered.replace("{conversation}", "{messages}")


async def offload_messages_to_backend(
    messages: list[AnyMessage],
    middleware: SummarizationMiddleware,
    *,
    thread_id: str,
    backend: BackendProtocol,
) -> str | None:
    """Append offloaded messages to backend storage for later retrieval."""
    path = f"/conversation_history/{thread_id}.md"

    filtered = middleware._filter_summary_messages(messages)
    if not filtered:
        return ""

    timestamp = datetime.now(timezone.utc).isoformat()
    section = f"## Offloaded at {timestamp}\n\n{get_buffer_string(filtered)}\n\n"

    existing_content = ""
    try:
        responses = await backend.adownload_files([path])
        response = responses[0] if responses else None
        if response and response.content is not None and response.error is None:
            existing_content = response.content.decode("utf-8")
    except Exception as exc:
        logger.warning(
            "Failed to read existing conversation history from %s: %s",
            path,
            exc,
            exc_info=True,
        )
        return None

    combined = existing_content + section

    try:
        result = (
            await backend.aedit(path, existing_content, combined)
            if existing_content
            else await backend.awrite(path, combined)
        )
    except Exception as exc:
        logger.warning(
            "Failed to write conversation history to %s: %s",
            path,
            exc,
            exc_info=True,
        )
        return None

    if result is None or getattr(result, "error", None):
        logger.warning(
            "Backend refused conversation history write to %s: %s",
            path,
            getattr(result, "error", "backend returned None"),
        )
        return None

    return path


async def perform_conversation_offload(
    *,
    middleware: SummarizationMiddleware,
    messages: list[AnyMessage],
    prior_event: SummarizationEvent | None,
    thread_id: str,
    backend: BackendProtocol,
) -> ConversationOffloadResult | None:
    """Summarize old messages and offload the originals to backend storage.

    Reuses the caller-provided resident summarization middleware so manual and
    automatic compaction share the same model, prompt, retention, and backend
    configuration.
    """
    effective_messages = middleware._apply_event_to_messages(messages, prior_event)

    cutoff = middleware._determine_cutoff_index(effective_messages)
    if cutoff <= 0:
        return None

    # Degenerate chained compaction guard: when the new absolute cutoff does not
    # advance past the prior summary, the only content left to summarize is the
    # previous summary itself. Skipping avoids progressively losing context.
    state_cutoff = middleware._compute_state_cutoff(prior_event, cutoff)
    if state_cutoff <= _event_cutoff(prior_event):
        return None

    to_summarize, to_keep = middleware._partition_messages(effective_messages, cutoff)
    if not to_summarize:
        return None

    tokens_summarized = middleware.token_counter(to_summarize)
    tokens_kept = middleware.token_counter(to_keep)
    tokens_before = tokens_summarized + tokens_kept

    summary = await middleware._acreate_summary(to_summarize)
    backend_path = await offload_messages_to_backend(
        to_summarize,
        middleware,
        thread_id=thread_id,
        backend=backend,
    )

    offload_warning: str | None = None
    if backend_path is None:
        offload_warning = (
            "Conversation history could not be saved to backend storage. "
            "Older messages were summarized but are not recoverable from "
            "conversation history."
        )

    file_path = backend_path or None
    summary_message = middleware._build_new_messages_with_path(summary, file_path)[0]
    tokens_summary = middleware.token_counter([summary_message])
    tokens_after = tokens_summary + tokens_kept
    pct_decrease = round((tokens_before - tokens_after) / tokens_before * 100) if tokens_before > 0 else 0

    summary_content = (
        summary_message.content if isinstance(summary_message.content, str) else str(summary_message.content)
    )
    summary_message.content = summary_content + (
        "\n\n"
        f"Offloaded {len(to_summarize)} messages and kept {len(to_keep)} recent "
        f"messages in active context. Approximate token usage: "
        f"{tokens_before} -> {tokens_after} ({pct_decrease}% reduction)."
    )

    new_event: SummarizationEvent = {
        "cutoff_index": state_cutoff,
        "summary_message": HumanMessage(
            content=summary_message.content,
            additional_kwargs=summary_message.additional_kwargs,
            name=getattr(summary_message, "name", None),
        ),
        "file_path": file_path,
    }

    return ConversationOffloadResult(
        new_event=new_event,
        messages_offloaded=len(to_summarize),
        messages_kept=len(to_keep),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        pct_decrease=pct_decrease,
        offload_warning=offload_warning,
    )
