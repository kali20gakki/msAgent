"""Conversation offload helpers for context compaction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from deepagents.middleware.summarization import (
    SummarizationEvent,
    SummarizationMiddleware,
)
from langchain_core.messages import AnyMessage, HumanMessage, get_buffer_string

from msagent.utils.compression import calculate_message_tokens

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol
    from langchain_core.language_models import BaseChatModel

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

    timestamp = datetime.now(UTC).isoformat()
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
    messages: list[AnyMessage],
    prior_event: SummarizationEvent | None,
    thread_id: str,
    model: BaseChatModel,
    backend: BackendProtocol,
    keep_messages: int = 0,
    summary_prompt: str | None = None,
) -> ConversationOffloadResult | None:
    """Summarize old messages and offload the originals to backend storage."""
    effective_keep_messages = max(keep_messages, 0)
    middleware = SummarizationMiddleware(
        model=model,
        backend=backend,
        # deepagents SummarizationMiddleware requires keep > 0, so when users
        # configure 0 we instantiate with a safe minimum and compute the final
        # partition ourselves below.
        keep=("messages", max(effective_keep_messages, 1)),
        trim_tokens_to_summarize=None,
        summary_prompt=summary_prompt or "Summarize the conversation so far.",
    )

    effective_messages = middleware._apply_event_to_messages(messages, prior_event)
    if effective_keep_messages == 0:
        cutoff = len(effective_messages)
    else:
        cutoff = middleware._determine_cutoff_index(effective_messages)
    if cutoff <= 0:
        return None

    to_summarize = effective_messages[:cutoff]
    to_keep = effective_messages[cutoff:]
    if not to_summarize:
        return None

    tokens_summarized = calculate_message_tokens(to_summarize, model)
    tokens_kept = calculate_message_tokens(to_keep, model)
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
    tokens_summary = calculate_message_tokens([summary_message], model)
    tokens_after = tokens_summary + tokens_kept
    pct_decrease = (
        round((tokens_before - tokens_after) / tokens_before * 100)
        if tokens_before > 0
        else 0
    )

    summary_content = (
        summary_message.content
        if isinstance(summary_message.content, str)
        else str(summary_message.content)
    )
    summary_message.content = summary_content + (
        "\n\n"
        f"Offloaded {len(to_summarize)} messages and kept {len(to_keep)} recent "
        f"messages in active context. Approximate token usage: "
        f"{tokens_before} -> {tokens_after} ({pct_decrease}% reduction)."
    )

    state_cutoff = middleware._compute_state_cutoff(prior_event, cutoff)
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
