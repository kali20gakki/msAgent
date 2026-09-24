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

"""Compression handling for chat sessions."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console, theme
from msagent.core.logging import get_logger
from msagent.utils.cost import format_tokens
from msagent.utils.offload import perform_conversation_offload

logger = get_logger(__name__)


class CompressionHandler:
    """Handles conversation history compression."""

    def __init__(self, session) -> None:
        """Initialize with reference to CLI session."""
        self.session = session

    async def handle(self) -> None:
        """Compress current conversation history inside the current thread."""
        try:
            ctx = self.session.context
            config_data = await initializer.load_agents_config(ctx.working_dir)
            agent_config = config_data.get_agent_config(ctx.agent)

            if not agent_config:
                console.print_error(f"Agent '{ctx.agent}' not found")
                console.print("")
                return

            if self.session.graph is None:
                console.print_error("Conversation graph is not ready for compression")
                console.print("")
                return

            middleware = getattr(self.session.graph, "_compression_middleware", None)
            if middleware is None:
                console.print_error("Conversation compression is not configured")
                console.print("")
                return

            backend = getattr(self.session.graph, "_agent_backend", None)
            if backend is None:
                console.print_error("Conversation backend is unavailable for compression")
                console.print("")
                return

            config = RunnableConfig(configurable={"thread_id": ctx.thread_id})
            snapshot = await self.session.graph.aget_state(config)
            state_values = snapshot.values if snapshot is not None else {}
            messages = list(state_values.get("messages", []) or [])

            if not messages:
                console.print_error("No conversation history found to compress")
                console.print("")
                return

            original_count = len(messages)
            original_tokens = middleware.token_counter(messages)

            with console.console.status(
                f"[{theme.spinner_color}]Offloading {original_count} messages ({format_tokens(original_tokens)} tokens)..."
            ):
                offload_result = await perform_conversation_offload(
                    middleware=middleware,
                    messages=messages,
                    prior_event=state_values.get("_summarization_event"),
                    thread_id=ctx.thread_id,
                    backend=backend,
                )

            if offload_result is None:
                console.print_warning("Conversation is already within the configured retention window")
                console.print("")
                return

            await self.session.graph.aupdate_state(
                config,
                {"_summarization_event": offload_result.new_event},
            )

            self.session.update_context(
                current_input_tokens=offload_result.tokens_after,
                current_output_tokens=0,
            )

            logger.info(
                "Conversation offloaded for thread %s: %d messages -> %d kept",
                ctx.thread_id,
                offload_result.messages_offloaded,
                offload_result.messages_kept,
            )

            console.print_success(
                "Context compacted in-place: "
                f"{offload_result.messages_offloaded} messages offloaded, "
                f"{offload_result.messages_kept} kept, "
                f"{format_tokens(offload_result.tokens_before)} -> "
                f"{format_tokens(offload_result.tokens_after)} "
                f"({offload_result.pct_decrease}% reduction)."
            )
            file_path = offload_result.new_event.get("file_path")
            if file_path:
                conversation_file = (
                    initializer.get_project_paths(ctx.working_dir).conversation_history_dir / f"{ctx.thread_id}.md"
                )
                console.print(f"[muted]Conversation history saved to {conversation_file}[/muted]")
            if offload_result.offload_warning:
                console.print_warning(offload_result.offload_warning)
            console.print("")

        except Exception as e:
            console.print_error(f"Error compressing conversation: {e}")
            console.print("")
            logger.debug("Compression error", exc_info=True)
