"""Compression handling for chat sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import cast

from langchain_core.runnables import RunnableConfig

from msagent.agents.context import AgentContext
from msagent.agents.local_context import (
    build_local_environment_context,
    ensure_local_context_prompt,
)
from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console, theme
from msagent.configs import CompressionConfig
from msagent.core.constants import OS_VERSION, PLATFORM
from msagent.core.logging import get_logger
from msagent.utils.compression import calculate_message_tokens
from msagent.utils.cost import format_tokens
from msagent.utils.offload import perform_conversation_offload
from msagent.utils.render import render_templates

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

            compression_config = agent_config.compression or CompressionConfig()
            prompt_str = compression_config.prompt
            prepared_prompt = (
                ensure_local_context_prompt(cast(str, prompt_str))
                if prompt_str
                else None
            )
            if self.session.graph is None:
                console.print_error("Conversation graph is not ready for compression")
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

            compression_llm_config = compression_config.llm or agent_config.llm
            compression_llm = initializer.llm_factory.create(compression_llm_config)
            now = datetime.now(timezone.utc).astimezone()
            user_memory = await initializer.load_user_memory(ctx.working_dir)
            agent_context = AgentContext(
                approval_mode=ctx.approval_mode,
                working_dir=ctx.working_dir,
                platform=PLATFORM,
                os_version=OS_VERSION,
                current_date_time_zoned=now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                local_environment_context=build_local_environment_context(
                    ctx.working_dir,
                    now=now,
                ),
                mcp_servers=(
                    ", ".join(initializer.cached_mcp_server_names)
                    if initializer.cached_mcp_server_names
                    else "None"
                ),
                user_memory=user_memory,
                tool_output_max_tokens=ctx.tool_output_max_tokens,
            )
            rendered_prompt = (
                str(render_templates(prepared_prompt, agent_context.template_vars))
                if prepared_prompt
                else None
            )
            backend = getattr(self.session.graph, "_agent_backend", None)
            if backend is None:
                console.print_error("Conversation backend is unavailable for compression")
                console.print("")
                return

            original_count = len(messages)
            original_tokens = calculate_message_tokens(messages, compression_llm)

            with console.console.status(
                f"[{theme.spinner_color}]Offloading {original_count} messages ({format_tokens(original_tokens)} tokens)..."
            ):
                offload_result = await perform_conversation_offload(
                    messages=messages,
                    prior_event=state_values.get("_summarization_event"),
                    thread_id=ctx.thread_id,
                    model=compression_llm,
                    backend=backend,
                    keep_messages=compression_config.messages_to_keep,
                    summary_prompt=rendered_prompt,
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
                console.print(f"[muted]Conversation history saved to {file_path}[/muted]")
            if offload_result.offload_warning:
                console.print_warning(offload_result.offload_warning)
            console.print("")

        except Exception as e:
            console.print_error(f"Error compressing conversation: {e}")
            console.print("")
            logger.debug("Compression error", exc_info=True)
