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

"""CLI-specific context dataclass."""

import uuid
from pathlib import Path

from pydantic import BaseModel

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.bootstrap.timer import timer
from msagent.configs import ApprovalMode, ExecuteApprovalMode, LLMProvider
from msagent.audit import resolve_audit_log_enabled
from msagent.core.logging import get_logger

logger = get_logger(__name__)


class Context(BaseModel):
    """Runtime CLI context."""

    agent: str
    agent_description: str | None = None
    model: str
    model_display: str | None = None
    thread_id: str
    working_dir: Path
    state_dir: Path | None = None
    approval_mode: ApprovalMode = ApprovalMode.ACTIVE
    execute_approval_mode: ExecuteApprovalMode | None = None
    bash_mode: bool = False
    current_input_tokens: int | None = None
    current_output_tokens: int | None = None
    context_window: int | None = None
    recursion_limit: int
    tool_output_max_tokens: int | None = None
    stream_output: bool = True
    trace_jsonl: Path | None = None
    audit_log_enabled: bool = False

    @staticmethod
    def format_provider_label(llm_config: object) -> str | None:
        """Build a user-facing provider label from resolved LLM config."""
        provider = getattr(llm_config, "provider", None)
        if provider is None:
            return None

        provider_value = getattr(provider, "value", provider)
        provider_labels = {
            LLMProvider.GOOGLE.value: "Gemini",
        }
        return provider_labels.get(str(provider_value), str(provider_value))

    @staticmethod
    def format_model_display(model_alias: str, llm_config: object) -> str:
        """Build a user-facing model label from alias and resolved config."""
        resolved_model = getattr(llm_config, "model", None) or model_alias
        provider_label = Context.format_provider_label(llm_config)

        if provider_label:
            return f"{resolved_model} ({provider_label})"
        return resolved_model

    @classmethod
    async def create(
        cls,
        agent: str | None,
        model: str | None,
        approval_mode: ApprovalMode | None,
        working_dir: Path,
        execute_approval_mode: ExecuteApprovalMode | None = None,
        stream_output: bool = True,
        trace_jsonl: Path | None = None,
    ) -> "Context":
        """Create context and populate from agent config."""
        with timer("Load agent config"):
            agent_config = await initializer.load_agent_config(agent, working_dir)

        thread_id = str(uuid.uuid4())
        resolved_agent = agent_config.name
        workspace_model = None
        if model is None:
            workspace_model = await initializer.get_current_model(resolved_agent, working_dir)
        selected_model = model or workspace_model

        if selected_model:
            with timer("Load LLM config"):
                llm_config = await initializer.load_llm_config(selected_model, working_dir)
        else:
            llm_config = agent_config.llm

        tool_output_max_tokens = agent_config.tools.output_max_tokens if agent_config.tools else None

        resolved_model = selected_model or agent_config.llm.alias
        resolved_model_display = cls.format_model_display(resolved_model, llm_config)

        logger.info(
            "Agent: %s, Model alias: %s, Model: %s",
            resolved_agent,
            resolved_model,
            resolved_model_display,
        )
        logger.info(f"Thread ID: {thread_id}")

        return cls(
            agent=resolved_agent,
            agent_description=agent_config.description or None,
            model=resolved_model,
            model_display=resolved_model_display,
            thread_id=thread_id,
            working_dir=working_dir,
            state_dir=initializer.get_project_paths(working_dir).root,
            approval_mode=approval_mode or ApprovalMode.ACTIVE,
            execute_approval_mode=execute_approval_mode,
            context_window=llm_config.context_window,
            recursion_limit=agent_config.recursion_limit,
            tool_output_max_tokens=tool_output_max_tokens,
            stream_output=stream_output,
            trace_jsonl=trace_jsonl,
            audit_log_enabled=resolve_audit_log_enabled(agent_config),
        )

    def cycle_approval_mode(self) -> ApprovalMode:
        """Cycle to the next approval mode."""
        modes = list(ApprovalMode)
        current_index = modes.index(self.approval_mode)
        next_index = (current_index + 1) % len(modes)
        self.approval_mode = modes[next_index]
        return self.approval_mode

    def toggle_bash_mode(self) -> bool:
        """Toggle bash mode on/off."""
        self.bash_mode = not self.bash_mode
        return self.bash_mode
