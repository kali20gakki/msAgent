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

"""Agent handling for chat sessions."""

from __future__ import annotations

import uuid

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import FormattedTextControl

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.core.context import Context
from msagent.audit import resolve_audit_log_enabled
from msagent.cli.theme import console, theme
from msagent.cli.ui.shared import SelectorState, create_selector_application
from msagent.configs import AgentConfig
from msagent.core.logging import get_logger
from msagent.core.settings import settings

logger = get_logger(__name__)


class AgentHandler:
    """Handles agent operations like switching and selection."""

    def __init__(self, session) -> None:
        """Initialize with reference to CLI session."""
        self.session = session

    async def handle(self) -> None:
        """Show interactive agent selector and switch to selected agent."""
        try:
            config_data = await initializer.load_agents_config(self.session.context.working_dir)
            configured_agents = [agent for agent in config_data.agents if isinstance(agent, AgentConfig)]
            current_agent_name = self.session.context.agent

            if not configured_agents:
                console.print_error("No agents configured")
                console.print("")
                return

            selected_agent_name = await self._get_agent_selection(configured_agents, current_agent_name)

            if selected_agent_name and selected_agent_name != current_agent_name:
                # Load the selected agent's config to get its model
                selected_agent_config = await initializer.load_agent_config(
                    selected_agent_name, self.session.context.working_dir
                )
                selected_model = await initializer.get_current_model(
                    selected_agent_name,
                    self.session.context.working_dir,
                )
                selected_llm = (
                    await initializer.load_llm_config(
                        selected_model,
                        self.session.context.working_dir,
                    )
                    if selected_model
                    else selected_agent_config.llm
                )
                selected_model = selected_model or selected_agent_config.llm.alias

                await initializer.set_current_agent(
                    selected_agent_name,
                    self.session.context.working_dir,
                )

                # Update context with both agent and its configured model
                self.session.update_context(
                    agent=selected_agent_name,
                    agent_description=selected_agent_config.description or None,
                    model=selected_model,
                    model_display=Context.format_model_display(
                        selected_model,
                        selected_llm,
                    ),
                    context_window=getattr(selected_llm, "context_window", None),
                    audit_log_enabled=resolve_audit_log_enabled(selected_agent_config),
                    # Agent prompts must not inherit another agent's checkpointed conversation.
                    thread_id=str(uuid.uuid4()),
                    current_input_tokens=None,
                    current_output_tokens=None,
                )
                logger.info(f"Switched to Agent: {selected_agent_name}, Model: {selected_model}")
                console.print_warning(f"Switched to Agent: {selected_agent_name}. A new conversation has started.")
                console.print("")

        except Exception as e:
            console.print_error(f"Error switching agents: {e}")
            console.print("")
            logger.debug("Agent switch error", exc_info=True)

    async def _get_agent_selection(self, agents: list[AgentConfig], current_agent_name: str) -> str:
        """Get agent selection from user using interactive list.

        Args:
            agents: List of agent configuration objects
            current_agent_name: Currently active agent name

        Returns:
            Selected agent name or empty string if canceled
        """
        if not agents:
            return ""

        initial_index = next(
            (index for index, agent in enumerate(agents) if agent.name == current_agent_name),
            0,
        )
        state = SelectorState(index=initial_index)

        # Create text control with formatted text
        text_control = FormattedTextControl(
            text=lambda: self._format_agent_list(agents, state.index, current_agent_name),
            focusable=True,
            show_cursor=False,
        )

        # Create key bindings
        kb = KeyBindings()

        @kb.add(Keys.Up)
        def _(event):
            state.move_cyclic(-1, size=len(agents))

        @kb.add(Keys.Down)
        def _(event):
            state.move_cyclic(1, size=len(agents))

        selected = [False]

        @kb.add(Keys.Enter)
        def _(event):
            selected[0] = True
            event.app.exit()

        @kb.add(Keys.Escape)
        @kb.add(Keys.ControlC)
        def _(event):
            event.app.exit()

        # Create application
        context = self.session.context
        app = create_selector_application(
            context=context,
            text_control=text_control,
            key_bindings=kb,
        )

        selected_agent_name = ""

        try:
            await app.run_async()

            if selected[0]:
                selected_agent_name = agents[state.index].name

        except (KeyboardInterrupt, EOFError):
            pass

        return selected_agent_name

    @staticmethod
    def _format_agent_list(agents: list[AgentConfig], selected_index: int, current_agent_name: str):
        """Format the agent list with highlighting.

        Args:
            agents: List of agent configuration objects
            selected_index: Index of currently selected agent
            current_agent_name: Name of the active agent

        Returns:
            FormattedText with styled lines
        """
        prompt_symbol = settings.cli.prompt_style.strip()
        lines = []
        for i, agent in enumerate(agents):
            agent_name = agent.name
            tags: list[str] = []
            if agent.name == current_agent_name:
                tags.append("current")
            tags_text = f" [{' | '.join(tags)}]" if tags else ""

            display_text = f"{agent_name}{tags_text}"

            if i == selected_index:
                # Use direct color code for selected line
                lines.append((f"{theme.selection_color}", f"{prompt_symbol} {display_text}"))
            else:
                lines.append(("", f"  {display_text}"))

            if agent.description:
                lines.append(("", "\n"))
                lines.append(("dim", f"    {agent.description}"))

            if i < len(agents) - 1:
                lines.append(("", "\n"))
                lines.append(("", "\n"))

        return FormattedText(lines)
