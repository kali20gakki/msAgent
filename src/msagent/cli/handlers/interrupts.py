"""HIL interrupt management for LangGraph execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import CompleteStyle
from typing_extensions import NotRequired, TypedDict

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console
from msagent.cli.ui.shared import (
    build_agent_prompt,
    create_bottom_toolbar,
    create_prompt_style,
)
from msagent.configs import ToolApprovalConfig
from msagent.core.logging import get_logger
from msagent.middlewares.approval import InterruptPayload

if TYPE_CHECKING:
    from langgraph.types import Interrupt

logger = get_logger(__name__)


class HITLActionRequest(TypedDict):
    """Action request payload emitted by HumanInTheLoopMiddleware."""

    name: str
    args: dict[str, Any]
    description: NotRequired[str]


class HITLReviewConfig(TypedDict):
    """Review policy payload emitted by HumanInTheLoopMiddleware."""

    action_name: str
    allowed_decisions: list[str]


class HITLRequest(TypedDict):
    """Interrupt payload emitted by HumanInTheLoopMiddleware."""

    action_requests: list[HITLActionRequest]
    review_configs: list[HITLReviewConfig]


class InterruptHandler:
    """Handles LangGraph interrupts and collects user input for resume."""

    def __init__(self, session) -> None:
        """Initialize with reference to CLI session."""
        self.session = session

    async def handle(self, interrupt_data: list[Interrupt]) -> Any:
        """
        Handle a LangGraph interrupts and collect user input.

        Args:
            interrupt_data: List of Interrupt objects from LangGraph

        Returns:
            Resume value to pass back to LangGraph:
            - For single interrupt: returns the resume value directly
            - For multiple interrupts: returns dict mapping interrupt IDs to resume values
        """
        try:
            if not interrupt_data:
                logger.warning("Empty interrupt data received")
                return None

            # Handle single interrupt - return value directly
            if len(interrupt_data) == 1:
                return await self._get_choice(interrupt_data[0])

            # Handle multiple interrupts - return dict mapping IDs to values
            resume_dict = {}
            for interrupt in interrupt_data:
                choice = await self._get_choice(interrupt)
                if choice is not None:
                    resume_dict[interrupt.id] = choice

            return resume_dict if resume_dict else None

        except Exception as e:
            console.print_error(f"Error handling interrupt: {e}")
            console.print("")
            return None

    async def _get_choice(self, interrupt: Interrupt) -> Any:
        """Choice selector with tab completion and Enter key support."""
        value = interrupt.value

        if (
            isinstance(value, dict)
            and "action_requests" in value
            and "review_configs" in value
        ):
            return await self._get_hitl_decisions(cast(HITLRequest, value))

        if isinstance(value, dict) and "question" in value and "options" in value:
            return await self._get_legacy_choice(value)

        logger.warning("Unknown interrupt payload shape: %s", type(value).__name__)
        return None

    async def _get_hitl_decisions(self, value: HITLRequest) -> dict[str, Any] | None:
        """Handle deepagents HITL action/review payload."""
        actions = value.get("action_requests") or []
        review_configs = value.get("review_configs") or []
        if not actions:
            return None

        approval_config = self._load_approval_config()
        should_persist = False
        review_by_action = {
            str(config.get("action_name", "")): config for config in review_configs
        }
        decisions: list[dict[str, Any]] = []
        for action in actions:
            tool_name = str(action.get("name", "") or "")
            tool_args = action.get("args")
            if not isinstance(tool_args, dict):
                tool_args = {}

            config: HITLReviewConfig = review_by_action.get(
                tool_name,
                {
                    "action_name": tool_name,
                    "allowed_decisions": ["approve", "reject"],
                },
            )
            allowed = list(config.get("allowed_decisions") or ["approve", "reject"])
            options = [opt for opt in allowed if opt in {"approve", "reject"}]
            if not options:
                options = ["approve", "reject"]

            policy = approval_config.resolve_decision(tool_name, tool_args)
            if policy == "always_approve":
                decisions.append({"type": "approve"})
                continue
            if policy == "always_reject":
                decisions.append(
                    {
                        "type": "reject",
                        "message": "Rejected by local approval policy.",
                    }
                )
                continue

            if "approve" in options:
                options.append("always_approve")
            if "reject" in options:
                options.append("always_reject")
            options = list(dict.fromkeys(options))

            selected = await self._prompt_hitl_decision(
                tool_name=tool_name,
                tool_args=tool_args,
                description=action.get("description"),
                options=options,
            )
            if selected is None:
                return None

            if selected == "always_approve":
                approval_config.prepend_decision_rule(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    decision="always_approve",
                )
                should_persist = True
                decisions.append({"type": "approve"})
                continue

            if selected == "always_reject":
                approval_config.prepend_decision_rule(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    decision="always_reject",
                )
                should_persist = True
                decisions.append(
                    {
                        "type": "reject",
                        "message": "Rejected by local approval policy.",
                    }
                )
                continue

            decisions.append(self._selection_to_decision(selected))

        if should_persist:
            self._save_approval_config(approval_config)

        return {"decisions": decisions}

    async def _prompt_hitl_decision(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        description: str | None,
        options: list[str],
    ) -> str | None:
        """Prompt user for one HITL decision."""
        args_text = json.dumps(tool_args, ensure_ascii=False)
        question_parts = []
        if description:
            question_parts.append(str(description))
        else:
            question_parts.append("Tool execution requires approval.")
        question_parts.append(f"Tool: {tool_name}")
        question_parts.append(f"Args: {args_text}")
        question = "\n".join(question_parts)

        return await self._prompt_from_options(question=question, options=options)

    @staticmethod
    def _selection_to_decision(selected: str) -> dict[str, Any]:
        """Convert interactive selection into HITL decision payload."""
        if selected == "approve":
            return {"type": "approve"}
        if selected == "reject":
            return {"type": "reject"}
        return {"type": "approve"}

    def _load_approval_config(self) -> ToolApprovalConfig:
        """Load approval config for the current working directory."""
        try:
            working_dir = Path(self.session.context.working_dir)
            registry = initializer.get_registry(working_dir)
            return registry.load_approval(force_reload=True)
        except Exception:
            logger.debug(
                "Failed to load approval config; using defaults", exc_info=True
            )
            return ToolApprovalConfig()

    def _save_approval_config(self, config: ToolApprovalConfig) -> None:
        """Persist approval config for the current working directory."""
        try:
            working_dir = Path(self.session.context.working_dir)
            registry = initializer.get_registry(working_dir)
            registry.save_approval(config)
        except Exception:
            logger.debug("Failed to save approval config", exc_info=True)

    async def _get_legacy_choice(
        self, value: InterruptPayload | dict[str, Any]
    ) -> str | None:
        """Handle legacy question/options style interrupt payload."""
        if isinstance(value, dict):
            question = str(value.get("question", "Approval required"))
            options = list(value.get("options") or [])
        else:
            question = value.question
            options = value.options
        return await self._prompt_from_options(question=question, options=options)

    async def _prompt_from_options(
        self,
        *,
        question: str,
        options: list[str],
    ) -> str | None:
        """Prompt a choice from options with tab completion and Enter key support."""
        # Measure actual rendered lines by capturing output
        with console.capture() as capture:
            console.print(f"[accent]{question}[/accent]")
        rendered_text = capture.get()
        # Count actual newlines in the rendered output (not stripping)
        # This gives us the exact number of line breaks
        lines_to_clear: int = rendered_text.count("\n")

        # Now print for real
        console.print(f"[accent]{question}[/accent]")

        # Get context and create shared UI components
        context = self.session.context

        # Create separate prompt session with shared styling and mode cycling
        style = create_prompt_style(context, bash_mode=False)

        # Create key bindings for mode cycling
        kb = KeyBindings()

        @kb.add(Keys.BackTab)
        def _(event):
            """Shift-Tab: Cycle approval mode."""
            if self.session.prompt.mode_change_callback:
                self.session.prompt.mode_change_callback()
                # Refresh style after mode change
                interrupt_session.style = create_prompt_style(context, bash_mode=False)
                event.app.invalidate()

        interrupt_session: PromptSession[str] = PromptSession(
            completer=WordCompleter(options, ignore_case=True),
            complete_style=CompleteStyle.COLUMN,
            complete_while_typing=False,
            style=style,
            key_bindings=kb,
            bottom_toolbar=lambda: create_bottom_toolbar(
                context,
                context.working_dir,
                bash_mode=False,
            ),
        )

        try:
            while True:
                try:

                    def pre_run():
                        interrupt_session.default_buffer.start_completion(
                            select_first=False
                        )

                    result = await interrupt_session.prompt_async(
                        [
                            ("class:prompt", build_agent_prompt(context)),
                        ],
                        pre_run=pre_run,
                    )

                    if not result.strip():
                        console.print_error("Please make a choice")
                        lines_to_clear += 2  # prompt + warning
                        continue

                    # Validate the result
                    result_lower = result.strip().lower()

                    # Check if matches option name (case-insensitive)
                    matched_option = None
                    for option in options:
                        if option.lower() == result_lower:
                            matched_option = option
                            break

                    if matched_option:
                        # Clear all interrupt-related lines
                        for __ in range(lines_to_clear + 1):  # +1 for the final prompt
                            sys.stdout.write("\033[F")
                            sys.stdout.write("\033[K")
                        sys.stdout.flush()
                        return matched_option

                    # Check partial matches
                    matches = [o for o in options if o.lower().startswith(result_lower)]
                    if len(matches) == 1:
                        # Clear all interrupt-related lines
                        for __ in range(lines_to_clear + 1):  # +1 for the final prompt
                            sys.stdout.write("\033[F")
                            sys.stdout.write("\033[K")
                        sys.stdout.flush()
                        return matches[0]
                    elif len(matches) > 1:
                        console.print_error(
                            f"Ambiguous choice. Options: {', '.join(matches)}"
                        )
                        lines_to_clear += 2  # prompt + warning
                        continue

                    console.print_error(f"Invalid choice '{result}'. Please try again.")
                    lines_to_clear += 2  # prompt + warning

                except KeyboardInterrupt:
                    # Clear all interrupt-related lines including the current prompt
                    for __ in range(lines_to_clear + 1):  # +1 for current prompt
                        sys.stdout.write("\033[F")
                        sys.stdout.write("\033[K")
                    sys.stdout.flush()
                    return None
                except EOFError:
                    # Clear all interrupt-related lines including the current prompt
                    for __ in range(lines_to_clear + 1):  # +1 for current prompt
                        sys.stdout.write("\033[F")
                        sys.stdout.write("\033[K")
                    sys.stdout.flush()
                    return None
        except Exception:
            logger.debug("Interrupt choice failed", exc_info=True)
            return None
