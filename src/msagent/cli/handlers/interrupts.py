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

"""HIL interrupt management for LangGraph execution."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import CompleteStyle
from typing_extensions import NotRequired, TypedDict

from msagent.cli.theme import console
from msagent.cli.ui.shared import (
    build_agent_prompt,
    create_bottom_toolbar,
    create_prompt_style,
)
from msagent.audit.user_interaction import build_user_response_fields
from msagent.configs import ExecuteApprovalMode, ToolApprovalConfig, ToolDecisionRule
from msagent.core.logging import get_logger
from msagent.middlewares.approval import InterruptPayload

if TYPE_CHECKING:
    from langgraph.types import Interrupt

logger = get_logger(__name__)

SWITCH_TO_CONVENIENCE = "Switch to Convenience Mode"
PROJECT_APPROVAL_FILE_NAME = "config.approval.json"
BLACKLIST_PATTERNS = (
    r"(?:^|[;&|]\s*)rm(?:\s+|$).*",
    r"(?:^|[;&|]\s*)del(?:\s+|$).*",
    r"(?:^|[;&|]\s*)remove-item(?:\s+|$).*",
    r"(?:^|[;&|]\s*)rmdir(?:\s+|$).*",
    r"git\s+push.*",
    r"git\s+reset\s+--hard.*",
    r"sudo\s+.*",
)
WRAPPED_COMMAND_PATTERNS = (
    r"(?:^|\s)(?:\S*[\\/])?(?:bash|sh|dash|zsh|ksh)(?:\.exe)?\s+(?:\S+\s+)*-[a-z]*c(?:\s|$)",
    r"(?:^|\s)(?:\S*[\\/])?cmd(?:\.exe)?\s+(?:\S+\s+)*/(?:c|k)(?:\s|$)",
    r"(?:^|\s)(?:\S*[\\/])?(?:powershell|pwsh)(?:\.exe)?\s+"
    r"(?:\S+\s+)*-(?:command|encodedcommand)(?:\s|$)",
    r"(?:^|\s)(?:\S*[\\/])?(?:python|python3|py)(?:\.exe)?\s+(?:\S+\s+)*-c(?:\s|$)",
    r"(?:^|\s)xargs(?:\s+[^;&|]+)?\s+(?:rm|del|remove-item|rmdir)(?:\s|$)",
    r"\$\(",
    r"`[^`]+`",
)
WHITELIST_PATTERNS = (
    r"^\s*ls(?:\s+.*)?\s*$",
    r"^\s*pwd(?:\s+.*)?\s*$",
    r"^\s*ps(?:\s+.*)?\s*$",
    r"^\s*echo(?:\s+.*)?\s*$",
    r"^\s*git\s+status(?:\s+.*)?\s*$",
)


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


class CommandRiskLevel(TypedDict):
    """Risk classification for one shell command."""

    category: str
    default_decision: str
    persistence_scope: str


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
                interrupt = interrupt_data[0]
                choice, user_interacted = await self._get_choice(interrupt)
                if choice is not None and user_interacted:
                    self._record_user_response(interrupt, choice)
                return choice

            # Handle multiple interrupts - return dict mapping IDs to values
            resume_dict = {}
            for interrupt in interrupt_data:
                choice, user_interacted = await self._get_choice(interrupt)
                if choice is not None:
                    resume_dict[interrupt.id] = choice
                    if user_interacted:
                        self._record_user_response(interrupt, choice)

            return resume_dict if resume_dict else None

        except Exception as e:
            console.print_error(f"Error handling interrupt: {e}")
            console.print("")
            return None

    def _record_user_response(self, interrupt: Interrupt, resume_value: Any) -> None:
        """Append a ``user.response`` audit event for one interrupt answer."""
        writer = getattr(self.session, "audit_writer", None)
        if writer is None or not writer.enabled:
            return

        fields = build_user_response_fields(interrupt, resume_value)
        if fields is None:
            return

        writer.emit_user_response(**fields)

    async def _get_choice(self, interrupt: Interrupt) -> tuple[Any, bool]:
        """Choice selector with tab completion and Enter key support.

        Returns:
            Tuple of resume value and whether the user was prompted interactively.
        """
        value = interrupt.value

        if isinstance(value, dict) and "action_requests" in value and "review_configs" in value:
            return await self._get_hitl_decisions(cast(HITLRequest, value))

        if isinstance(value, dict) and "question" in value and "options" in value:
            choice = await self._get_legacy_choice(value)
            return choice, choice is not None

        logger.warning("Unknown interrupt payload shape: %s", type(value).__name__)
        return None, False

    async def _get_hitl_decisions(self, value: HITLRequest) -> tuple[dict[str, Any] | None, bool]:
        """Handle deepagents HITL action/review payload."""
        actions = value.get("action_requests") or []
        review_configs = value.get("review_configs") or []
        if not actions:
            return None, False

        approval_config = self._load_project_approval_config()
        user_interacted = False
        review_by_action = {str(config.get("action_name", "")): config for config in review_configs}
        decisions: list[dict[str, Any]] = []
        for action in actions:
            tool_name = str(action.get("name", "") or "")
            tool_args = action.get("args")
            if not isinstance(tool_args, dict):
                tool_args = {}

            mode_prompted = await self._ensure_execute_approval_mode(tool_name=tool_name)
            if mode_prompted is None and tool_name == "execute":
                return None, user_interacted
            if mode_prompted:
                user_interacted = True

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

            command_profile = self._classify_command(
                tool_name=tool_name,
                tool_args=tool_args,
            )
            policy = self._resolve_decision(
                approval_config=approval_config,
                tool_name=tool_name,
                tool_args=tool_args,
                command_profile=command_profile,
            )
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

            if policy == "approve":
                decisions.append({"type": "approve"})
                continue

            options = self._approval_options(
                allowed=options,
                command_profile=command_profile,
            )

            selected = await self._prompt_hitl_decision(
                tool_name=tool_name,
                tool_args=tool_args,
                description=action.get("description"),
                options=options,
                command_profile=command_profile,
            )
            if selected is None:
                return None, user_interacted

            user_interacted = True

            if selected == SWITCH_TO_CONVENIENCE:
                self._set_execute_approval_mode("convenience")
                decisions.append({"type": "approve"})
                continue

            if selected == "always_approve":
                self._persist_decision(
                    approval_config=approval_config,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    decision="always_approve",
                    scope=command_profile["persistence_scope"],
                )
                decisions.append({"type": "approve"})
                continue

            if selected == "always_reject":
                self._persist_decision(
                    approval_config=approval_config,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    decision="always_reject",
                    scope=command_profile["persistence_scope"],
                )
                decisions.append(
                    {
                        "type": "reject",
                        "message": "Rejected by local approval policy.",
                    }
                )
                continue

            decisions.append(self._selection_to_decision(selected))

        return {"decisions": decisions}, user_interacted

    async def _prompt_hitl_decision(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        description: str | None,
        options: list[str],
        command_profile: CommandRiskLevel,
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
        if tool_name == "execute":
            question_parts.append(f"Mode: {self._get_execute_approval_mode() or 'unset'}")
            question_parts.append(f"Policy: {command_profile['category']}")
        if "always_approve" in options:
            question_parts.append(self._always_rule_note("always_approve", command_profile["persistence_scope"]))
        if "always_reject" in options:
            question_parts.append(self._always_rule_note("always_reject", command_profile["persistence_scope"]))
        question = "\n".join(question_parts)

        return await self._prompt_from_options(question=question, options=options)

    def _resolve_decision(
        self,
        *,
        approval_config: ToolApprovalConfig,
        tool_name: str,
        tool_args: dict[str, Any],
        command_profile: CommandRiskLevel | None = None,
    ) -> str:
        for rule in self._session_decision_rules():
            if rule.matches_call(tool_name, tool_args):
                return rule.decision

        command_profile = command_profile or self._classify_command(tool_name=tool_name, tool_args=tool_args)
        if command_profile["category"] == "blacklist":
            return command_profile["default_decision"]

        persisted = approval_config.resolve_decision(tool_name, tool_args)
        if persisted in {"always_approve", "always_reject"}:
            return persisted
        return command_profile["default_decision"]

    async def _ensure_execute_approval_mode(self, *, tool_name: str) -> bool | None:
        """Prompt once per session for shell approval behavior."""
        if tool_name != "execute":
            return False
        if self._get_execute_approval_mode() is not None:
            return False

        selected = await self._prompt_execute_approval_mode()
        if selected is None:
            return None
        self._set_execute_approval_mode(selected)
        return True

    async def _prompt_execute_approval_mode(self) -> ExecuteApprovalMode | None:
        """Ask the user which shell approval mode to use for this session."""
        question = "\n".join(
            [
                "First command execution: choose shell approval mode.",
                "Safe Mode: whitelist commands auto-approve; blacklist and ordinary commands ask.",
                "Convenience Mode: whitelist and ordinary commands auto-approve; blacklist asks.",
            ]
        )
        selected = await self._prompt_from_options(
            question=question,
            options=["Safe Mode", "Convenience Mode"],
        )
        if selected == "Safe Mode":
            return "safe"
        if selected == "Convenience Mode":
            return "convenience"
        return None

    def _get_execute_approval_mode(self) -> ExecuteApprovalMode | None:
        mode = getattr(self.session, "execute_approval_mode", None)
        if mode in {"convenience", "safe"}:
            return cast(ExecuteApprovalMode, mode)
        return None

    def _set_execute_approval_mode(self, mode: ExecuteApprovalMode) -> None:
        setattr(self.session, "execute_approval_mode", mode)

    def _project_approval_path(self) -> Path:
        state_dir = getattr(self.session.context, "state_dir", None)
        if state_dir is None:
            raise RuntimeError("Project state directory is required for approval rules")
        state_dir = Path(state_dir)
        return state_dir / PROJECT_APPROVAL_FILE_NAME

    def _load_project_approval_config(self) -> ToolApprovalConfig:
        return ToolApprovalConfig.from_json_file(self._project_approval_path())

    def _session_decision_rules(self) -> list[ToolDecisionRule]:
        rules = getattr(self.session, "approval_session_rules", None)
        return list(rules) if isinstance(rules, list) else []

    def _prepend_session_decision_rule(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        decision: str,
    ) -> None:
        rules = getattr(self.session, "approval_session_rules", None)
        if not isinstance(rules, list):
            rules = []
            setattr(self.session, "approval_session_rules", rules)
        config = ToolApprovalConfig(decision_rules=list(rules))
        config.prepend_decision_rule(tool_name=tool_name, tool_args=tool_args, decision=decision)
        rules[:] = config.decision_rules

    def _persist_decision(
        self,
        *,
        approval_config: ToolApprovalConfig,
        tool_name: str,
        tool_args: dict[str, Any],
        decision: str,
        scope: str,
    ) -> None:
        if scope == "project":
            merged_config = ToolApprovalConfig.prepend_rule_to_json_file(
                self._project_approval_path(),
                tool_name=tool_name,
                tool_args=tool_args,
                decision=cast(Any, decision),
            )
            approval_config.decision_rules = merged_config.decision_rules
            return
        self._prepend_session_decision_rule(
            tool_name=tool_name,
            tool_args=tool_args,
            decision=decision,
        )

    def _approval_options(
        self,
        *,
        allowed: list[str],
        command_profile: CommandRiskLevel,
    ) -> list[str]:
        options = list(allowed)
        if "approve" in allowed:
            options.append("always_approve")
        if "reject" in allowed:
            options.append("always_reject")
        if self._get_execute_approval_mode() == "safe":
            options.append(SWITCH_TO_CONVENIENCE)
        return list(dict.fromkeys(options))

    def _classify_command(self, *, tool_name: str, tool_args: dict[str, Any]) -> CommandRiskLevel:
        if tool_name != "execute":
            return {
                "category": "non-execute",
                "default_decision": "ask",
                "persistence_scope": "project",
            }

        command_segments = self._split_command_segments(self._extract_execute_command(tool_args))
        if any(
            self._matches_any(command, BLACKLIST_PATTERNS + WRAPPED_COMMAND_PATTERNS) for command in command_segments
        ):
            return {
                "category": "blacklist",
                "default_decision": "ask",
                "persistence_scope": "session",
            }
        if command_segments and all(self._matches_any(command, WHITELIST_PATTERNS) for command in command_segments):
            return {
                "category": "whitelist",
                "default_decision": "approve",
                "persistence_scope": "session",
            }
        if self._get_execute_approval_mode() == "convenience":
            return {
                "category": "ordinary",
                "default_decision": "approve",
                "persistence_scope": "session",
            }
        return {
            "category": "ordinary",
            "default_decision": "ask",
            "persistence_scope": "project",
        }

    @staticmethod
    def _matches_any(command: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _split_command_segments(command: str) -> list[str]:
        segments: list[str] = []
        current: list[str] = []
        quote: str | None = None
        escaped = False
        idx = 0

        while idx < len(command):
            char = command[idx]
            next_char = command[idx + 1] if idx + 1 < len(command) else ""

            if escaped:
                current.append(char)
                escaped = False
                idx += 1
                continue

            if char == "\\":
                current.append(char)
                escaped = True
                idx += 1
                continue

            if quote:
                current.append(char)
                if char == quote:
                    quote = None
                idx += 1
                continue

            if char in {"'", '"'}:
                current.append(char)
                quote = char
                idx += 1
                continue

            if char in {";", "|", "&"}:
                segment = "".join(current).strip()
                if segment:
                    segments.append(segment)
                current = []
                idx += 2 if char in {"&", "|"} and next_char == char else 1
                continue

            current.append(char)
            idx += 1

        segment = "".join(current).strip()
        if segment:
            segments.append(segment)
        return segments

    @staticmethod
    def _extract_execute_command(tool_args: dict[str, Any]) -> str:
        for key in ("command", "cmd"):
            value = tool_args.get(key)
            if value is not None:
                return str(value).strip()
        return ""

    @staticmethod
    def _always_rule_note(decision: str, scope: str) -> str:
        if scope == "project":
            return (
                f"Warning: choosing {decision} will persist for this project and auto-apply only when "
                "a later command has exactly the same command text."
            )
        return f"Warning: choosing {decision} will apply only to this session for the same call."

    @staticmethod
    def _selection_to_decision(selected: str) -> dict[str, Any]:
        """Convert interactive selection into HITL decision payload."""
        if selected == "approve":
            return {"type": "approve"}
        if selected == "reject":
            return {"type": "reject"}
        return {"type": "approve"}

    async def _get_legacy_choice(self, value: InterruptPayload | dict[str, Any]) -> str | None:
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
                        interrupt_session.default_buffer.start_completion(select_first=False)

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
                        console.print_error(f"Ambiguous choice. Options: {', '.join(matches)}")
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
