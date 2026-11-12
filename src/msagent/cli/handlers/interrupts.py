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

import re
import shlex
import sys
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import NotRequired, TypedDict

from msagent.audit.user_interaction import build_user_response_fields
from msagent.cli.theme import console
from msagent.cli.ui.approval import ApprovalChoice, ApprovalMenu, format_approval_details
from msagent.configs import ExecuteApprovalMode, ToolApprovalConfig
from msagent.core.logging import get_logger
from msagent.middlewares.approval import InterruptPayload

if TYPE_CHECKING:
    from langgraph.types import Interrupt

logger = get_logger(__name__)

SWITCH_TO_AUTO = "Switch to Auto Mode"
ALWAYS_ALLOW_PROJECT_SCRIPT = "always_allow_project_script"
PROJECT_APPROVAL_FILE_NAME = "config.approval.json"
DANGEROUS_PATTERNS = (
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
READONLY_PATTERNS = (
    r"^\s*ls(?:\s+.*)?\s*$",
    r"^\s*cd(?:\s+.*)?\s*$",
    r"^\s*du(?:\s+.*)?\s*$",
    r"^\s*sort(?:\s+.*)?\s*$",
    r"^\s*head(?:\s+.*)?\s*$",
    r"^\s*tail(?:\s+.*)?\s*$",
    r"^\s*ldd\s+--version(?:\s+2>&1)?\s*$",
    r"^\s*pwd(?:\s+.*)?\s*$",
    r"^\s*ps(?:\s+.*)?\s*$",
    r"^\s*echo(?:\s+.*)?\s*$",
    r"^\s*git\s+status(?:\s+.*)?\s*$",
)
FORBIDDEN_PATTERNS = (
    r"^\s*rm\s+(?:-[^\s]+\s+)*(?:/|~)\s*$",
    r"^\s*remove-item\s+.*(?:-recurse\s+.*)?(?:[a-z]:\\|/)\s*$",
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
    family: NotRequired[str]
    script_interpreter: NotRequired[str]
    script_path: NotRequired[str]


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
        """Resolve an interrupt and track whether it prompted the user.

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

            mode_prompted = await self._ensure_execute_approval_mode(
                tool_name=tool_name,
                approval_config=approval_config,
            )
            if mode_prompted is None and tool_name == "execute":
                return None, user_interacted
            if mode_prompted:
                user_interacted = True

            command_profile = self._classify_command(tool_name=tool_name, tool_args=tool_args)
            boundary_rejected = False
            external_directories = []
            if command_profile["category"] not in {"dangerous", "forbidden"}:
                external_directories = self._external_directories_requiring_approval(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    approval_config=approval_config,
                    inspect_command_paths=command_profile["category"] != "opaque",
                )
            for directory in external_directories:
                selected = await self._prompt_external_directory(directory=directory)
                if selected is None:
                    return None, user_interacted
                user_interacted = True
                if selected == "reject":
                    decisions.append({"type": "reject", "message": "External directory access was rejected."})
                    boundary_rejected = True
                    break
                if selected == "always_approve":
                    merged = ToolApprovalConfig.prepend_external_directory_rule_to_json_file(
                        self._project_approval_path(), directory
                    )
                    approval_config.external_directories = merged.external_directories
            if boundary_rejected:
                continue

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
                approval_config=approval_config,
            )

            selected_result = await self._prompt_hitl_decision(
                tool_name=tool_name,
                tool_args=tool_args,
                description=action.get("description"),
                options=options,
                command_profile=command_profile,
            )
            if selected_result is None:
                return None, user_interacted

            if isinstance(selected_result, ApprovalChoice):
                selected = selected_result.value
                rejection_message = selected_result.message
            else:
                selected = selected_result
                rejection_message = None

            user_interacted = True

            if selected == SWITCH_TO_AUTO:
                self._set_execute_approval_mode("auto", persist=True, source="project")
                decisions.append({"type": "approve"})
                continue

            if selected == ALWAYS_ALLOW_PROJECT_SCRIPT:
                interpreter = command_profile.get("script_interpreter")
                script_path = command_profile.get("script_path")
                if interpreter and script_path:
                    merged = ToolApprovalConfig.prepend_script_rule_to_json_file(
                        self._project_approval_path(),
                        interpreter=interpreter,
                        path=Path(script_path),
                        decision="always_approve",
                    )
                    approval_config.script_rules = merged.script_rules
                decisions.append({"type": "approve"})
                continue

            if selected == "always_approve":
                self._persist_decision(
                    approval_config=approval_config,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    decision="always_approve",
                    scope=command_profile["persistence_scope"],
                    family=command_profile.get("family"),
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
                    family=command_profile.get("family"),
                )
                decisions.append(
                    {
                        "type": "reject",
                        "message": "Rejected by local approval policy.",
                    }
                )
                continue

            decision = self._selection_to_decision(selected)
            if selected == "reject" and rejection_message:
                decision["message"] = rejection_message
            decisions.append(decision)

        return {"decisions": decisions}, user_interacted

    async def _prompt_hitl_decision(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        description: str | None,
        options: list[str],
        command_profile: CommandRiskLevel,
    ) -> ApprovalChoice | str | None:
        """Prompt user for one HITL decision."""
        details = format_approval_details(
            tool_name=tool_name,
            tool_args=tool_args,
            description=description,
        )
        metadata: list[str] = []
        notes: list[str] = []
        if tool_name == "execute":
            metadata.append(f"Mode: {self.session.execute_approval_mode or 'unset'}")
        if "always_approve" in options:
            notes.append(
                self._always_rule_note(
                    "always_approve",
                    command_profile["persistence_scope"],
                    family=command_profile.get("family"),
                )
            )
        if "always_reject" in options:
            notes.append(self._always_rule_note("always_reject", command_profile["persistence_scope"]))
        if ALWAYS_ALLOW_PROJECT_SCRIPT in options:
            notes.append(
                "Note: this allows the named script with any arguments for this project; "
                "later changes to that script will also run."
            )
        menu = ApprovalMenu(
            context=self.session.context,
            title=f"{tool_name} Requires Approval",
            details=details,
            metadata=metadata,
            notes=notes,
            options=options,
            option_labels=self._approval_option_labels(command_profile),
        )
        return await menu.run_async()

    @staticmethod
    def _approval_option_labels(command_profile: CommandRiskLevel) -> dict[str, str]:
        """Describe the actual persistence scope directly in menu rows."""
        labels = {SWITCH_TO_AUTO: SWITCH_TO_AUTO}
        if command_profile.get("script_path"):
            labels[ALWAYS_ALLOW_PROJECT_SCRIPT] = "Always allow this project script"
        family = command_profile.get("family")
        if family:
            labels["always_approve"] = f"Always allow {family} for this project"
        elif command_profile["persistence_scope"] == "project":
            labels["always_approve"] = "Always allow this exact command for this project"
            labels["always_reject"] = "Always reject this exact command for this project"
        return labels

    async def _prompt_external_directory(self, *, directory: Path) -> str | None:
        """Prompt before a tool crosses the current project boundary."""
        choice = await self._prompt_approval_menu(
            title="External Directory Requires Approval",
            details="Tool access is outside the current project.",
            metadata=[f"Directory: {directory}"],
            notes=["Note: always allow grants this directory and its descendants for this project."],
            options=["approve", "always_approve", "reject"],
            option_labels={"always_approve": "Always allow this directory for this project"},
        )
        return choice.value if choice else None

    def _resolve_decision(
        self,
        *,
        approval_config: ToolApprovalConfig,
        tool_name: str,
        tool_args: dict[str, Any],
        command_profile: CommandRiskLevel | None = None,
    ) -> str:
        command_profile = command_profile or self._classify_command(tool_name=tool_name, tool_args=tool_args)
        if command_profile["category"] in {"dangerous", "forbidden"}:
            return command_profile["default_decision"]

        interpreter = command_profile.get("script_interpreter")
        script_path = command_profile.get("script_path")
        if interpreter and script_path:
            persisted_script = approval_config.resolve_script_decision(
                interpreter=interpreter,
                path=script_path,
            )
            if persisted_script in {"always_approve", "always_reject"}:
                return persisted_script

        if command_profile["category"] == "opaque":
            persisted_family = approval_config.resolve_family_decision(command_profile.get("family"))
            if persisted_family in {"always_approve", "always_reject"}:
                return persisted_family
            return command_profile["default_decision"]

        if command_profile["category"] == "readonly":
            return command_profile["default_decision"]

        persisted = approval_config.resolve_decision(tool_name, tool_args)
        if persisted in {"always_approve", "always_reject"}:
            return persisted
        return command_profile["default_decision"]

    async def _ensure_execute_approval_mode(
        self,
        *,
        tool_name: str,
        approval_config: ToolApprovalConfig,
    ) -> bool | None:
        """Restore the project shell mode or prompt once and persist it."""
        if tool_name != "execute":
            return False
        if self.session.execute_approval_mode is not None:
            return False

        if approval_config.execute_approval_mode is not None:
            self._set_execute_approval_mode(approval_config.execute_approval_mode, source="project")
            return False

        selected = await self._prompt_execute_approval_mode()
        if selected is None:
            return None
        self._set_execute_approval_mode(selected, source="project")
        merged = ToolApprovalConfig.update_mode_in_json_file(self._project_approval_path(), selected)
        approval_config.execute_approval_mode = merged.execute_approval_mode
        return True

    async def _prompt_execute_approval_mode(self) -> ExecuteApprovalMode | None:
        """Ask the user which shell approval mode to use for this session."""
        question = "\n".join(
            [
                "First command execution: choose shell approval mode.",
                "Manual Mode: readonly commands auto-approve; code and other commands ask.",
                "Auto Mode: all normal commands, including inline code, auto-approve.",
                "Dangerous commands always ask; forbidden commands are rejected.",
            ]
        )
        choice = await self._prompt_approval_menu(
            title="Choose Approval Mode",
            details=question,
            options=["Manual Mode", "Auto Mode"],
        )
        selected = choice.value if choice else None
        if selected == "Manual Mode":
            return "manual"
        if selected == "Auto Mode":
            return "auto"
        return None

    def _set_execute_approval_mode(
        self,
        mode: ExecuteApprovalMode,
        *,
        persist: bool = False,
        source: str | None = None,
    ) -> None:
        setattr(self.session, "execute_approval_mode", mode)
        if source is not None:
            setattr(self.session, "execute_approval_mode_source", source)
        if persist:
            ToolApprovalConfig.update_mode_in_json_file(self._project_approval_path(), mode)

    def _project_approval_path(self) -> Path:
        state_dir = getattr(self.session.context, "state_dir", None)
        if state_dir is None:
            raise RuntimeError("Project state directory is required for approval rules")
        state_dir = Path(state_dir)
        return state_dir / PROJECT_APPROVAL_FILE_NAME

    def _load_project_approval_config(self) -> ToolApprovalConfig:
        return ToolApprovalConfig.from_json_file(self._project_approval_path())

    def _persist_decision(
        self,
        *,
        approval_config: ToolApprovalConfig,
        tool_name: str,
        tool_args: dict[str, Any],
        decision: str,
        scope: str,
        family: str | None = None,
    ) -> None:
        if scope == "project":
            if family:
                merged_config = ToolApprovalConfig.prepend_family_rule_to_json_file(
                    self._project_approval_path(),
                    family=family,
                    decision=cast(Any, decision),
                )
                approval_config.family_rules = merged_config.family_rules
                return
            merged_config = ToolApprovalConfig.prepend_rule_to_json_file(
                self._project_approval_path(),
                tool_name=tool_name,
                tool_args=tool_args,
                decision=cast(Any, decision),
            )
            approval_config.decision_rules = merged_config.decision_rules
            return

    def _approval_options(
        self,
        *,
        allowed: list[str],
        command_profile: CommandRiskLevel,
        approval_config: ToolApprovalConfig | None = None,
    ) -> list[str]:
        category = command_profile["category"]
        options = list(allowed)
        if category in {"dangerous", "forbidden", "readonly"}:
            return list(dict.fromkeys(options))
        if category == "opaque":
            opaque_options = []
            if "approve" in allowed:
                opaque_options.extend(["approve", "always_approve"])
            if "reject" in allowed:
                opaque_options.append("reject")
            if self.session.execute_approval_mode == "manual":
                opaque_options.append(SWITCH_TO_AUTO)
            return opaque_options
        if self._is_project_script_eligible(command_profile, approval_config):
            script_options = []
            if "approve" in allowed:
                script_options.extend(["approve", ALWAYS_ALLOW_PROJECT_SCRIPT])
            if "reject" in allowed:
                script_options.extend(["reject", "always_reject"])
            if self.session.execute_approval_mode == "manual":
                script_options.append(SWITCH_TO_AUTO)
            return script_options
        if "approve" in allowed:
            options.append("always_approve")
        if category == "ordinary" and "reject" in allowed:
            options.append("always_reject")
        if category == "ordinary" and self.session.execute_approval_mode == "manual":
            options.append(SWITCH_TO_AUTO)
        return list(dict.fromkeys(options))

    def _classify_command(self, *, tool_name: str, tool_args: dict[str, Any]) -> CommandRiskLevel:
        if tool_name != "execute":
            return {
                "category": "readonly",
                "default_decision": "approve",
                "persistence_scope": "none",
            }

        command_segments = self._split_command_segments(self._extract_execute_command(tool_args))
        if any(self._matches_any(command, FORBIDDEN_PATTERNS) for command in command_segments):
            return {
                "category": "forbidden",
                "default_decision": "always_reject",
                "persistence_scope": "none",
            }
        if any(self._matches_any(command, DANGEROUS_PATTERNS) for command in command_segments):
            return {
                "category": "dangerous",
                "default_decision": "ask",
                "persistence_scope": "none",
            }
        opaque_family = self._opaque_command_family(command_segments)
        if opaque_family:
            return {
                "category": "opaque",
                "default_decision": "approve" if self.session.execute_approval_mode == "auto" else "ask",
                "persistence_scope": "project",
                "family": opaque_family,
            }
        if command_segments and all(
            self._matches_any(command, READONLY_PATTERNS) and self._is_safe_readonly_segment(command)
            for command in command_segments
        ):
            return {
                "category": "readonly",
                "default_decision": "approve",
                "persistence_scope": "none",
            }
        profile: CommandRiskLevel = {
            "category": "ordinary",
            "default_decision": "approve" if self.session.execute_approval_mode == "auto" else "ask",
            "persistence_scope": "none" if self.session.execute_approval_mode == "auto" else "project",
        }
        script = self._direct_script_launch(command_segments)
        if script:
            profile["script_interpreter"] = script[0]
            profile["script_path"] = script[1]
        return profile

    def _direct_script_launch(self, command_segments: list[str]) -> tuple[str, str] | None:
        """Recognize one script launch surrounded only by limited readonly shell steps."""
        base = Path(self.session.context.working_dir).expanduser().resolve(strict=False)
        script: tuple[str, str] | None = None
        for command in command_segments:
            cd_target = self._cd_target(command, base=base)
            if cd_target is not None:
                if script is not None:
                    return None
                base = cd_target
                continue
            launch = self._script_launch_from_segment(command, base=base)
            if launch is not None:
                if script is not None:
                    return None
                script = launch
                continue
            if not (self._matches_any(command, READONLY_PATTERNS) and self._is_safe_readonly_segment(command)):
                return None
        return script

    def _script_launch_from_segment(self, command: str, *, base: Path) -> tuple[str, str] | None:
        """Recognize an interpreter-file invocation, optionally preceded by time."""
        if self._has_output_redirection(command):
            return None
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return None
        if tokens and tokens[0] == "time":
            tokens = tokens[1:]
        if len(tokens) < 2:
            return None
        executable = Path(tokens[0]).name.lower().removesuffix(".exe")
        if executable not in {"python", "python3", "py", "bash", "sh"}:
            return None
        script_token = tokens[1]
        if script_token.startswith("-"):
            return None
        candidate = Path(script_token).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.resolve(strict=False)
        if not candidate.is_file():
            return None
        if executable in {"python", "python3", "py"} and candidate.suffix.lower() != ".py":
            return None
        return executable, str(candidate)

    @staticmethod
    def _cd_target(command: str, *, base: Path) -> Path | None:
        """Return a simple cd target, leaving other readonly commands untouched."""
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            return None
        if len(tokens) != 2 or tokens[0].lower() != "cd":
            return None
        return InterruptHandler._resolve_path(tokens[1], base=base, directory=True)

    def _is_project_script_eligible(
        self,
        command_profile: CommandRiskLevel,
        approval_config: ToolApprovalConfig | None,
    ) -> bool:
        """Remember only scripts inside the project or an already approved directory."""
        script_path = command_profile.get("script_path")
        if not script_path:
            return False
        script = Path(script_path)
        base = Path(self.session.context.working_dir).expanduser().resolve(strict=False)
        if any(script == root or root in script.parents for root in self._project_roots(base)):
            return True
        return approval_config is not None and approval_config.allows_external_path(script)

    @staticmethod
    def _opaque_command_family(command_segments: list[str]) -> str | None:
        """Return a stable, intentionally narrow family for an opaque wrapper."""
        joined = " && ".join(command_segments)
        family_patterns = (
            (
                r"(?:^|\s)(?:\S*[\\/])?(python3?|py)(?:\.exe)?\s+(?:\S+\s+)*(?:-c|-)(?=\s|$)",
                1,
                " inline code",
            ),
            (r"(?:^|\s)(?:\S*[\\/])?(bash|sh|dash|zsh|ksh)(?:\.exe)?\s+(?:\S+\s+)*-[a-z]*c(?:\s|$)", 1, " -c"),
            (r"(?:^|\s)(?:\S*[\\/])?(cmd)(?:\.exe)?\s+(?:\S+\s+)*/(?:c|k)(?:\s|$)", 1, " /c"),
            (
                r"(?:^|\s)(?:\S*[\\/])?(powershell|pwsh)(?:\.exe)?\s+(?:\S+\s+)*-(?:command|encodedcommand)(?:\s|$)",
                1,
                " -command",
            ),
        )
        for pattern, group, suffix in family_patterns:
            match = re.search(pattern, joined, flags=re.IGNORECASE)
            if match:
                return f"{match.group(group).lower()}{suffix}"
        if re.search(r"(?:^|\s)xargs(?:\s+[^;&|]+)?\s+(?:rm|del|remove-item|rmdir)(?:\s|$)", joined, re.I):
            return "xargs destructive"
        if re.search(r"\$\(|`[^`]+`", joined):
            return "shell substitution"
        return None

    def _external_directories_requiring_approval(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        approval_config: ToolApprovalConfig,
        inspect_command_paths: bool = True,
    ) -> list[Path]:
        """Return explicit, reliably parsed directory accesses outside project roots."""
        candidates: list[Path] = []
        base = Path(self.session.context.working_dir).expanduser().resolve(strict=False)

        for key in ("cwd", "workdir"):
            value = tool_args.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(self._resolve_path(value, base=base, directory=True))

        if tool_name == "execute":
            for segment in self._split_command_segments(self._extract_execute_command(tool_args)):
                try:
                    lexer = shlex.shlex(segment, posix=True)
                    lexer.whitespace_split = True
                    lexer.commenters = ""
                    if sys.platform == "win32":
                        # Windows paths use backslashes as separators, not shell escapes.
                        lexer.escape = ""
                    tokens = list(lexer)
                except ValueError:
                    continue
                if len(tokens) >= 2 and tokens[0].lower() == "cd":
                    candidates.append(self._resolve_path(tokens[1], base=base, directory=True))
                    continue
                if inspect_command_paths:
                    for token in tokens[1:]:
                        windows_path = sys.platform == "win32" and PureWindowsPath(token).anchor
                        if windows_path or token.startswith(("/", "~/", "../", "..\\")):
                            candidates.append(self._resolve_path(token, base=base, directory=False))
        else:
            for key in ("path", "file_path", "directory", "root"):
                value = tool_args.get(key)
                if not isinstance(value, str) or not value.strip():
                    continue
                candidates.append(
                    self._resolve_path(
                        value,
                        base=base,
                        directory=key in {"directory", "root"} or tool_name in {"glob", "grep"},
                    )
                )

        roots = self._project_roots(base)
        result: list[Path] = []
        for candidate in candidates:
            if any(candidate == root or root in candidate.parents for root in roots):
                continue
            if approval_config.allows_external_path(candidate):
                continue
            if candidate not in result:
                result.append(candidate)
        return result

    @staticmethod
    def _resolve_path(value: str, *, base: Path, directory: bool) -> Path:
        """Resolve an explicit path and reduce existing files to their parent directory."""
        raw = value.strip()
        wildcard_at = min((raw.find(char) for char in "*?[" if char in raw), default=-1)
        if wildcard_at >= 0:
            raw = raw[:wildcard_at].rstrip("/\\") or "."
            directory = True
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = base / path
        resolved = path.resolve(strict=False)
        if not directory and (resolved.is_file() or resolved.suffix):
            return resolved.parent
        return resolved

    @staticmethod
    def _project_roots(working_dir: Path) -> list[Path]:
        """Return the working directory and its nearest Git worktree root."""
        roots = [working_dir]
        for candidate in (working_dir, *working_dir.parents):
            if (candidate / ".git").exists():
                if candidate not in roots:
                    roots.append(candidate)
                break
        return roots

    @staticmethod
    def _matches_any(command: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, command, flags=re.IGNORECASE) for pattern in patterns)

    @staticmethod
    def _is_safe_readonly_segment(command: str) -> bool:
        if InterruptHandler._has_output_redirection(command):
            return False
        if re.match(r"^\s*sort(?:\s|$)", command, flags=re.IGNORECASE):
            return not re.search(r"(?:^|\s)(?:-o|--output|--compress-program)(?:\s|=|$)", command, re.IGNORECASE)
        return True

    @staticmethod
    def _has_output_redirection(command: str) -> bool:
        return bool(re.search(r"(?<![<&])(?:\d+)?>{1,2}(?![>&])", command))

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
                if char == "&" and current and current[-1] == ">":
                    current.append(char)
                    idx += 1
                    continue
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
    def _always_rule_note(decision: str, scope: str, *, family: str | None = None) -> str:
        if family:
            return (
                f"Note: choosing {decision} will allow the '{family}' command family "
                "for this project until the rule is removed with /permissions."
            )
        if scope == "project":
            return (
                f"Note: choosing {decision} will persist for this project and auto-apply only when "
                "a later command has exactly the same command text."
            )
        return f"Note: choosing {decision} will apply only to this session for the same call."

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
        choice = await self._prompt_approval_menu(
            title="Approval Required",
            details=question,
            options=options,
        )
        return choice.value if choice else None

    async def _prompt_approval_menu(
        self,
        *,
        title: str,
        details: str,
        options: list[str],
        metadata: list[str] | None = None,
        notes: list[str] | None = None,
        option_labels: dict[str, str] | None = None,
    ) -> ApprovalChoice | None:
        """Render the shared approval TUI for every interactive approval path."""
        menu = ApprovalMenu(
            context=self.session.context,
            title=title,
            details=details,
            metadata=metadata,
            notes=notes,
            options=options,
            option_labels=option_labels,
        )
        return await menu.run_async()
