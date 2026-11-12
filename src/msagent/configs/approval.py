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

"""Tool approval/HITL configuration classes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

DecisionType = Literal["approve", "edit", "reject"]
ToolDecision = Literal["ask", "always_approve", "always_reject"]
ExecuteApprovalMode = Literal["auto", "manual"]


def _stable_rule_id(kind: str, payload: dict[str, Any]) -> str:
    """Build a short deterministic ID so legacy rules are addressable after reload."""
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(f"{kind}:{encoded}".encode("utf-8")).hexdigest()[:12]
    return f"{kind}-{digest}"


class ApprovalMode(str, Enum):
    """Tool approval mode for interactive sessions."""

    SEMI_ACTIVE = "semi-active"  # No effect
    ACTIVE = "active"  # Conservative interactive mode
    AGGRESSIVE = "aggressive"  # Most permissive interactive mode


def _default_allowed_decisions() -> list[DecisionType]:
    """Default allowed HITL decisions."""
    return ["approve", "reject"]


class InterruptOnRule(BaseModel):
    """Per-tool Human-in-the-Loop rule for deepagents."""

    allowed_decisions: list[DecisionType] = Field(default_factory=_default_allowed_decisions)
    description: str | None = None
    args_schema: dict[str, Any] | None = None


class ToolDecisionRule(BaseModel):
    """Rule for deciding whether a matching tool call should prompt or auto-resolve."""

    id: str = ""
    name: str
    args: dict[str, Any] | None = None
    decision: ToolDecision = "ask"

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = _stable_rule_id(
                "exact",
                {"name": self.name, "args": self.args, "decision": self.decision},
            )

    def matches_call(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Check if this rule matches a specific tool call."""
        if self.name != tool_name:
            return False

        if not self.args:
            return True

        for key, expected_value in self.args.items():
            if key not in tool_args:
                return False

            actual_value = str(tool_args[key])
            expected_str = str(expected_value)

            if actual_value == expected_str:
                continue

            try:
                pattern = re.compile(expected_str)
                if pattern.search(actual_value):
                    continue
            except re.error:
                pass

            return False

        return True


class CommandFamilyRule(BaseModel):
    """Project rule for one opaque command family."""

    id: str = ""
    family: str
    decision: ToolDecision = "ask"

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = _stable_rule_id("family", {"family": self.family, "decision": self.decision})


class ProjectScriptRule(BaseModel):
    """Project rule for one directly launched script, irrespective of arguments."""

    id: str = ""
    interpreter: str
    path: str
    decision: ToolDecision = "ask"

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = _stable_rule_id(
                "script",
                {"interpreter": self.interpreter, "path": self.path, "decision": self.decision},
            )


class ExternalDirectoryRule(BaseModel):
    """Project grant for a canonical external directory and its descendants."""

    id: str = ""
    path: str

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = _stable_rule_id("external", {"path": self.path})


def _default_interrupt_on_rules() -> dict[str, InterruptOnRule]:
    """Default HITL rules for high-risk tools."""
    rules = {
        "execute": InterruptOnRule(
            allowed_decisions=["approve", "reject"],
            description="Review shell command execution before running.",
        ),
    }
    for tool_name in ("ls", "read_file", "write_file", "edit_file", "glob", "grep"):
        rules[tool_name] = InterruptOnRule(
            allowed_decisions=["approve", "reject"],
            description="Review access outside the project before continuing.",
        )
    return rules


def _default_interrupt_on_field() -> dict[str, bool | InterruptOnRule]:
    """Return default interrupt_on payload with the field's wider value type."""
    return cast(dict[str, bool | InterruptOnRule], _default_interrupt_on_rules())


def _default_decision_rules() -> list[ToolDecisionRule]:
    """Return an empty list; built-in policy is resolved by the interrupt handler."""
    return []


class ToolApprovalConfig(BaseModel):
    """Configuration for deepagents HITL tool approvals."""

    model_config = ConfigDict(extra="ignore")

    interrupt_on: dict[str, bool | InterruptOnRule] = Field(default_factory=_default_interrupt_on_field)
    decision_rules: list[ToolDecisionRule] = Field(default_factory=_default_decision_rules)
    family_rules: list[CommandFamilyRule] = Field(default_factory=list)
    script_rules: list[ProjectScriptRule] = Field(default_factory=list)
    external_directories: list[ExternalDirectoryRule] = Field(default_factory=list)
    execute_approval_mode: ExecuteApprovalMode | None = None

    def to_interrupt_on_payload(self) -> dict[str, bool | dict[str, Any]] | None:
        """Return interrupt_on payload compatible with deepagents create_deep_agent."""
        payload: dict[str, bool | dict[str, Any]] = {}
        for tool_name, rule in self.interrupt_on.items():
            if isinstance(rule, bool):
                if rule:
                    payload[tool_name] = True
                continue

            rule_payload = rule.model_dump(exclude_none=True)
            allowed = list(rule_payload.get("allowed_decisions", []))
            if not allowed:
                continue
            payload[tool_name] = rule_payload

        return payload or None

    def resolve_decision(self, tool_name: str, tool_args: dict[str, Any]) -> ToolDecision:
        """Resolve decision policy for a tool call based on decision_rules."""
        for rule in self.decision_rules:
            if rule.matches_call(tool_name, tool_args):
                return rule.decision
        return "ask"

    def resolve_family_decision(self, family: str | None) -> ToolDecision:
        """Resolve a persisted opaque-command family decision."""
        if not family:
            return "ask"
        for rule in self.family_rules:
            if rule.family == family:
                return rule.decision
        return "ask"

    def resolve_script_decision(self, *, interpreter: str, path: str) -> ToolDecision:
        """Resolve a direct-script rule, intentionally ignoring invocation arguments."""
        for rule in self.script_rules:
            if rule.interpreter == interpreter and rule.path == path:
                return rule.decision
        return "ask"

    def allows_external_path(self, path: Path) -> bool:
        """Return whether a path is covered by a persisted directory grant."""
        candidate = path.expanduser().resolve(strict=False)
        for rule in self.external_directories:
            allowed = Path(rule.path).expanduser().resolve(strict=False)
            if candidate == allowed or allowed in candidate.parents:
                return True
        return False

    def prepend_decision_rule(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        decision: ToolDecision,
    ) -> None:
        """Add a high-priority in-memory rule for a concrete tool call."""
        normalized_args: dict[str, Any] = {}
        for key, value in (tool_args or {}).items():
            if isinstance(value, str):
                normalized_args[key] = rf"^{re.escape(value)}$"
            else:
                normalized_args[key] = value

        self.decision_rules = [
            rule
            for rule in self.decision_rules
            if not (rule.name == tool_name and rule.args == normalized_args and rule.decision == decision)
        ]
        self.decision_rules.insert(
            0,
            ToolDecisionRule(
                name=tool_name,
                args=normalized_args or None,
                decision=decision,
            ),
        )

    def prepend_family_rule(self, *, family: str, decision: ToolDecision) -> None:
        """Add a high-priority exact family rule."""
        self.family_rules = [rule for rule in self.family_rules if rule.family != family]
        self.family_rules.insert(0, CommandFamilyRule(family=family, decision=decision))

    def prepend_script_rule(self, *, interpreter: str, path: Path, decision: ToolDecision) -> None:
        """Add a high-priority rule for one canonical script and interpreter."""
        normalized_path = str(path.expanduser().resolve(strict=False))
        self.script_rules = [
            rule for rule in self.script_rules if not (rule.interpreter == interpreter and rule.path == normalized_path)
        ]
        self.script_rules.insert(
            0,
            ProjectScriptRule(interpreter=interpreter, path=normalized_path, decision=decision),
        )

    def prepend_external_directory_rule(self, directory: Path) -> None:
        """Add a canonical recursive external-directory grant."""
        normalized = str(directory.expanduser().resolve(strict=False))
        self.external_directories = [rule for rule in self.external_directories if rule.path != normalized]
        self.external_directories.insert(0, ExternalDirectoryRule(path=normalized))

    @classmethod
    def from_json_file(cls, file_path: Path) -> "ToolApprovalConfig":
        """Load project approval rules, falling back to an empty config."""
        if not file_path.is_file():
            return cls(decision_rules=[])
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return cls(decision_rules=[])
        if not isinstance(payload, dict):
            return cls(decision_rules=[])
        try:
            return cls.model_validate(payload)
        except (TypeError, ValueError):
            return cls(decision_rules=[])

    def save_to_json_file(self, file_path: Path) -> None:
        """Atomically save project approval rules while holding the project lock."""
        with _approval_file_lock(file_path):
            self._save_to_json_file_unlocked(file_path)

    @classmethod
    def prepend_rule_to_json_file(
        cls,
        file_path: Path,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        decision: ToolDecision,
    ) -> "ToolApprovalConfig":
        """Reload, prepend, and save a rule as one cross-process operation."""
        with _approval_file_lock(file_path):
            config = cls.from_json_file(file_path)
            config.prepend_decision_rule(
                tool_name=tool_name,
                tool_args=tool_args,
                decision=decision,
            )
            config._save_to_json_file_unlocked(file_path)
            return config

    @classmethod
    def update_mode_in_json_file(
        cls,
        file_path: Path,
        mode: ExecuteApprovalMode,
    ) -> "ToolApprovalConfig":
        """Reload and save the project shell mode as one operation."""
        with _approval_file_lock(file_path):
            config = cls.from_json_file(file_path)
            config.execute_approval_mode = mode
            config._save_to_json_file_unlocked(file_path)
            return config

    @classmethod
    def prepend_family_rule_to_json_file(
        cls,
        file_path: Path,
        *,
        family: str,
        decision: ToolDecision,
    ) -> "ToolApprovalConfig":
        """Reload, prepend, and save an opaque-command family rule."""
        with _approval_file_lock(file_path):
            config = cls.from_json_file(file_path)
            config.prepend_family_rule(family=family, decision=decision)
            config._save_to_json_file_unlocked(file_path)
            return config

    @classmethod
    def prepend_script_rule_to_json_file(
        cls,
        file_path: Path,
        *,
        interpreter: str,
        path: Path,
        decision: ToolDecision,
    ) -> "ToolApprovalConfig":
        """Reload, prepend, and save a direct-script rule."""
        with _approval_file_lock(file_path):
            config = cls.from_json_file(file_path)
            config.prepend_script_rule(interpreter=interpreter, path=path, decision=decision)
            config._save_to_json_file_unlocked(file_path)
            return config

    @classmethod
    def prepend_external_directory_rule_to_json_file(
        cls,
        file_path: Path,
        directory: Path,
    ) -> "ToolApprovalConfig":
        """Reload, prepend, and save a recursive external-directory grant."""
        with _approval_file_lock(file_path):
            config = cls.from_json_file(file_path)
            config.prepend_external_directory_rule(directory)
            config._save_to_json_file_unlocked(file_path)
            return config

    @classmethod
    def remove_rule_from_json_file(cls, file_path: Path, rule_id: str) -> bool:
        """Remove one project rule by stable ID, preserving mode and other rules."""
        with _approval_file_lock(file_path):
            config = cls.from_json_file(file_path)
            original_count = (
                len(config.decision_rules)
                + len(config.family_rules)
                + len(config.script_rules)
                + len(config.external_directories)
            )
            config.decision_rules = [rule for rule in config.decision_rules if rule.id != rule_id]
            config.family_rules = [rule for rule in config.family_rules if rule.id != rule_id]
            config.script_rules = [rule for rule in config.script_rules if rule.id != rule_id]
            config.external_directories = [rule for rule in config.external_directories if rule.id != rule_id]
            updated_count = (
                len(config.decision_rules)
                + len(config.family_rules)
                + len(config.script_rules)
                + len(config.external_directories)
            )
            if updated_count == original_count:
                return False
            config._save_to_json_file_unlocked(file_path)
            return True

    def _save_to_json_file_unlocked(self, file_path: Path) -> None:
        """Atomically replace an approval file; caller coordinates concurrent writers."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            file_path.parent.chmod(0o700)
        except OSError:
            pass
        payload = self.model_dump(mode="json", exclude_none=True)
        temp_file = file_path.with_name(f".{file_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            temp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            try:
                temp_file.chmod(0o600)
            except OSError:
                pass
            temp_file.replace(file_path)
        finally:
            if temp_file.exists():
                temp_file.unlink()


@contextmanager
def _approval_file_lock(file_path: Path) -> Iterator[None]:
    """Serialize approval-file updates across msagent processes."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = file_path.with_name(f".{file_path.name}.lock")
    with lock_path.open("a+b") as lock_file:
        try:
            lock_path.chmod(0o600)
        except OSError:
            pass
        lock_file.seek(0)
        if lock_file.read(1) == b"":
            lock_file.write(b"0")
            lock_file.flush()
        lock_file.seek(0)

        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)  # type: ignore[attr-defined]
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
