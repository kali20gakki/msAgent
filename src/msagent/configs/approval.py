"""Tool approval/HITL configuration classes."""

from __future__ import annotations

import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

DecisionType = Literal["approve", "edit", "reject"]
ToolDecision = Literal["ask", "always_approve", "always_reject"]


class ApprovalMode(str, Enum):
    """Tool approval mode for interactive sessions."""

    SEMI_ACTIVE = "semi-active"  # No effect
    ACTIVE = "active"  # Conservative interactive mode
    AGGRESSIVE = "aggressive"  # Most permissive interactive mode


class ToolApprovalRule(BaseModel):
    """Legacy rule for approving/denying specific tool calls."""

    name: str
    args: dict[str, Any] | None = None

    def matches_call(self, tool_name: str, tool_args: dict[str, Any]) -> bool:
        """Check if this legacy rule matches a specific tool call."""
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


def _default_allowed_decisions() -> list[DecisionType]:
    """Default allowed HITL decisions."""
    return ["approve", "reject"]


class InterruptOnRule(BaseModel):
    """Per-tool Human-in-the-Loop rule for deepagents."""

    allowed_decisions: list[DecisionType] = Field(
        default_factory=_default_allowed_decisions
    )
    description: str | None = None
    args_schema: dict[str, Any] | None = None


class ToolDecisionRule(BaseModel):
    """Rule for deciding whether a matching tool call should prompt or auto-resolve."""

    name: str
    args: dict[str, Any] | None = None
    decision: ToolDecision = "ask"

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


def _default_interrupt_on_rules() -> dict[str, InterruptOnRule]:
    """Default HITL rules for high-risk tools."""
    return {
        "execute": InterruptOnRule(
            allowed_decisions=["approve", "reject"],
            description="Review shell command execution before running.",
        ),
    }


def _default_interrupt_on_field() -> dict[str, bool | InterruptOnRule]:
    """Return default interrupt_on payload with the field's wider value type."""
    return cast(dict[str, bool | InterruptOnRule], _default_interrupt_on_rules())


def _default_decision_rules() -> list[ToolDecisionRule]:
    """Default fine-grained rules for shell execution approvals."""
    return [
        ToolDecisionRule(
            name="execute",
            args={"command": r"rm\s+-rf.*"},
            decision="ask",
        ),
        ToolDecisionRule(
            name="execute",
            args={"command": r"git\s+push.*"},
            decision="ask",
        ),
        ToolDecisionRule(
            name="execute",
            args={"command": r"git\s+reset\s+--hard.*"},
            decision="ask",
        ),
        ToolDecisionRule(
            name="execute",
            args={"command": r"sudo\s+.*"},
            decision="ask",
        ),
        ToolDecisionRule(
            name="execute",
            args={"command": r".*"},
            decision="always_approve",
        ),
    ]


def _legacy_rule_to_tool_name(rule: ToolApprovalRule) -> str | None:
    """Map legacy rule names into deepagents tool names."""
    name = str(rule.name or "").strip()
    if not name:
        return None

    if name == "run_command":
        return "execute"
    return name


def _legacy_rules_to_interrupt_on(raw: dict[str, Any]) -> dict[str, InterruptOnRule]:
    """Convert legacy allow/deny/ask lists into interrupt_on rules."""
    interrupt_on = _default_interrupt_on_rules()

    legacy_rules: list[ToolApprovalRule] = []
    for key in ("always_ask", "always_deny"):
        candidates = raw.get(key)
        if not isinstance(candidates, list):
            continue
        for item in candidates:
            try:
                legacy_rules.append(ToolApprovalRule.model_validate(item))
            except Exception:
                continue

    for rule in legacy_rules:
        tool_name = _legacy_rule_to_tool_name(rule)
        if not tool_name:
            continue
        interrupt_on[tool_name] = InterruptOnRule(
            allowed_decisions=["approve", "reject"]
        )

    return interrupt_on


def _legacy_rules_to_decision_rules(raw: dict[str, Any]) -> list[ToolDecisionRule]:
    """Convert legacy allow/deny/ask lists into decision_rules."""
    decision_rules: list[ToolDecisionRule] = []
    mappings: list[tuple[str, ToolDecision]] = [
        ("always_deny", "always_reject"),
        ("always_ask", "ask"),
        ("always_allow", "always_approve"),
    ]
    for key, decision in mappings:
        candidates = raw.get(key)
        if not isinstance(candidates, list):
            continue
        for item in candidates:
            try:
                legacy_rule = ToolApprovalRule.model_validate(item)
            except Exception:
                continue

            tool_name = _legacy_rule_to_tool_name(legacy_rule)
            if not tool_name:
                continue
            decision_rules.append(
                ToolDecisionRule(name=tool_name, args=legacy_rule.args, decision=decision)
            )

    if not decision_rules:
        return _default_decision_rules()

    has_execute_fallback = any(
        rule.name == "execute"
        and (rule.args or {}).get("command") == r".*"
        and rule.decision == "always_approve"
        for rule in decision_rules
    )
    if not has_execute_fallback:
        decision_rules.append(
            ToolDecisionRule(
                name="execute",
                args={"command": r".*"},
                decision="always_approve",
            )
        )

    return decision_rules


class ToolApprovalConfig(BaseModel):
    """Configuration for deepagents HITL tool approvals."""

    model_config = ConfigDict(extra="ignore")

    interrupt_on: dict[str, bool | InterruptOnRule] = Field(
        default_factory=_default_interrupt_on_field
    )
    decision_rules: list[ToolDecisionRule] = Field(default_factory=_default_decision_rules)

    @classmethod
    def from_json_file(cls, file_path: Path) -> ToolApprovalConfig:
        """Load configuration from JSON file with legacy migration."""
        if not file_path.exists():
            config = cls()
            config.save_to_json_file(file_path)
            return config

        try:
            with open(file_path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raw = {}

            migrated = False
            if "interrupt_on" not in raw:
                raw["interrupt_on"] = {
                    name: value.model_dump(exclude_none=True)
                    for name, value in _legacy_rules_to_interrupt_on(raw).items()
                }
                migrated = True

            if "decision_rules" not in raw:
                raw["decision_rules"] = [
                    rule.model_dump(exclude_none=True)
                    for rule in _legacy_rules_to_decision_rules(raw)
                ]
                migrated = True

            config = cls.model_validate(raw)
            if migrated:
                config.save_to_json_file(file_path)
            return config
        except Exception:
            return cls()

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

    def prepend_decision_rule(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, Any],
        decision: ToolDecision,
    ) -> None:
        """Persist a high-priority rule for a concrete tool call."""
        normalized_args: dict[str, Any] = {}
        for key, value in (tool_args or {}).items():
            if isinstance(value, str):
                normalized_args[key] = rf"^{re.escape(value)}$"
            else:
                normalized_args[key] = value

        self.decision_rules = [
            rule
            for rule in self.decision_rules
            if not (
                rule.name == tool_name
                and rule.args == normalized_args
                and rule.decision == decision
            )
        ]
        self.decision_rules.insert(
            0,
            ToolDecisionRule(
                name=tool_name,
                args=normalized_args or None,
                decision=decision,
            ),
        )

    def save_to_json_file(self, file_path: Path) -> None:
        """Save configuration to JSON file (interrupt_on + decision_rules)."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "interrupt_on": self.to_interrupt_on_payload() or {},
            "decision_rules": [
                rule.model_dump(exclude_none=True) for rule in self.decision_rules
            ],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
