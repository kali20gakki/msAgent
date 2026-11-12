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

"""Handlers for shell approval mode and project approval rules."""

from __future__ import annotations

from typing import cast

from msagent.cli.handlers.interrupts import InterruptHandler
from msagent.cli.theme import console
from msagent.configs import ExecuteApprovalMode, ToolApprovalConfig, ToolDecisionRule


class PermissionsHandler:
    """Manage execute approval mode and persisted project rules."""

    def __init__(self, session) -> None:
        self.session = session
        self.interrupt_handler = InterruptHandler(session)

    async def handle(self, args: list[str]) -> None:
        """Handle `/permissions` commands."""
        if not args:
            self._show_permissions()
            return

        command = args[0].lower()
        if command == "clear-project":
            self._clear_project_rules()
            console.print_success("Cleared project approval rules. The approval mode is retained.")
            console.print("")
            return

        if command == "remove" and len(args) == 2:
            removed = ToolApprovalConfig.remove_rule_from_json_file(
                self.interrupt_handler._project_approval_path(), args[1]
            )
            if removed:
                console.print_success(f"Removed project approval rule {args[1]}.")
            else:
                console.print_warning(f"Project approval rule not found: {args[1]}")
            console.print("")
            return

        if command == "explain" and len(args) >= 2:
            self._explain_command(" ".join(args[1:]))
            return

        if command == "mode" and len(args) == 2:
            mode = args[1].lower()
            if mode in {"manual", "auto"}:
                self._set_mode(cast(ExecuteApprovalMode, mode))
                console.print_success(f"Execute approval mode switched to {mode}.")
                console.print("")
                return

        console.print_warning(self._usage())
        console.print("")

    def _set_mode(self, mode: ExecuteApprovalMode) -> None:
        self.session.execute_approval_mode = mode
        self.session.execute_approval_mode_source = "project"
        ToolApprovalConfig.update_mode_in_json_file(self.interrupt_handler._project_approval_path(), mode)

    def _clear_project_rules(self) -> None:
        project_path = self.interrupt_handler._project_approval_path()
        config = ToolApprovalConfig.from_json_file(project_path)
        config.decision_rules = []
        config.family_rules = []
        config.script_rules = []
        config.external_directories = []
        config.save_to_json_file(project_path)

    def _show_permissions(self) -> None:
        project_path = self.interrupt_handler._project_approval_path()
        project_config = ToolApprovalConfig.from_json_file(project_path)
        runtime_mode = getattr(self.session, "execute_approval_mode", None)
        mode = runtime_mode or project_config.execute_approval_mode or "unset"
        source = getattr(self.session, "execute_approval_mode_source", None)
        if source == "cli":
            mode_source = "CLI override"
        elif runtime_mode or project_config.execute_approval_mode:
            mode_source = "project"
        else:
            mode_source = "unset"
        console.print(f"Execute approval mode: {mode} ({mode_source})")
        console.print(f"Project approval file: {project_path}")
        self._print_rule_group("Project rules", project_config.decision_rules)
        console.print(f"Project family rules: {len(project_config.family_rules)}")
        for index, family_rule in enumerate(project_config.family_rules, start=1):
            console.print(f"  {index}. [{family_rule.id}] {family_rule.decision} {family_rule.family}")
        console.print(f"Project script rules: {len(project_config.script_rules)}")
        for index, script_rule in enumerate(project_config.script_rules, start=1):
            console.print(
                f"  {index}. [{script_rule.id}] {script_rule.decision} {script_rule.interpreter} {script_rule.path}"
            )
        console.print(f"External directories: {len(project_config.external_directories)}")
        for index, external_rule in enumerate(project_config.external_directories, start=1):
            console.print(f"  {index}. [{external_rule.id}] allow {external_rule.path} (recursive)")
        console.print(self._usage())
        console.print("")

    @staticmethod
    def _print_rule_group(title: str, rules: list[ToolDecisionRule]) -> None:
        console.print(f"{title}: {len(rules)}")
        for index, rule in enumerate(rules, start=1):
            args = rule.args or {}
            args_text = ", ".join(f"{key}={value}" for key, value in args.items()) if args else "*"
            console.print(f"  {index}. [{rule.id}] {rule.decision} {rule.name} {args_text}")

    def _explain_command(self, command: str) -> None:
        """Explain command policy without executing or changing project state."""
        config = ToolApprovalConfig.from_json_file(self.interrupt_handler._project_approval_path())
        args = {"command": command}
        profile = self.interrupt_handler._classify_command(tool_name="execute", tool_args=args)
        decision = self.interrupt_handler._resolve_decision(
            approval_config=config,
            tool_name="execute",
            tool_args=args,
            command_profile=profile,
        )
        console.print(f"Category: {profile['category']}")
        if profile.get("family"):
            console.print(f"Family: {profile['family']}")
        if profile.get("script_path"):
            console.print(f"Script: {profile['script_interpreter']} {profile['script_path']}")
        console.print(f"Decision: {decision}")
        matched_id = None
        if profile.get("family"):
            matched_id = next(
                (rule.id for rule in config.family_rules if rule.family == profile["family"]),
                None,
            )
        elif profile.get("script_path"):
            matched_id = next(
                (
                    rule.id
                    for rule in config.script_rules
                    if rule.interpreter == profile["script_interpreter"] and rule.path == profile["script_path"]
                ),
                None,
            )
        elif profile["category"] == "ordinary":
            matched_id = next(
                (rule.id for rule in config.decision_rules if rule.matches_call("execute", args)),
                None,
            )
        console.print(f"Matched rule: {matched_id or 'none'}")
        console.print("")

    @staticmethod
    def _usage() -> str:
        return (
            "Usage: /permissions | /permissions mode <manual|auto> | "
            "/permissions remove <rule-id> | /permissions clear-project | "
            "/permissions explain <command>"
        )
