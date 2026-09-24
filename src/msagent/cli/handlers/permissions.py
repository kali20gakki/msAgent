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
            console.print_success("Cleared project approval rules.")
            console.print("")
            return

        if command == "mode" and len(args) == 2:
            mode = args[1].lower()
            if mode in {"safe", "convenience"}:
                self._set_mode(mode)
                console.print_success(f"Execute approval mode switched to {mode}.")
                console.print("")
                return

        console.print_warning("Usage: /permissions | /permissions mode <safe|convenience> | /permissions clear-project")
        console.print("")

    def _set_mode(self, mode: ExecuteApprovalMode) -> None:
        self.session.execute_approval_mode = mode

    def _clear_project_rules(self) -> None:
        ToolApprovalConfig(decision_rules=[]).save_to_json_file(self.interrupt_handler._project_approval_path())

    def _show_permissions(self) -> None:
        mode = getattr(self.session, "execute_approval_mode", None) or "unset"
        project_path = self.interrupt_handler._project_approval_path()
        project_config = ToolApprovalConfig.from_json_file(project_path)
        session_rules = getattr(self.session, "approval_session_rules", None)
        if not isinstance(session_rules, list):
            session_rules = []

        console.print(f"Execute approval mode: {mode}")
        console.print(f"Project approval file: {project_path}")
        self._print_rule_group("Session rules", session_rules)
        self._print_rule_group("Project rules", project_config.decision_rules)
        console.print("Usage: /permissions | /permissions mode <safe|convenience> | /permissions clear-project")
        console.print("")

    @staticmethod
    def _print_rule_group(title: str, rules: list[ToolDecisionRule]) -> None:
        console.print(f"{title}: {len(rules)}")
        for index, rule in enumerate(rules, start=1):
            args = rule.args or {}
            args_text = ", ".join(f"{key}={value}" for key, value in args.items()) if args else "*"
            console.print(f"  {index}. {rule.decision} {rule.name} {args_text}")
