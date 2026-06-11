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

"""Dry-run executor."""

from __future__ import annotations


class DryRunExecutor:
    def preview(self, plan: dict) -> list[str]:
        return [action["apply_command"] for action in plan.get("apply_actions", [])]

    def apply(self, plan: dict) -> dict:
        return {
            "executor_backend": "dry-run",
            "status": "preview_only",
            "commands": self.preview(plan),
            "message": "dry-run executor does not modify system state",
        }

    def rollback(self, rollback_state: dict) -> dict:
        commands = [action.get("rollback_command") for action in rollback_state.get("actions", [])]
        return {
            "executor_backend": "dry-run",
            "status": "preview_only",
            "commands": [command for command in commands if command],
            "message": "dry-run executor does not modify system state",
        }
