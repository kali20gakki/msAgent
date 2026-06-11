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

"""Future internal binding script executor."""

from __future__ import annotations


class InternalScriptExecutor:
    def __init__(self, script_path: str | None = None) -> None:
        self.script_path = script_path

    def preview(self, plan: dict) -> list[str]:
        return [action["apply_command"] for action in plan.get("apply_actions", [])]

    def apply(self, plan: dict) -> dict:
        raise NotImplementedError("internal script executor is intentionally not enabled in the MVP")

    def rollback(self, rollback_state: dict) -> dict:
        raise NotImplementedError("internal script rollback is intentionally not enabled in the MVP")
