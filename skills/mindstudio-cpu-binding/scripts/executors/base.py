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

"""Executor interface for affinity actions."""

from __future__ import annotations

from typing import Protocol


class BindingExecutor(Protocol):
    def preview(self, plan: dict) -> list[str]:
        """Return commands that would be executed."""

    def apply(self, plan: dict) -> dict:
        """Apply a plan."""

    def rollback(self, rollback_state: dict) -> dict:
        """Rollback applied actions."""
