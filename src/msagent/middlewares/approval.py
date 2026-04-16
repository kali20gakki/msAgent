#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Approval middleware for human-in-the-loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class InterruptPayload:
    """Payload for interrupt requests."""

    question: str
    options: list[str]
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None


class ApprovalMiddleware:
    """Middleware for handling tool call approvals (HIL)."""

    def __init__(self, enabled: bool = True):
        """Initialize approval middleware.

        Args:
            enabled: Whether approval is enabled
        """
        self.enabled = enabled
