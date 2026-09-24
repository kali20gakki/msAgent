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

"""Handlers for explicit memory commands."""

from __future__ import annotations

import asyncio

from rich.panel import Panel

from msagent.cli.theme import console
from msagent.tools.internal.memory import (
    append_memory_entry,
    ensure_memory_file,
    is_default_memory_content,
    read_memory_file,
)


class MemoryHandler:
    """Handle explicit memory read/write commands."""

    def __init__(self, session) -> None:
        self.session = session

    async def remember(self, args: list[str]) -> None:
        """Store durable memory via `/remember <content>`."""
        content = " ".join(args).strip()
        if not content:
            console.print_warning("Usage: /remember <content>")
            console.print("")
            return

        context = self.session.context
        memory_path = await asyncio.to_thread(
            append_memory_entry,
            content,
            context.working_dir,
            state_dir=getattr(context, "state_dir", None),
        )
        console.print_success(f"Saved memory to {memory_path}")
        console.print("")

    async def show(self, args: list[str]) -> None:
        """Show current project memory."""
        _ = args
        context = self.session.context
        state_dir = getattr(context, "state_dir", None)
        await asyncio.to_thread(
            ensure_memory_file,
            context.working_dir,
            state_dir=state_dir,
        )
        content = await asyncio.to_thread(
            read_memory_file,
            context.working_dir,
            state_dir=state_dir,
        )
        content = content.strip()
        if not content or is_default_memory_content(content):
            console.print_warning("No memory saved yet")
            console.print("")
            return

        console.console.print(
            Panel(
                content,
                title="[accent]Memory[/accent]",
                border_style="border",
                padding=(1, 2),
            )
        )
        console.print("")
