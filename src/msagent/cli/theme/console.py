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

"""Rich styles and formatting utilities with theme support."""

import os
from typing import Literal

from rich.console import Console

from msagent.cli.theme.base import BaseTheme


class ThemedConsole:
    """Console wrapper with configurable theme."""

    def __init__(self, console_theme: BaseTheme):
        color_system: Literal["truecolor"] | None = None
        if (
            os.getenv("COLORTERM", "").lower() in {"truecolor", "24bit"}
            or os.getenv("WT_SESSION")
            or os.getenv("TERM_PROGRAM") in {"vscode", "WarpTerminal"}
            or os.getenv("WSL_DISTRO_NAME")
        ):
            color_system = "truecolor"

        self.console = Console(
            theme=console_theme.rich_theme,
            force_terminal=True,
            color_system=color_system,
        )

    def print(self, *args, style: str = "default", **kwargs):
        """Print with theme-aware styling."""
        self.console.print(*args, style=style, **kwargs)

    def print_error(self, content: str):
        """Print error message."""
        self.console.print(f"[error]\u274c[/error] {content}")

    def print_warning(self, content: str):
        """Print warning message."""
        self.console.print(f"[warning]\u26a0\ufe0f[/warning] {content}")

    def print_success(self, content: str):
        """Print success message."""
        self.console.print(f"[success]\u2705[/success] {content}")

    def clear(self):
        """Clear the console."""
        self.console.clear()

    def capture(self):
        """Capture console output for measuring rendered lines."""
        return self.console.capture()

    @property
    def width(self):
        """Get console width."""
        return self.console.width
