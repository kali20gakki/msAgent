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

"""State container for expandable tool output previews."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolOutputEntry:
    """Tool output that can be expanded in the interactive viewer."""

    tool_call_id: str
    tool_name: str
    preview_content: str
    full_content: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    indent_level: int = 0
    origin_label: str | None = None
    expanded: bool = False
    duration: float | None = None
    sequence: int = 0

    @property
    def can_expand(self) -> bool:
        """Whether the preview differs from the full output."""
        return self.preview_content != self.full_content
