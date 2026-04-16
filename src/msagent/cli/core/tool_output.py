"""State container for expandable tool output previews."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ToolOutputEntry:
    """Tool output that can be expanded in the interactive viewer."""

    tool_call_id: str
    tool_name: str
    preview_content: str
    full_content: str
    indent_level: int = 0
    origin_label: str | None = None
    expanded: bool = False
    duration: float | None = None
    sequence: int = 0

    @property
    def can_expand(self) -> bool:
        """Whether the preview differs from the full output."""
        return self.preview_content != self.full_content
