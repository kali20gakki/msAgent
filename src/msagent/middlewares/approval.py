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
