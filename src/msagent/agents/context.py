from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from msagent.configs import ApprovalMode
from msagent.skills.factory import Skill


@dataclass(frozen=True, slots=True)
class RetryNotice:
    """Runtime-only retry notice emitted for TUI feedback."""

    notice_id: str
    scope: Literal["llm", "tool"]
    attempt: int
    max_retries: int
    delay: float
    target_name: str | None = None
    phase: Literal["scheduled", "cleared"] = "scheduled"


class AgentContext(BaseModel):
    approval_mode: ApprovalMode
    working_dir: Path
    platform: str = Field(default="")
    os_version: str = Field(default="")
    current_date_time_zoned: str = Field(default="")
    mcp_servers: str = Field(default="")
    user_memory: str = Field(default="")
    tool_catalog: list[BaseTool] = Field(default_factory=list, exclude=True)
    skill_catalog: list[Skill] = Field(default_factory=list, exclude=True)
    tool_output_max_tokens: int | None = None
    retry_notice_handler: (
        Callable[[RetryNotice], None | Awaitable[None]] | None
    ) = Field(default=None, exclude=True)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def template_vars(self) -> dict[str, Any]:
        return {
            "working_dir": str(self.working_dir),
            "platform": self.platform,
            "os_version": self.os_version,
            "current_date_time_zoned": self.current_date_time_zoned,
            "mcp_servers": self.mcp_servers,
            "user_memory": self.user_memory,
        }
