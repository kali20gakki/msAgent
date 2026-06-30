from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TraceBuilder:
    case_id: str
    prompt: str
    agent: dict[str, Any] = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: str = field(default_factory=utc_now)
    started_perf: float = field(default_factory=perf_counter, repr=False)
    events: list[dict[str, Any]] = field(default_factory=list)
    ended_at: str | None = None
    duration_ms: int | None = None
    token_usage: dict[str, int] = field(default_factory=dict)

    def add(self, event_type: str, **payload: Any) -> None:
        event = {
            "step": len(self.events) + 1,
            "timestamp": utc_now(),
            "type": event_type,
            **payload,
        }
        self.events.append(event)

    def thought(self, content: str) -> None:
        self.add("thought", content=content)

    def tool_call(self, tool: str, tool_input: dict[str, Any]) -> None:
        self.add("tool_call", tool=tool, input=tool_input)

    def tool_result(self, tool: str, output: Any, duration_ms: int | None = None) -> None:
        payload: dict[str, Any] = {"tool": tool, "output": output}
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        self.add("tool_result", **payload)

    def observation(self, content: str) -> None:
        self.add("observation", content=content)

    def final_answer(self, answer: dict[str, Any]) -> None:
        self.add("final_answer", answer=answer)

    def finish(self, token_usage: dict[str, int] | None = None) -> None:
        self.ended_at = utc_now()
        self.duration_ms = round((perf_counter() - self.started_perf) * 1000)
        if token_usage is not None:
            self.token_usage = token_usage

    def to_dict(self) -> dict[str, Any]:
        if self.ended_at is None:
            self.finish()
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "agent": self.agent,
            "prompt": self.prompt,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": self.duration_ms,
            "token_usage": self.token_usage,
            "event_count": len(self.events),
            "events": self.events,
        }
