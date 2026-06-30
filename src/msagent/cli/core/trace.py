"""Machine-readable CLI run tracing."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from msagent.middlewares.token_cost import extract_usage_counts
from msagent.utils.render import TOOL_TIMING_RESPONSE_METADATA_KEY


_TEXT_PREVIEW_LIMIT = 4000
_TOOL_OUTPUT_PREVIEW_LIMIT = 4000


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _trim_text(value: str, limit: int = _TEXT_PREVIEW_LIMIT) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}... (truncated, original length: {len(value)})"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return str(value)


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(_json_safe(content), ensure_ascii=False)


def _message_id(message: BaseMessage) -> str | None:
    value = getattr(message, "id", None)
    return str(value) if value else None


def _tool_call_id(tool_call: dict[str, Any]) -> str | None:
    value = tool_call.get("id")
    return str(value) if value else None


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    return str(tool_call.get("name") or "unknown")


def _tool_call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    args = tool_call.get("args")
    return _json_safe(args) if isinstance(args, dict) else {}


class CliRunRecorder:
    """Write structured JSONL events for a CLI session."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.started_perf: float | None = None
        self._event_index = 0
        self._finished = False
        self._seen_usage_keys: set[str] = set()
        self._seen_tool_call_ids: set[str] = set()
        self._seen_tool_result_ids: set[str] = set()
        self._tool_names_by_id: dict[str, str] = {}
        self._input_tokens = 0
        self._output_tokens = 0

    @property
    def token_usage(self) -> dict[str, Any]:
        total = self._input_tokens + self._output_tokens
        return {
            "available": total > 0,
            "source": "msagent-cli-jsonl" if total > 0 else "msagent-cli-jsonl-no-usage-event",
            "input_tokens": self._input_tokens,
            "output_tokens": self._output_tokens,
            "total_tokens": total,
        }

    def start(self, *, context: Any, stream_output: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self.started_perf = time.perf_counter()
        approval_mode = getattr(context, "approval_mode", None)
        self._record(
            "session_started",
            agent=getattr(context, "agent", None),
            model=getattr(context, "model", None),
            model_display=getattr(context, "model_display", None),
            thread_id=getattr(context, "thread_id", None),
            working_dir=str(getattr(context, "working_dir", "")),
            approval_mode=str(getattr(approval_mode, "value", approval_mode or "")),
            stream_output=stream_output,
        )

    def record_error(self, error: BaseException) -> None:
        self._record(
            "error",
            error_type=type(error).__name__,
            message=_trim_text(str(error), 2000),
        )

    def record_assistant_message(self, message: AIMessage, *, origin: str | None = None) -> None:
        self.record_usage_from_message(message)
        text = _message_text(message)
        payload: dict[str, Any] = {
            "message_id": _message_id(message),
            "content": _trim_text(text),
            "content_chars": len(text),
            "content_truncated": len(text) > _TEXT_PREVIEW_LIMIT,
            "tool_call_count": len(getattr(message, "tool_calls", []) or []),
        }
        if origin:
            payload["origin"] = origin
        self._record("assistant_message", **payload)

    def record_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        origin: str | None = None,
    ) -> None:
        item_id = _tool_call_id(tool_call)
        if item_id and item_id in self._seen_tool_call_ids:
            return
        if item_id:
            self._seen_tool_call_ids.add(item_id)

        tool_name = _tool_call_name(tool_call)
        if item_id:
            self._tool_names_by_id[item_id] = tool_name

        payload: dict[str, Any] = {
            "raw_type": "assistant.tool_call",
            "tool": tool_name,
            "item_id": item_id,
            "input": _tool_call_args(tool_call),
        }
        if origin:
            payload["origin"] = origin
        self._record("tool_call", **payload)

    def record_tool_result(
        self,
        message: ToolMessage,
        *,
        tool_call: dict[str, Any] | None = None,
        origin: str | None = None,
    ) -> None:
        item_id = str(getattr(message, "tool_call_id", "") or "")
        message_id = _message_id(message)
        dedupe_id = item_id or message_id
        if dedupe_id and dedupe_id in self._seen_tool_result_ids:
            return
        if dedupe_id:
            self._seen_tool_result_ids.add(dedupe_id)

        text = _message_text(message)
        tool_name = (
            _tool_call_name(tool_call)
            if tool_call is not None
            else self._tool_names_by_id.get(item_id) or str(getattr(message, "name", None) or "tool")
        )
        duration = self._tool_duration(message)
        output: dict[str, Any] = {
            "content": _trim_text(text, _TOOL_OUTPUT_PREVIEW_LIMIT),
            "content_chars": len(text),
            "content_truncated": len(text) > _TOOL_OUTPUT_PREVIEW_LIMIT,
            "is_error": bool(getattr(message, "is_error", False)),
        }
        short_content = getattr(message, "short_content", None)
        if isinstance(short_content, str) and short_content != text:
            output["short_content"] = _trim_text(short_content, 2000)

        payload: dict[str, Any] = {
            "raw_type": "tool.result",
            "tool": tool_name,
            "item_id": item_id or None,
            "message_id": message_id,
            "output": output,
        }
        if duration is not None:
            payload["duration_ms"] = round(duration * 1000)
        if origin:
            payload["origin"] = origin
        self._record("tool_result", **payload)

    def record_usage_from_message(self, message: AIMessage) -> None:
        input_tokens, output_tokens = extract_usage_counts(message)
        if input_tokens is None and output_tokens is None:
            return

        input_value = int(input_tokens or 0)
        output_value = int(output_tokens or 0)
        if input_value <= 0 and output_value <= 0:
            return

        usage_key = _message_id(message)
        if usage_key is None:
            usage_key = f"msg:{input_value}:{output_value}:{_trim_text(_message_text(message), 200)}"
        if usage_key in self._seen_usage_keys:
            return
        self._seen_usage_keys.add(usage_key)

        self._input_tokens += input_value
        self._output_tokens += output_value
        self._record(
            "token_usage",
            raw_type="assistant.usage_metadata",
            message_id=_message_id(message),
            usage={
                "input_tokens": input_value,
                "output_tokens": output_value,
                "total_tokens": input_value + output_value,
            },
            cumulative=self.token_usage,
        )

    def finish(self, *, context: Any, exit_code: int) -> None:
        if self._finished:
            return
        self._finished = True

        usage = self.token_usage
        if not usage["available"]:
            input_tokens = int(getattr(context, "current_input_tokens", None) or 0)
            output_tokens = int(getattr(context, "current_output_tokens", None) or 0)
            if input_tokens > 0 or output_tokens > 0:
                usage = {
                    "available": True,
                    "source": "msagent-cli-context-snapshot",
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }

        self._record(
            "session_finished",
            exit_code=exit_code,
            duration_ms=self.elapsed_ms,
            token_usage=usage,
        )

    @property
    def elapsed_ms(self) -> int:
        if self.started_perf is None:
            return 0
        return round((time.perf_counter() - self.started_perf) * 1000)

    def _record(self, event_type: str, **payload: Any) -> None:
        self._event_index += 1
        event = {
            "index": self._event_index,
            "type": event_type,
            "timestamp": _utc_now(),
            "elapsed_ms": self.elapsed_ms,
            **_json_safe(payload),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")

    @staticmethod
    def _tool_duration(message: ToolMessage) -> float | None:
        response_metadata = getattr(message, "response_metadata", {}) or {}
        timing = response_metadata.get(TOOL_TIMING_RESPONSE_METADATA_KEY)
        if not isinstance(timing, dict):
            return None
        duration = timing.get("duration_seconds")
        if not isinstance(duration, (int, float)):
            return None
        return max(float(duration), 0.0)
