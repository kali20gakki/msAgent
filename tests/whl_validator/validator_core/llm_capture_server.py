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

"""A lightweight OpenAI-compatible server for deterministic validation tests."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True)
class ToolCallScript:
    """Describe one deterministic tool call followed by a final answer.

    ``tool_name`` is the raw MCP function name, such as ``analyze_overlap``.
    When ``server_name`` is provided, the responder resolves the actual name
    advertised by msagent (for example ``msprof-mcp__analyze_overlap``). This
    keeps tests compatible with both separators supported by msagent.
    """

    tool_name: str
    arguments: dict[str, Any]
    server_name: str | None = None
    final_content: str = "MOCK_TOOL_CALL_COMPLETED"

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if self.server_name is not None and (not isinstance(self.server_name, str) or not self.server_name.strip()):
            raise ValueError("server_name must be a non-empty string or None")
        if not isinstance(self.arguments, dict):
            raise TypeError("arguments must be a dictionary")
        if not isinstance(self.final_content, str) or not self.final_content:
            raise ValueError("final_content must be a non-empty string")


def _extract_tool_names(request: dict) -> list[str]:
    """Extract OpenAI function names without depending on pytest helpers."""
    tools = request.get("tools", [])
    if not isinstance(tools, list):
        return []

    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _matches_scripted_tool(
    advertised_name: str,
    *,
    server_name: str | None,
    raw_tool_name: str,
) -> bool:
    if server_name is None:
        return advertised_name == raw_tool_name
    return advertised_name in {
        f"{server_name}__{raw_tool_name}",
        f"{server_name}_{raw_tool_name}",
    }


@dataclass
class _ResponseState:
    """Thread-safe response state shared by all HTTP handler instances."""

    requests: list[dict]
    script: ToolCallScript | None
    errors: list[str]
    _response_index: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def next_response(self, request: dict) -> tuple[dict, str]:
        """Return ``(assistant_message, finish_reason)`` for this request."""
        with self._lock:
            response_index = self._response_index
            self._response_index += 1

        if self.script is None:
            return {
                "role": "assistant",
                "content": "SYS_PROMPT_CAPTURE_OK",
            }, "stop"

        # The first turn deterministically requests the real tool. Subsequent
        # turns close the agent loop after msagent has submitted the result.
        if response_index == 0:
            advertised_names = _extract_tool_names(request)
            matching_names = [
                name
                for name in advertised_names
                if _matches_scripted_tool(
                    name,
                    server_name=self.script.server_name,
                    raw_tool_name=self.script.tool_name,
                )
            ]
            if len(matching_names) != 1:
                error = (
                    "expected exactly one advertised tool matching "
                    f"{self.script.server_name!r}/{self.script.tool_name!r}, "
                    f"found {matching_names!r}; advertised tools: {advertised_names!r}"
                )
                self.errors.append(error)
                # A normal assistant response lets pytest report the missing
                # tool through trace assertions instead of an opaque HTTP retry.
                return {"role": "assistant", "content": f"MOCK_ERROR: {error}"}, "stop"

            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-msagent-validator-1",
                        "type": "function",
                        "function": {
                            "name": matching_names[0],
                            "arguments": json.dumps(
                                self.script.arguments,
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }, "tool_calls"

        messages = request.get("messages")
        has_tool_result = isinstance(messages, list) and any(
            isinstance(message, dict) and message.get("role") == "tool" for message in messages
        )
        if not has_tool_result:
            error = "the follow-up LLM request did not contain a tool result message"
            self.errors.append(error)
            return {"role": "assistant", "content": f"MOCK_ERROR: {error}"}, "stop"

        return {
            "role": "assistant",
            "content": self.script.final_content,
        }, "stop"


def _completion_chunks(
    *,
    message: dict,
    finish_reason: str,
    model: str,
    created: int,
) -> list[dict]:
    """Convert a complete mock message into OpenAI-compatible SSE chunks."""
    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": index, **tool_call} for index, tool_call in enumerate(message["tool_calls"])]
    else:
        delta["content"] = str(message.get("content") or "")

    base = {
        "id": "chatcmpl-msagent-capture",
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
    }
    return [
        {
            **base,
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        },
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        },
        {
            **base,
            "choices": [],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        },
    ]


def _handler(state: _ResponseState):
    class CaptureHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.send_error(400, f"invalid JSON request: {exc}")
                return
            if not isinstance(payload, dict):
                self.send_error(400, "request body must be a JSON object")
                return

            # Never persist headers: even a local test may receive an API key.
            state.requests.append({"path": self.path, "body": payload})
            self._send_response(payload)

        def _send_response(self, request: dict) -> None:
            created = int(time.time())
            model = str(request.get("model") or "capture-model")
            message, finish_reason = state.next_response(request)

            if request.get("stream"):
                chunks = _completion_chunks(
                    message=message,
                    finish_reason=finish_reason,
                    model=model,
                    created=created,
                )
                body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                content_type = "text/event-stream"
            else:
                body = json.dumps(
                    {
                        "id": "chatcmpl-msagent-capture",
                        "object": "chat.completion",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 3,
                            "total_tokens": 13,
                        },
                    }
                )
                content_type = "application/json"

            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)
            self.close_connection = True

        def log_message(self, format: str, *args) -> None:
            return

    return CaptureHandler


@dataclass
class LlmCaptureServer:
    """Running local endpoint and the JSON request bodies it received."""

    base_url: str
    requests: list[dict]
    errors: list[str]
    _server: ThreadingHTTPServer = field(repr=False)
    _thread: threading.Thread = field(repr=False)

    @classmethod
    def start(cls, script: ToolCallScript | None = None) -> LlmCaptureServer:
        requests: list[dict] = []
        errors: list[str] = []
        state = _ResponseState(requests=requests, script=script, errors=errors)
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            name="msagent-llm-capture-server",
            daemon=True,
        )
        thread.start()
        host, port = server.server_address[:2]
        return cls(
            base_url=f"http://{host}:{port}/v1",
            requests=requests,
            errors=errors,
            _server=server,
            _thread=thread,
        )

    @property
    def latest_request(self) -> dict:
        if not self.requests:
            raise RuntimeError("本地 LLM 捕获服务没有收到请求")
        return self.requests[-1]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
