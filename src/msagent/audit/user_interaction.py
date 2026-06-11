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

"""Build audit payloads for user turn and interrupt response events."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

MAX_AUDIT_TEXT_LENGTH = 1000


def extract_last_agent_prompt(messages: list[AnyMessage]) -> str | None:
    """Return the latest main-agent assistant text before a new user turn."""
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        text = _extract_message_text(message)
        if text:
            return text
    return None


def _extract_message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        text = "".join(parts).strip()
        if text:
            return text

    short_content = getattr(message, "short_content", None)
    if isinstance(short_content, str) and short_content.strip():
        return short_content.strip()
    return str(content or "").strip()


def truncate_audit_text(value: str | None, *, max_length: int = MAX_AUDIT_TEXT_LENGTH) -> str:
    """Truncate free-text audit fields."""
    if not value:
        return ""
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def build_user_response_fields(interrupt: Any, resume_value: Any) -> dict[str, Any] | None:
    """Map one interrupt resume value to ``user.response`` audit fields."""
    value = getattr(interrupt, "value", None)
    interrupt_id = getattr(interrupt, "id", None)

    if isinstance(value, dict) and "action_requests" in value and "review_configs" in value:
        return _build_approval_response(
            value,
            resume_value,
            interrupt_id=str(interrupt_id) if interrupt_id else None,
        )

    if isinstance(value, dict) and "question" in value and "options" in value:
        return _build_choice_response(
            value,
            resume_value,
            interrupt_id=str(interrupt_id) if interrupt_id else None,
        )

    return None


def _build_approval_response(
    interrupt_value: dict[str, Any],
    resume_value: Any,
    *,
    interrupt_id: str | None,
) -> dict[str, Any]:
    actions = list(interrupt_value.get("action_requests") or [])
    decisions = []
    if isinstance(resume_value, dict):
        raw_decisions = resume_value.get("decisions")
        if isinstance(raw_decisions, list):
            decisions = raw_decisions

    response_items: list[dict[str, str]] = []
    prompt_parts: list[str] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        tool_name = str(action.get("name") or "")
        tool_args = action.get("args") if isinstance(action.get("args"), dict) else {}
        description = str(action.get("description") or "").strip()
        if description:
            prompt_parts.append(description)
        else:
            prompt_parts.append(f"Tool: {tool_name}")

        decision = decisions[index] if index < len(decisions) else {}
        decision_type = decision.get("type", "unknown") if isinstance(decision, dict) else "unknown"
        response_items.append({"tool_name": tool_name, "decision": str(decision_type)})
        if tool_name:
            prompt_parts[-1] = _append_tool_args_summary(prompt_parts[-1], tool_args)

    if len(response_items) == 1:
        response: Any = response_items[0]["decision"]
    elif response_items:
        response = response_items
    else:
        response = resume_value

    context: dict[str, Any] = {}
    if interrupt_id:
        context["interrupt_id"] = interrupt_id
    if len(response_items) > 1:
        context["batch"] = True
    if len(response_items) == 1 and response_items[0]["tool_name"]:
        context["tool_name"] = response_items[0]["tool_name"]

    return {
        "kind": "approval",
        "prompt": "\n".join(prompt_parts) or "Tool execution requires approval.",
        "options": ["approve", "reject", "always_approve", "always_reject"],
        "response": response,
        "context": context or None,
    }


def _build_choice_response(
    interrupt_value: dict[str, Any],
    resume_value: Any,
    *,
    interrupt_id: str | None,
) -> dict[str, Any]:
    options = [str(option) for option in (interrupt_value.get("options") or [])]
    context = {"interrupt_id": interrupt_id} if interrupt_id else None
    return {
        "kind": "choice",
        "prompt": str(interrupt_value.get("question") or "Approval required"),
        "options": options or None,
        "response": str(resume_value),
        "context": context,
    }


def _append_tool_args_summary(prompt: str, tool_args: dict[str, Any]) -> str:
    if not tool_args:
        return prompt
    args_text = json.dumps(tool_args, ensure_ascii=False)
    return f"{prompt}\nArgs: {args_text}"
