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

"""Helpers for inspecting OpenAI-compatible request payloads."""

from __future__ import annotations


def content_to_text(content) -> str:
    """Normalize string and content-block message formats."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def extract_system_prompt(payload: dict) -> str:
    """Combine all non-empty system and developer messages."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"LLM 请求缺少 messages: {payload!r}")
    parts = [
        content_to_text(message.get("content"))
        for message in messages
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}
    ]
    prompt = "\n".join(part for part in parts if part.strip())
    if not prompt.strip():
        raise ValueError("LLM 请求中没有非空的 system/developer 消息")
    return prompt


def extract_tool_names(payload: dict) -> list[str]:
    """Extract function names from an OpenAI-compatible tools schema."""
    tools = payload.get("tools", [])
    if not isinstance(tools, list):
        raise ValueError(f"LLM 请求中的 tools 不是列表: {tools!r}")

    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names
