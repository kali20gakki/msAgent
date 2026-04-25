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

"""Compatibility patches for langchain_openai."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk

_MessageT = TypeVar("_MessageT", bound=BaseMessage | BaseMessageChunk)


def patch_chat_openai_reasoning_content_support() -> None:
    """Preserve DeepSeek-style reasoning_content through ChatOpenAI adapters."""
    from langchain_openai.chat_models import base as openai_base

    if getattr(openai_base, "_msagent_reasoning_content_patch_applied", False):
        return

    original_convert_message_to_dict = openai_base._convert_message_to_dict
    original_convert_dict_to_message = openai_base._convert_dict_to_message
    original_convert_delta_to_message_chunk = openai_base._convert_delta_to_message_chunk

    def _extract_reasoning_content(payload: Mapping[str, Any]) -> str | None:
        value = payload.get("reasoning_content")
        return value if isinstance(value, str) else None

    def _attach_reasoning_content_to_message(
        message: _MessageT,
        reasoning_content: str | None,
    ) -> _MessageT:
        if reasoning_content is None or not isinstance(message, (AIMessage, AIMessageChunk)):
            return message
        additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
        additional_kwargs["reasoning_content"] = reasoning_content
        return cast(_MessageT, message.model_copy(update={"additional_kwargs": additional_kwargs}))

    def _convert_message_to_dict(
        message: BaseMessage,
        api: str = "chat/completions",
    ) -> dict[str, Any]:
        message_dict = original_convert_message_to_dict(message, api=api)
        if api == "chat/completions" and isinstance(message, AIMessage):
            reasoning_content = _extract_reasoning_content(getattr(message, "additional_kwargs", {}) or {})
            if reasoning_content is not None:
                message_dict["reasoning_content"] = reasoning_content
        return message_dict

    def _convert_dict_to_message(payload: Mapping[str, Any]) -> BaseMessage:
        message = original_convert_dict_to_message(payload)
        return _attach_reasoning_content_to_message(
            message,
            _extract_reasoning_content(payload),
        )

    def _convert_delta_to_message_chunk(
        payload: Mapping[str, Any],
        default_class: type[BaseMessageChunk],
    ) -> BaseMessageChunk:
        chunk = original_convert_delta_to_message_chunk(payload, default_class)
        return _attach_reasoning_content_to_message(
            chunk,
            _extract_reasoning_content(payload),
        )

    openai_base._convert_message_to_dict = _convert_message_to_dict
    openai_base._convert_dict_to_message = _convert_dict_to_message
    openai_base._convert_delta_to_message_chunk = _convert_delta_to_message_chunk
    openai_base._msagent_reasoning_content_patch_applied = True
