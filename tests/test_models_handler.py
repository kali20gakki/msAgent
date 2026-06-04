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

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.cli.handlers import models as models_module
from msagent.cli.handlers.models import ModelHandler
from msagent.configs import LLMConfig, LLMProvider


def _build_llm_config(alias: str, provider: LLMProvider = LLMProvider.OPENAI) -> LLMConfig:
    return LLMConfig.model_construct(
        alias=alias,
        provider=provider,
        model="gpt-4o-mini",
        context_window=128000,
    )


def _build_session(tmp_path: Path, *, model: str = "default") -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(
            agent="Profiler",
            working_dir=tmp_path,
            model=model,
            thread_id="thread-1",
        ),
        update_context=lambda **kwargs: None,
    )


def test_model_handler_groups_models_by_provider() -> None:
    models = [
        _build_llm_config("default", LLMProvider.OPENAI),
        _build_llm_config("fast", LLMProvider.OPENAI),
        _build_llm_config("gemini-pro", LLMProvider.GOOGLE),
    ]

    grouped = ModelHandler._group_models_by_provider(models)

    assert "openai" in grouped
    assert "google" in grouped
    assert len(grouped["openai"]) == 2
    assert len(grouped["google"]) == 1


def test_model_handler_format_tabbed_model_list_shows_current_and_default_markers() -> None:
    models = [
        _build_llm_config("default", LLMProvider.OPENAI),
        _build_llm_config("fast", LLMProvider.OPENAI),
    ]

    handler = ModelHandler(_build_session(Path.cwd(), model="default"))
    providers = handler._group_models_by_provider(models)
    provider_names = list(providers.keys())

    formatted = handler._format_tabbed_model_list(
        providers,
        provider_names,
        selected_provider_idx=0,
        selected_model_idx=0,
        current_model="default",
        default_model="default",
    )

    text = "".join(fragment[1] for fragment in formatted)
    assert "[current]" in text
    assert "[default]" in text


def test_model_handler_format_tabbed_model_list_shows_provider_tabs() -> None:
    models = [
        _build_llm_config("gpt-4o", LLMProvider.OPENAI),
        _build_llm_config("gemini-pro", LLMProvider.GOOGLE),
    ]

    handler = ModelHandler(_build_session(Path.cwd(), model="gpt-4o"))
    providers = handler._group_models_by_provider(models)
    provider_names = list(providers.keys())

    formatted = handler._format_tabbed_model_list(
        providers,
        provider_names,
        selected_provider_idx=0,
        selected_model_idx=0,
        current_model="gpt-4o",
        default_model="gpt-4o",
    )

    text = "".join(fragment[1] for fragment in formatted)
    assert "openai" in text
    assert "google" in text


@pytest.mark.asyncio
async def test_model_handler_reports_no_other_models_when_only_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(models_module.console, "print_error", errors.append)
    monkeypatch.setattr(models_module.console, "print", lambda *_args, **_kwargs: None)

    agent_config = SimpleNamespace(llm=SimpleNamespace(alias="default"))
    config_data = SimpleNamespace(llms=[_build_llm_config("default")])

    async def fake_load_agent_config(_agent, _working_dir):
        return agent_config

    async def fake_load_llms_config(_working_dir):
        return config_data

    monkeypatch.setattr(models_module.initializer, "load_agent_config", fake_load_agent_config)
    monkeypatch.setattr(models_module.initializer, "load_llms_config", fake_load_llms_config)

    handler = ModelHandler(_build_session(tmp_path))
    await handler.handle()

    assert "No other models available" in errors


@pytest.mark.asyncio
async def test_model_handler_updates_context_on_successful_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    new_model = _build_llm_config("fast", LLMProvider.OPENAI)
    default_model = _build_llm_config("default", LLMProvider.OPENAI)
    agent_config = SimpleNamespace(llm=SimpleNamespace(alias="default"))
    config_data = SimpleNamespace(llms=[default_model, new_model])

    context_updates: dict[str, object] = {}
    session = SimpleNamespace(
        context=SimpleNamespace(
            agent="Profiler",
            working_dir=tmp_path,
            model="default",
            thread_id="thread-1",
        ),
        update_context=lambda **kwargs: context_updates.update(kwargs),
    )

    async def fake_load_agent_config(_agent, _working_dir):
        return agent_config

    async def fake_load_llms_config(_working_dir):
        return config_data

    async def fake_update_agent_llm(_agent, _model, _working_dir):
        pass

    async def fake_get_model_selection(_models, _current, _default):
        return "fast"

    monkeypatch.setattr(models_module.initializer, "load_agent_config", fake_load_agent_config)
    monkeypatch.setattr(models_module.initializer, "load_llms_config", fake_load_llms_config)
    monkeypatch.setattr(models_module.initializer, "update_agent_llm", fake_update_agent_llm)

    handler = ModelHandler(session)
    monkeypatch.setattr(handler, "_get_model_selection", fake_get_model_selection)

    await handler.handle()

    assert context_updates.get("model") == "fast"


@pytest.mark.asyncio
async def test_model_handler_skips_update_when_selection_canceled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    default_model = _build_llm_config("default", LLMProvider.OPENAI)
    new_model = _build_llm_config("fast", LLMProvider.OPENAI)
    agent_config = SimpleNamespace(llm=SimpleNamespace(alias="default"))
    config_data = SimpleNamespace(llms=[default_model, new_model])

    context_updates: dict[str, object] = {}
    session = SimpleNamespace(
        context=SimpleNamespace(
            agent="Profiler",
            working_dir=tmp_path,
            model="default",
            thread_id="thread-1",
        ),
        update_context=lambda **kwargs: context_updates.update(kwargs),
    )

    async def fake_load_agent_config(_agent, _working_dir):
        return agent_config

    async def fake_load_llms_config(_working_dir):
        return config_data

    async def fake_get_model_selection(_models, _current, _default):
        return ""

    monkeypatch.setattr(models_module.initializer, "load_agent_config", fake_load_agent_config)
    monkeypatch.setattr(models_module.initializer, "load_llms_config", fake_load_llms_config)

    handler = ModelHandler(session)
    monkeypatch.setattr(handler, "_get_model_selection", fake_get_model_selection)

    await handler.handle()

    assert context_updates == {}


@pytest.mark.asyncio
async def test_model_handler_handles_exception_gracefully(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(models_module.console, "print_error", errors.append)
    monkeypatch.setattr(models_module.console, "print", lambda *_args, **_kwargs: None)

    async def fake_load_agent_config(_agent, _working_dir):
        raise RuntimeError("config load failed")

    monkeypatch.setattr(models_module.initializer, "load_agent_config", fake_load_agent_config)

    handler = ModelHandler(_build_session(tmp_path))
    await handler.handle()

    assert any("Error switching models" in e for e in errors)
