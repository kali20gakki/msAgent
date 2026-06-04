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

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from msagent.utils import rate_limiter as limiter_module


class _FakeClock:
    def __init__(self, current: float = 100.0) -> None:
        self.current = current

    def time(self) -> float:
        return self.current

    def sleep(self, seconds: float) -> None:
        self.current += seconds


class _FakeLLM:
    def __init__(self) -> None:
        self.sync_calls: list[tuple[list[HumanMessage], object | None, dict[str, object]]] = []
        self.async_calls: list[tuple[list[HumanMessage], object | None, dict[str, object]]] = []

    def _call(self, messages, config=None, **kwargs):
        self.sync_calls.append((messages, config, kwargs))
        return "sync-result"

    async def _acall(self, messages, config=None, **kwargs):
        self.async_calls.append((messages, config, kwargs))
        return "async-result"


def test_token_bucket_limiter_updates_and_consumes_tokens(monkeypatch) -> None:
    fake_clock = _FakeClock()
    monkeypatch.setattr(limiter_module.time, "time", fake_clock.time)

    limiter = limiter_module.TokenBucketLimiter(2.0, 50.0, 40.0, max_bucket_size=10)
    limiter.request_bucket = 0.0
    limiter.input_token_bucket = 0.0
    limiter.output_token_bucket = 0.0
    limiter.last_update_time = 100.0

    fake_clock.current = 101.0
    limiter._update_buckets()
    assert limiter.request_bucket == 2
    assert limiter.input_token_bucket == 50
    assert limiter.output_token_bucket == 40

    logged: list[str] = []
    monkeypatch.setattr(limiter_module.logger, "info", logged.append)
    limiter.request_bucket = 1.0
    limiter.input_token_bucket = 1000.0
    limiter.output_token_bucket = 1000.0
    limiter.last_update_time = fake_clock.current

    assert limiter._consume(input_tokens=600, output_tokens=700) is True
    assert limiter.request_bucket == 0.0
    assert limiter.input_token_bucket == 400.0
    assert limiter.output_token_bucket == 300.0
    assert list(limiter.recent_input_tokens) == [600]
    assert list(limiter.recent_output_tokens) == [700]
    assert logged and "Rate usage:" in logged[0]


def test_token_bucket_limiter_acquire_variants(monkeypatch) -> None:
    limiter = limiter_module.TokenBucketLimiter(1.0, 10.0, 10.0)

    states = iter([False, True])
    monkeypatch.setattr(limiter, "_consume", lambda *args, **kwargs: next(states))
    sleeps: list[float] = []
    monkeypatch.setattr(limiter_module.time, "sleep", sleeps.append)

    assert limiter.acquire(blocking=True) is True
    assert sleeps == [limiter.check_every_n_seconds]

    monkeypatch.setattr(limiter, "_consume", lambda *args, **kwargs: False)
    assert limiter.acquire(blocking=False) is False


@pytest.mark.asyncio
async def test_token_bucket_limiter_async_acquire_and_async_call(monkeypatch) -> None:
    limiter = limiter_module.TokenBucketLimiter(1.0, 10.0, 10.0)

    states = iter([False, True])
    monkeypatch.setattr(limiter, "_consume", lambda *args, **kwargs: next(states))
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    import asyncio

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    assert await limiter.aacquire(blocking=True) is True
    assert sleeps == [limiter.check_every_n_seconds]

    monkeypatch.setattr(limiter, "_consume", lambda *args, **kwargs: True)
    assert await limiter.aacquire(blocking=False) is True

    llm = _FakeLLM()
    consumed: list[int] = []
    monkeypatch.setattr(
        limiter, "_consume", lambda input_tokens=0, output_tokens=0: consumed.append(input_tokens) or True
    )

    result = await limiter._acall(
        llm,
        [HumanMessage(content="abcdefgh")],
        config=SimpleNamespace(name="cfg"),
        stream=True,
    )

    assert result == "async-result"
    assert consumed == [2]
    assert llm.async_calls[0][2] == {"stream": True}


def test_token_bucket_limiter_call_estimates_tokens_and_invokes_llm(monkeypatch) -> None:
    limiter = limiter_module.TokenBucketLimiter(1.0, 10.0, 10.0)
    llm = _FakeLLM()
    consumed: list[int] = []
    monkeypatch.setattr(
        limiter, "_consume", lambda input_tokens=0, output_tokens=0: consumed.append(input_tokens) or True
    )
    monkeypatch.setattr(limiter_module.time, "sleep", lambda _seconds: None)

    result = limiter(
        llm,
        [HumanMessage(content="abcd"), HumanMessage(content="efgh")],
        config=SimpleNamespace(name="cfg"),
        temperature=0.1,
    )

    assert result == "sync-result"
    assert consumed == [2]
    assert llm.sync_calls[0][2] == {"temperature": 0.1}
