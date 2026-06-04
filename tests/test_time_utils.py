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

from datetime import datetime, timezone

from msagent.utils import time as time_module


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 1, 1, 0, 0, 0, tzinfo=tz)


def test_format_relative_time_supports_seconds_minutes_hours_days_months_and_years(monkeypatch) -> None:
    monkeypatch.setattr(time_module, "datetime", _FrozenDateTime)

    assert time_module.format_relative_time("2025-12-31T23:59:59") == "1 second ago"
    assert time_module.format_relative_time("2025-12-31T23:58:00") == "2 minutes ago"
    assert time_module.format_relative_time("2025-12-31T22:00:00") == "2 hours ago"
    assert time_module.format_relative_time("2025-12-30T00:00:00") == "2 days ago"
    assert time_module.format_relative_time("2025-11-01T00:00:00") == "2 months ago"
    assert time_module.format_relative_time("2024-01-01T00:00:00") == "2 years ago"


def test_format_relative_time_supports_timezone_z_suffix_numeric_and_datetime_inputs(monkeypatch) -> None:
    monkeypatch.setattr(time_module, "datetime", _FrozenDateTime)

    assert time_module.format_relative_time("2025-12-31T16:00:00Z") == "8 hours ago"
    assert time_module.format_relative_time(1767225600) == "0 seconds ago"
    aware_dt = _FrozenDateTime(2025, 12, 31, 16, 0, 0, tzinfo=timezone.utc)
    assert time_module.format_relative_time(aware_dt) == "8 hours ago"


def test_format_relative_time_handles_future_and_invalid_values(monkeypatch) -> None:
    monkeypatch.setattr(time_module, "datetime", _FrozenDateTime)

    assert time_module.format_relative_time("2026-01-01T00:00:01") == "in the future"
    assert time_module.format_relative_time(object()) == "unknown"
    assert time_module.format_relative_time("not-a-time") == "unknown"
