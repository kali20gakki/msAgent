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

from __future__ import annotations

from dataclasses import dataclass

import pytest

import msagent.utils.version as version_module


@dataclass
class _Response:
    status_code: int
    version: str = "0.0.0"

    def json(self) -> dict[str, dict[str, str]]:
        return {"info": {"version": self.version}}


class _FeatureResource:
    def __init__(self, content: str) -> None:
        self.content = content
        self.joined_path: str | None = None

    def joinpath(self, path: str) -> _FeatureResource:
        self.joined_path = path
        return self

    def read_text(self, *, encoding: str) -> str:
        assert encoding == "utf-8"
        return self.content


def test_get_latest_features_returns_recent_compatible_items_when_versions_span_future_and_past(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = _FeatureResource(
        """
max_display: 3
features_by_version:
  27.1.x: [future]
  26.1.x: [current-a, current-b]
  25.12.x: [previous-a, previous-b]
"""
    )
    monkeypatch.setattr(version_module.importlib.resources, "files", lambda package: resource)
    monkeypatch.setattr(version_module, "get_version", lambda: "26.1.4")

    result = version_module.get_latest_features()

    assert result == ["current-a", "current-b", "previous-a"]
    assert resource.joined_path == "features/notes.yml"


def test_get_latest_features_uses_default_limit_when_max_display_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = _FeatureResource(
        """
features_by_version:
  26.1.x: [one, two, three, four, five]
"""
    )
    monkeypatch.setattr(version_module.importlib.resources, "files", lambda package: resource)
    monkeypatch.setattr(version_module, "get_version", lambda: "26.1.0")

    assert version_module.get_latest_features() == ["one", "two", "three", "four"]


def test_get_latest_features_returns_empty_when_feature_resource_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource = _FeatureResource("features_by_version: [not-a-mapping]")
    monkeypatch.setattr(version_module.importlib.resources, "files", lambda package: resource)

    assert version_module.get_latest_features() == []


def test_check_for_updates_skips_network_when_current_version_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(version_module, "get_version", lambda: "unknown")

    def unexpected_get(*args: object, **kwargs: object) -> None:
        raise AssertionError("network must not be called")

    monkeypatch.setattr(version_module.httpx, "get", unexpected_get)

    assert version_module.check_for_updates() is None


def test_check_for_updates_returns_upgrade_command_when_pypi_version_is_newer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float, bool]] = []
    monkeypatch.setattr(version_module, "get_version", lambda: "26.1.0")

    def fake_get(url: str, *, timeout: float, follow_redirects: bool) -> _Response:
        calls.append((url, timeout, follow_redirects))
        return _Response(status_code=200, version="26.2.0")

    monkeypatch.setattr(version_module.httpx, "get", fake_get)

    assert version_module.check_for_updates() == (
        "26.2.0",
        "uv tool install mindstudio-agent --upgrade",
    )
    assert calls == [("https://pypi.org/pypi/mindstudio-agent/json", 2.0, True)]


@pytest.mark.parametrize("latest_version", ["26.1.0", "25.12.9"])
def test_check_for_updates_returns_none_when_pypi_version_is_not_newer(
    monkeypatch: pytest.MonkeyPatch,
    latest_version: str,
) -> None:
    monkeypatch.setattr(version_module, "get_version", lambda: "26.1.0")
    monkeypatch.setattr(
        version_module.httpx,
        "get",
        lambda *args, **kwargs: _Response(200, latest_version),
    )

    assert version_module.check_for_updates() is None


def test_check_for_updates_returns_none_when_pypi_response_is_unsuccessful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(version_module, "get_version", lambda: "26.1.0")
    monkeypatch.setattr(version_module.httpx, "get", lambda *args, **kwargs: _Response(503))

    assert version_module.check_for_updates() is None


def test_check_for_updates_returns_none_when_network_request_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(version_module, "get_version", lambda: "26.1.0")
    monkeypatch.setattr(
        version_module.httpx,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(version_module.httpx.ConnectError("offline")),
    )

    assert version_module.check_for_updates() is None
