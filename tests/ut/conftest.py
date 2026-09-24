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

from collections.abc import Iterator
from pathlib import Path

import pytest

from msagent.cli.bootstrap.initializer import initializer
from msagent.core.paths import AppPaths


@pytest.fixture(autouse=True)
def isolate_msagent_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Prevent unit tests from reading or writing the user's real msAgent home."""
    test_home = tmp_path / "msagent-home"
    test_paths = AppPaths.from_home(test_home)
    previous_paths = initializer.app_paths
    previous_registries = initializer._registries

    monkeypatch.setenv("MSAGENT_HOME", str(test_home))
    initializer.app_paths = test_paths
    initializer._registries = {}
    try:
        yield
    finally:
        initializer.app_paths = previous_paths
        initializer._registries = previous_registries
