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

import json
from pathlib import Path

from msagent.core.paths import AppPaths


def test_app_paths_uses_msagent_home(tmp_path: Path) -> None:
    home = tmp_path / "custom-home"

    paths = AppPaths.resolve({"MSAGENT_HOME": str(home)})

    assert paths.home == home.resolve()
    assert paths.metadata_file == home.resolve() / "metadata.json"
    assert paths.config_dir == home.resolve() / "config"
    assert paths.projects_dir == home.resolve() / "state" / "projects"


def test_project_paths_are_stable_and_isolated(tmp_path: Path) -> None:
    app_paths = AppPaths.from_home(tmp_path / ".msagent")
    first = tmp_path / "a" / "same-name"
    second = tmp_path / "b" / "same-name"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    first_paths = app_paths.for_project(first)
    repeated = app_paths.for_project(first)
    second_paths = app_paths.for_project(second)

    assert first_paths.project_id == repeated.project_id
    assert first_paths.project_id != second_paths.project_id
    assert first_paths.root.parent == app_paths.projects_dir


def test_project_ensure_writes_only_under_global_home(tmp_path: Path) -> None:
    app_paths = AppPaths.from_home(tmp_path / "global" / ".msagent")
    working_dir = tmp_path / "workspace"
    working_dir.mkdir()
    project = app_paths.for_project(working_dir)

    project.ensure()

    assert project.metadata_file.is_file()
    assert not (working_dir / ".msagent").exists()


def test_project_persists_agent_and_model_preferences(tmp_path: Path) -> None:
    app_paths = AppPaths.from_home(tmp_path / "global" / ".msagent")
    working_dir = tmp_path / "workspace"
    working_dir.mkdir()
    project = app_paths.for_project(working_dir)

    project.set_current_agent("Minos")
    project.set_current_model("Minos", "fast")

    assert project.get_current_agent() == "Minos"
    assert project.get_current_model("Minos") == "fast"
    assert project.get_current_model("Profiler") is None
    metadata = json.loads(project.metadata_file.read_text(encoding="utf-8"))
    assert metadata["working_dir"] == str(working_dir.resolve())
    assert metadata["current_agent"] == "Minos"
    assert metadata["agent_models"] == {"Minos": "fast"}


def test_project_preferences_are_isolated_by_workspace(tmp_path: Path) -> None:
    app_paths = AppPaths.from_home(tmp_path / "global" / ".msagent")
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = app_paths.for_project(first_dir)
    second = app_paths.for_project(second_dir)

    first.set_current_agent("Minos")

    assert first.get_current_agent() == "Minos"
    assert second.get_current_agent() is None
