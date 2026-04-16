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

from deepagents.backends import utils as backend_utils
from deepagents.middleware import filesystem as filesystem_middleware

from msagent.utils.deepagents_compat import (
    patch_deepagents_windows_absolute_paths,
    validate_deepagents_path,
)


def test_validate_deepagents_path_accepts_windows_absolute_paths() -> None:
    path = r"C:\ProfileData\trace\result.json"

    assert validate_deepagents_path(path) == path


def test_patch_deepagents_windows_absolute_paths_updates_deepagents_modules() -> None:
    patch_deepagents_windows_absolute_paths()

    path = r"C:\ProfileData\trace\result.json"

    assert backend_utils.validate_path(path) == path
    assert filesystem_middleware.validate_path(path) == path
