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

import importlib


def test_core_package_surfaces_import() -> None:
    modules = [
        "msagent.agents",
        "msagent.cli.bootstrap.app",
        "msagent.cli.bootstrap.web",
        "msagent.configs",
        "msagent.core.settings",
        "msagent.llms.factory",
        "msagent.mcp.factory",
        "msagent.middlewares",
        "msagent.skills.factory",
        "msagent.tools.factory",
        "msagent.web.graph",
        "msagent.web.launcher",
        "msagent.web.runtime",
        "msagent.web.ui",
    ]

    for module_name in modules:
        assert importlib.import_module(module_name) is not None
