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

"""Middleware - now provided by deepagents."""

# Deepagents provides:
# - FilesystemMiddleware (ls, read_file, write_file, edit_file, glob, grep, execute)
# - SkillsMiddleware (skill loading)
# - MemoryMiddleware (AGENTS.md memory)
# - TodoListMiddleware (todos tool)

# Re-export from deepagents for convenience.
from deepagents.middleware import (
    FilesystemMiddleware,
    MemoryMiddleware,
    SkillsMiddleware,
)
from langchain.agents.middleware import TodoListMiddleware
from msagent.middlewares.tool_result_eviction import ToolResultEvictionMiddleware

__all__ = [
    "FilesystemMiddleware",
    "TodoListMiddleware",
    "SkillsMiddleware",
    "MemoryMiddleware",
    "ToolResultEvictionMiddleware",
]
