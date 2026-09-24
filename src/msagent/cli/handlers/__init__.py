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

"""Handlers for executing specific commands and workflows."""

from msagent.cli.handlers.add_skill import AddSkillHandler
from msagent.cli.handlers.agents import AgentHandler
from msagent.cli.handlers.compress import CompressionHandler
from msagent.cli.handlers.interrupts import InterruptHandler
from msagent.cli.handlers.memory import MemoryHandler
from msagent.cli.handlers.mcp import MCPHandler
from msagent.cli.handlers.models import ModelHandler
from msagent.cli.handlers.permissions import PermissionsHandler
from msagent.cli.handlers.skills import SkillsHandler
from msagent.cli.handlers.tool_outputs import ToolOutputHandler
from msagent.cli.handlers.threads import ThreadsHandler
from msagent.cli.handlers.tools import ToolsHandler

__all__ = [
    "AddSkillHandler",
    "AgentHandler",
    "CompressionHandler",
    "InterruptHandler",
    "MemoryHandler",
    "MCPHandler",
    "ModelHandler",
    "PermissionsHandler",
    "SkillsHandler",
    "ToolOutputHandler",
    "ThreadsHandler",
    "ToolsHandler",
]
