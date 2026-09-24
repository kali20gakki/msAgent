#!/usr/bin/python3
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

"""Configuration module for msagent."""

from msagent.configs.agent import (
    AgentConfig,
    BaseAgentConfig,
    BaseBatchConfig,
    BatchAgentConfig,
    BatchSubAgentConfig,
    CompressionConfig,
    ModelRetryConfig,
    RetryPolicyConfig,
    SkillsConfig,
    AuditLogConfig,
    SubAgentConfig,
    ToolRetryConfig,
    ToolsConfig,
)
from msagent.configs.approval import (
    ApprovalMode,
    ExecuteApprovalMode,
    InterruptOnRule,
    ToolDecision,
    ToolDecisionRule,
    ToolApprovalConfig,
)
from msagent.configs.base import VersionedConfig
from msagent.configs.checkpointer import (
    BatchCheckpointerConfig,
    CheckpointerConfig,
    CheckpointerProvider,
)
from msagent.configs.llm import BatchLLMConfig, LLMConfig, LLMProvider, RateConfig
from msagent.configs.mcp import MCPConfig, MCPServerConfig, MCPTransport
from msagent.configs.registry import ConfigRegistry
from msagent.configs.utils import load_prompt_content

__all__ = [
    # Base
    "VersionedConfig",
    # LLM
    "LLMConfig",
    "BatchLLMConfig",
    "LLMProvider",
    "RateConfig",
    # Checkpointer
    "CheckpointerConfig",
    "BatchCheckpointerConfig",
    "CheckpointerProvider",
    # Agent
    "BaseAgentConfig",
    "AgentConfig",
    "BatchAgentConfig",
    "SubAgentConfig",
    "BatchSubAgentConfig",
    "BaseBatchConfig",
    "CompressionConfig",
    "AuditLogConfig",
    "ModelRetryConfig",
    "ToolsConfig",
    "SkillsConfig",
    "RetryPolicyConfig",
    "ToolRetryConfig",
    # MCP
    "MCPConfig",
    "MCPServerConfig",
    "MCPTransport",
    # Approval
    "ApprovalMode",
    "ExecuteApprovalMode",
    "InterruptOnRule",
    "ToolDecision",
    "ToolDecisionRule",
    "ToolApprovalConfig",
    # Registry
    "ConfigRegistry",
    # Utils
    "load_prompt_content",
]
