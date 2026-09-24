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

"""Tests for the packaged ascend-knowledge subagent and its wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from msagent.configs.agent import BatchAgentConfig, BatchSubAgentConfig
from msagent.configs.checkpointer import BatchCheckpointerConfig
from msagent.configs.llm import BatchLLMConfig
from msagent.configs.mcp import MCPConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_ROOT = PROJECT_ROOT / "resources" / "configs" / "default"

# ascend-doc-mcp tool names must not be hardcoded in the prompt: if the MCP
# interface changes, a name-pinned prompt silently goes stale.
_MCP_TOOL_NAMES = (
    "read_llms_txt",
    "list_products",
    "list_versions",
    "get_latest_version",
    "get_popular_docs",
    "get_doc_toc",
    "get_doc_outline",
    "get_doc_content",
    "browse_docs",
    "list_pdf_downloads",
    "search_docs",
    "search_docs_in_product",
    "search_troubleshooting",
    "search_repos",
    "list_repo_files",
    "get_repo_file",
    "get_download_options",
    "get_install_commands",
    "get_version_announcements",
    "get_code_samples",
)


def _load_prompt() -> str:
    return (DEFAULT_CONFIG_ROOT / "prompts" / "subagents" / "ascend-knowledge.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_packaged_defaults_contain_ascend_knowledge_subagent() -> None:
    llms = await BatchLLMConfig.from_yaml(
        file_path=DEFAULT_CONFIG_ROOT / "config.llms.yml",
        dir_path=DEFAULT_CONFIG_ROOT / "llms",
    )
    subagents = await BatchSubAgentConfig.from_yaml(
        dir_path=DEFAULT_CONFIG_ROOT / "subagents",
        batch_llm_config=llms,
    )

    subagent = subagents.get_subagent_config("ascend-knowledge")
    assert subagent is not None

    patterns = list(subagent.tools.patterns or [])
    assert patterns == ["mcp:ascend-doc-mcp:*"]
    assert subagent.skills is not None
    assert not (subagent.skills.patterns or [])
    # Deterministic validation errors must not be blindly retried by the tool
    # middleware; failures return to the model so it can fix arguments.
    assert subagent.retry is not None
    assert subagent.retry.tool.enabled is False
    assert subagent.retry.tool.max_retries == 0


@pytest.mark.asyncio
async def test_profiler_lists_ascend_knowledge_subagent() -> None:
    llms = await BatchLLMConfig.from_yaml(
        file_path=DEFAULT_CONFIG_ROOT / "config.llms.yml",
        dir_path=DEFAULT_CONFIG_ROOT / "llms",
    )
    checkpointers = await BatchCheckpointerConfig.from_yaml(
        dir_path=DEFAULT_CONFIG_ROOT / "checkpointers",
    )
    subagents = await BatchSubAgentConfig.from_yaml(
        dir_path=DEFAULT_CONFIG_ROOT / "subagents",
        batch_llm_config=llms,
    )
    agents = await BatchAgentConfig.from_yaml(
        dir_path=DEFAULT_CONFIG_ROOT / "agents",
        batch_llm_config=llms,
        batch_checkpointer_config=checkpointers,
        batch_subagent_config=subagents,
    )

    profiler = agents.get_agent_config("Profiler")
    assert profiler is not None
    names = [sub.name for sub in (profiler.subagents or [])]
    assert "ascend-knowledge" in names


@pytest.mark.asyncio
async def test_default_mcp_config_registers_ascend_doc_mcp_as_enabled() -> None:
    mcp = await MCPConfig.from_json(DEFAULT_CONFIG_ROOT / "config.mcp.json")

    server = mcp.servers["ascend-doc-mcp"]
    assert server.enabled is True
    assert server.command == "msagent-ascend-doc-mcp"
    assert server.transport.value == "stdio"


def test_ascend_knowledge_config_has_no_inert_fields() -> None:
    raw = (DEFAULT_CONFIG_ROOT / "subagents" / "ascend-knowledge.yml").read_text(encoding="utf-8")
    assert "runtime_timeout_seconds" not in raw
    assert "recursion_limit" not in raw


def test_ascend_knowledge_prompt_is_schema_driven_not_version_pinned() -> None:
    """The prompt must describe behavior, not a snapshot of MCP tool names.

    Runtime tool definitions (names + schemas) are the source of truth, so an
    interface change cannot silently invalidate the instructions.
    """
    prompt = _load_prompt()
    for tool in _MCP_TOOL_NAMES:
        assert tool not in prompt
    assert "schema" in prompt
    assert "Unrecognized key" in prompt
    assert "命中即止" in prompt
    assert "不可信" in prompt


def test_ascend_knowledge_prompt_has_no_external_channel_references() -> None:
    prompt = _load_prompt()
    lowered = prompt.lower()
    assert "github" not in lowered
    assert "web_search" not in lowered
    assert "raw.githubusercontent" not in lowered


def test_github_raw_fetch_skill_is_docs_first_and_uses_official_repos() -> None:
    skill = (PROJECT_ROOT / "skills" / "basic" / "github-raw-fetch" / "SKILL.md").read_text(encoding="utf-8")
    assert "Ascend/msprof" in skill
    assert "不是前置依赖" in skill
    assert "kali20gakki" not in skill
    assert "actioncloud" not in skill


def test_profiler_prompt_delegates_doc_queries_to_ascend_knowledge() -> None:
    prompt = (DEFAULT_CONFIG_ROOT / "prompts" / "agents" / "Profiler.md").read_text(encoding="utf-8")
    assert "ascend-knowledge" in prompt
    assert "github-raw-fetch" not in prompt
    assert "kali20gakki" not in prompt
