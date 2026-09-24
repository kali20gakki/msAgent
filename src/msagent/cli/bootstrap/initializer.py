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

"""Initializer for assembling deepagents runtime dependencies."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from msagent.agents.context import AgentContext
from msagent.agents.factory import AgentFactory
from msagent.cli.bootstrap.timer import timer
from msagent.cli.theme import console
from msagent.configs import (
    AgentConfig,
    BatchAgentConfig,
    BatchCheckpointerConfig,
    BatchLLMConfig,
    CheckpointerConfig,
    CheckpointerProvider,
    ConfigRegistry,
    LLMConfig,
    MCPConfig,
    ToolApprovalConfig,
)
from msagent.core.paths import AppPaths, ProjectPaths
from msagent.llms.factory import LLMFactory
from msagent.mcp.factory import MCPFactory
from msagent.skills.factory import Skill, SkillFactory
from msagent.testing.fake_graph import FakeGraph
from msagent.tools.factory import ToolFactory

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class Initializer:
    """Centralized service for initializing and caching runtime resources."""

    def __init__(self, app_paths: AppPaths | None = None) -> None:
        self.app_paths = app_paths or AppPaths.resolve()
        self.tool_factory = ToolFactory()
        self.skill_factory = SkillFactory()
        self.llm_factory = LLMFactory()
        self.mcp_factory = MCPFactory(tool_factory=self.tool_factory)
        self.agent_factory = AgentFactory(
            llm_factory=self.llm_factory,
            tool_factory=self.tool_factory,
        )

        self.cached_llm_tools: list[BaseTool] = []
        self.cached_tools_in_catalog: list[BaseTool | object] = []
        self.cached_agent_skills: list[Skill] = []
        self.cached_mcp_server_names: list[str] = []

        self._registries: dict[Path, ConfigRegistry] = {}

    def get_registry(self, working_dir: Path) -> ConfigRegistry:
        canonical = working_dir.expanduser().resolve()
        if canonical not in self._registries:
            self._registries[canonical] = ConfigRegistry(canonical, self.app_paths)
        return self._registries[canonical]

    def get_project_paths(self, working_dir: Path) -> ProjectPaths:
        return self.app_paths.for_project(working_dir)

    async def load_llms_config(self, working_dir: Path) -> BatchLLMConfig:
        return await self.get_registry(working_dir).load_llms()

    async def load_llm_config(self, model: str, working_dir: Path) -> LLMConfig:
        return await self.get_registry(working_dir).get_llm(model)

    async def load_checkpointers_config(self, working_dir: Path) -> BatchCheckpointerConfig:
        return await self.get_registry(working_dir).load_checkpointers()

    async def load_agents_config(self, working_dir: Path) -> BatchAgentConfig:
        return await self.get_registry(working_dir).load_agents()

    async def load_agent_config(self, agent: str | None, working_dir: Path) -> AgentConfig:
        return await self.get_registry(working_dir).get_agent(agent)

    async def load_mcp_config(self, working_dir: Path) -> MCPConfig:
        return await self.get_registry(working_dir).load_mcp()

    async def save_mcp_config(self, mcp_config: MCPConfig, working_dir: Path) -> None:
        await self.get_registry(working_dir).save_mcp(mcp_config)

    async def update_agent_llm(self, agent_name: str, new_llm_name: str, working_dir: Path) -> None:
        await self.get_registry(working_dir).update_agent_llm(agent_name, new_llm_name)

    async def set_current_agent(self, agent_name: str, working_dir: Path) -> None:
        await self.get_registry(working_dir).set_current_agent(agent_name)

    async def get_current_model(self, agent_name: str, working_dir: Path) -> str | None:
        return await self.get_registry(working_dir).get_current_model(agent_name)

    async def set_current_model(self, agent_name: str, model_name: str, working_dir: Path) -> None:
        await self.get_registry(working_dir).set_current_model(agent_name, model_name)

    async def update_default_agent(self, agent_name: str, working_dir: Path) -> None:
        """Compatibility alias for workspace-scoped Agent selection."""
        await self.set_current_agent(agent_name, working_dir)

    async def add_agent_skill_pattern(self, agent_name: str, skill_pattern: str, working_dir: Path) -> bool:
        return await self.get_registry(working_dir).add_agent_skill_pattern(agent_name, skill_pattern)

    async def load_user_memory(self, working_dir: Path) -> str:
        return await self.get_registry(working_dir).load_user_memory()

    def resolve_skills_dirs(self, working_dir: Path) -> list[Path]:
        """Resolve skill search directories in precedence order."""
        return self._resolve_skills_dirs(working_dir)

    async def refresh_cached_skills(self, *, agent: str | None, working_dir: Path) -> list[Skill]:
        """Refresh cached skill metadata for the current working directory and agent."""
        agent_config = await self.load_agent_config(agent, working_dir)
        skills_dirs = self._resolve_skills_dirs(working_dir)
        skill_map = await self.skill_factory.load_skills(skills_dirs)
        cached_skills = [skill for category in skill_map.values() for skill in category.values()]
        skills_config = getattr(agent_config, "skills", None)
        skill_patterns = list(skills_config.patterns or []) if skills_config is not None else []
        filtered_skills = self._filter_skills_by_patterns(
            cached_skills,
            patterns=skill_patterns,
        )
        self.cached_agent_skills = filtered_skills
        return filtered_skills

    @asynccontextmanager
    async def get_checkpointer(self, agent: str, working_dir: Path) -> AsyncIterator[BaseCheckpointSaver]:
        """Open the configured checkpointer for a given agent."""
        agent_config = await self.load_agent_config(agent, working_dir)
        project_paths = self.get_project_paths(working_dir)
        await asyncio.to_thread(project_paths.ensure)
        checkpointer_ctx = self._create_checkpointer(
            cast(CheckpointerConfig | None, agent_config.checkpointer),
            str(project_paths.checkpoints_db),
        )
        checkpointer = await checkpointer_ctx.__aenter__()
        try:
            yield checkpointer
        finally:
            await checkpointer_ctx.__aexit__(None, None, None)

    async def create_graph(
        self,
        agent: str | None,
        model: str | None,
        working_dir: Path,
    ) -> tuple[CompiledStateGraph | FakeGraph, Callable[[], Awaitable[None]]]:
        if os.getenv("MSAGENT_FAKE_BACKEND", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }:
            fake_graph = FakeGraph()
            self.cached_llm_tools = []
            self.cached_tools_in_catalog = []
            self.cached_agent_skills = []
            self.cached_mcp_server_names = []

            async def fake_cleanup() -> None:
                return None

            return fake_graph, fake_cleanup

        registry = self.get_registry(working_dir)
        project_paths = getattr(registry, "project_paths", self.get_project_paths(working_dir))

        with timer("Load configs"):
            if model:
                agent_config, llm_config, mcp_config = await asyncio.gather(
                    registry.get_agent(agent),
                    registry.get_llm(model),
                    registry.load_mcp(),
                )
            else:
                agent_config, mcp_config = await asyncio.gather(
                    registry.get_agent(agent),
                    registry.load_mcp(),
                )
                llm_config = None

        with timer("Load approval config"):
            approval_config = ToolApprovalConfig()
            interrupt_on = approval_config.to_interrupt_on_payload()

        with timer("Create checkpointer"):
            checkpointer_ctx = self._create_checkpointer(
                cast(CheckpointerConfig | None, agent_config.checkpointer),
                str(project_paths.checkpoints_db),
            )
            checkpointer = await checkpointer_ctx.__aenter__()

        with timer("Create MCP client"):
            default_timeout = (
                float(agent_config.tools.execution_timeout_seconds) if agent_config.tools is not None else None
            )
            mcp_client = await self.mcp_factory.create(
                config=mcp_config,
                cache_dir=self.app_paths.mcp_cache_dir,
                oauth_dir=self.app_paths.mcp_oauth_dir,
                default_invoke_timeout=default_timeout,
            )
            mcp_module_map = dict(getattr(mcp_client, "module_map", {}) or {})

        with timer("Load skills metadata"):
            skills_dirs = self._resolve_skills_dirs(working_dir)
            filtered_skills = await self.refresh_cached_skills(agent=agent_config.name, working_dir=working_dir)
            skills_config = getattr(agent_config, "skills", None)
            skill_patterns = list(skills_config.patterns or []) if skills_config is not None else []
            runtime_skills_dirs = (
                skills_dirs if any(pattern and not pattern.startswith("!") for pattern in skill_patterns) else None
            )

        with timer("Create and compile graph"):
            graph = await self.agent_factory.create(
                config=agent_config,
                working_dir=working_dir,
                project_state_dir=project_paths.root,
                context_schema=AgentContext,
                checkpointer=checkpointer,
                mcp_client=mcp_client,
                llm_config=llm_config,
                skills_dir=runtime_skills_dirs,
                allowed_skills=filtered_skills,
                interrupt_on=interrupt_on,
            )

        self.cached_llm_tools = list(getattr(graph, "_llm_tools", []))
        self.cached_tools_in_catalog = list(
            getattr(graph, "_tools_in_catalog", self.cached_llm_tools) or self.tool_factory.get_catalog_tools()
        )
        self.cached_mcp_server_names = self._resolve_cached_mcp_server_names(
            tools=self.cached_llm_tools,
            mcp_config=mcp_config,
            mcp_module_map=mcp_module_map,
        )
        self._warn_unavailable_mcp_servers(mcp_client)

        async def cleanup() -> None:
            await mcp_client.close()
            await checkpointer_ctx.__aexit__(None, None, None)

        return graph, cleanup

    def _warn_unavailable_mcp_servers(self, mcp_client: Any) -> None:
        """Report enabled MCP servers that could not be started.

        These servers no longer abort session startup, so the user has to be
        told which capabilities are missing.
        """
        unavailable = dict(getattr(mcp_client, "unavailable_servers", {}) or {})
        if not unavailable:
            return
        names = ", ".join(sorted(unavailable))
        logger.warning("MCP servers unavailable: %s", unavailable)
        console.print_warning(
            f"MCP server(s) unavailable: {names}. Their tools are not loaded; "
            "run the installer to fix them or use /mcp to disable them."
        )
        console.print("")

    def _resolve_skills_dirs(self, working_dir: Path) -> list[Path]:
        candidates = [
            working_dir / "skills",
            self.app_paths.skills_dir,
            self.skill_factory.get_default_skills_dir(),
        ]

        unique_paths: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            normalized = str(path.resolve())
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_paths.append(path)
        return unique_paths

    def _resolve_cached_mcp_server_names(
        self,
        *,
        tools: list[BaseTool],
        mcp_config: MCPConfig,
        mcp_module_map: dict[str, str],
    ) -> list[str]:
        enabled_servers = {name for name, server in mcp_config.servers.items() if server.enabled}
        if not enabled_servers:
            return []

        visible_servers: set[str] = set()
        for tool in tools:
            tool_name = self.agent_factory._tool_name(tool)
            module, _raw_name = self.agent_factory._resolve_mcp_tool_identity(
                tool_name=tool_name,
                mcp_module_map=mcp_module_map,
                mcp_servers=enabled_servers,
            )
            if module != "unknown":
                visible_servers.add(module)

        return [name for name in mcp_config.servers.keys() if name in visible_servers]

    @asynccontextmanager
    async def _create_checkpointer(
        self,
        config: CheckpointerConfig | None,
        db_path: str | None = None,
    ) -> AsyncIterator[BaseCheckpointSaver]:
        if config is None or config.type == CheckpointerProvider.MEMORY:
            yield InMemorySaver()
            return

        if config.type == CheckpointerProvider.SQLITE:
            sqlite_path = config.connection_string or db_path
            if sqlite_path:
                import aiosqlite

                conn = await aiosqlite.connect(sqlite_path)
                try:
                    yield AsyncSqliteSaver(conn)
                finally:
                    await conn.close()
                return

        yield InMemorySaver()

    @staticmethod
    def _filter_skills_by_patterns(skills: list[Skill], patterns: list[str]) -> list[Skill]:
        if not patterns:
            return []

        positive_patterns = [p for p in patterns if p and not p.startswith("!")]
        negative_patterns = [p[1:] for p in patterns if p.startswith("!")]
        if not positive_patterns:
            return []

        def matches(pattern: str, *, category: str, name: str) -> bool:
            parts = pattern.split(":")
            if len(parts) != 2:
                return False
            category_p, name_p = parts
            return fnmatch(category, category_p) and fnmatch(name, name_p)

        filtered: list[Skill] = []
        for skill in skills:
            if not any(matches(pattern, category=skill.category, name=skill.name) for pattern in positive_patterns):
                continue
            if any(matches(pattern, category=skill.category, name=skill.name) for pattern in negative_patterns):
                continue
            filtered.append(skill)
        return filtered

    @asynccontextmanager
    async def get_graph(
        self,
        agent: str | None,
        model: str | None,
        working_dir: Path,
    ) -> AsyncIterator[CompiledStateGraph | FakeGraph]:
        graph, cleanup = await self.create_graph(agent, model, working_dir)
        try:
            yield graph
        finally:
            await cleanup()


initializer = Initializer()
