"""Initializer for assembling deepagents runtime dependencies."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from fnmatch import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from msagent.agents.context import AgentContext
from msagent.agents.factory import AgentFactory
from msagent.cli.bootstrap.timer import timer
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
from msagent.core.constants import (
    CONFIG_CHECKPOINTS_URL_FILE_NAME,
    CONFIG_MCP_CACHE_DIR,
    CONFIG_MCP_OAUTH_DIR,
    CONFIG_SKILLS_DIR,
)
from msagent.llms.factory import LLMFactory
from msagent.mcp.factory import MCPFactory
from msagent.skills.factory import Skill, SkillFactory
from msagent.testing.fake_graph import FakeGraph
from msagent.tools.factory import ToolFactory

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph


class Initializer:
    """Centralized service for initializing and caching runtime resources."""

    def __init__(self) -> None:
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
        if working_dir not in self._registries:
            self._registries[working_dir] = ConfigRegistry(working_dir)
        return self._registries[working_dir]

    async def load_llms_config(self, working_dir: Path) -> BatchLLMConfig:
        return await self.get_registry(working_dir).load_llms()

    async def load_llm_config(self, model: str, working_dir: Path) -> LLMConfig:
        return await self.get_registry(working_dir).get_llm(model)

    async def load_checkpointers_config(
        self, working_dir: Path
    ) -> BatchCheckpointerConfig:
        return await self.get_registry(working_dir).load_checkpointers()

    async def load_agents_config(self, working_dir: Path) -> BatchAgentConfig:
        return await self.get_registry(working_dir).load_agents()

    async def load_agent_config(
        self, agent: str | None, working_dir: Path
    ) -> AgentConfig:
        return await self.get_registry(working_dir).get_agent(agent)

    async def load_mcp_config(self, working_dir: Path) -> MCPConfig:
        return await self.get_registry(working_dir).load_mcp()

    async def save_mcp_config(self, mcp_config: MCPConfig, working_dir: Path) -> None:
        await self.get_registry(working_dir).save_mcp(mcp_config)

    async def update_agent_llm(
        self, agent_name: str, new_llm_name: str, working_dir: Path
    ) -> None:
        await self.get_registry(working_dir).update_agent_llm(agent_name, new_llm_name)

    async def update_default_agent(self, agent_name: str, working_dir: Path) -> None:
        await self.get_registry(working_dir).update_default_agent(agent_name)

    async def load_user_memory(self, working_dir: Path) -> str:
        return await self.get_registry(working_dir).load_user_memory()

    @asynccontextmanager
    async def get_checkpointer(
        self, agent: str, working_dir: Path
    ) -> AsyncIterator[BaseCheckpointSaver]:
        """Open the configured checkpointer for a given agent."""
        agent_config = await self.load_agent_config(agent, working_dir)
        checkpointer_ctx = self._create_checkpointer(
            cast(CheckpointerConfig | None, agent_config.checkpointer),
            str(working_dir / CONFIG_CHECKPOINTS_URL_FILE_NAME),
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
            load_approval = getattr(registry, "load_approval", None)
            if callable(load_approval):
                approval_config = load_approval()
            else:
                approval_config = ToolApprovalConfig()
            interrupt_on = approval_config.to_interrupt_on_payload()

        with timer("Create checkpointer"):
            checkpointer_ctx = self._create_checkpointer(
                cast(CheckpointerConfig | None, agent_config.checkpointer),
                str(working_dir / CONFIG_CHECKPOINTS_URL_FILE_NAME),
            )
            checkpointer = await checkpointer_ctx.__aenter__()

        with timer("Create MCP client"):
            default_timeout = (
                float(agent_config.tools.execution_timeout_seconds)
                if agent_config.tools is not None
                else None
            )
            mcp_client = await self.mcp_factory.create(
                config=mcp_config,
                cache_dir=working_dir / CONFIG_MCP_CACHE_DIR,
                oauth_dir=working_dir / CONFIG_MCP_OAUTH_DIR,
                sandbox_bindings=None,
                default_invoke_timeout=default_timeout,
            )
            mcp_module_map = dict(getattr(mcp_client, "module_map", {}) or {})

        with timer("Load skills metadata"):
            skills_dirs = self._resolve_skills_dirs(working_dir)
            skill_map = await self.skill_factory.load_skills(skills_dirs)
            cached_skills = [
                skill for category in skill_map.values() for skill in category.values()
            ]
            skills_config = getattr(agent_config, "skills", None)
            skill_patterns = (
                list(skills_config.patterns or []) if skills_config is not None else []
            )
            filtered_skills = self._filter_skills_by_patterns(
                cached_skills,
                patterns=skill_patterns,
            )
            runtime_skills_dirs = (
                skills_dirs
                if any(
                    pattern and not pattern.startswith("!")
                    for pattern in skill_patterns
                )
                else None
            )

        with timer("Create and compile graph"):
            graph = await self.agent_factory.create(
                config=agent_config,
                working_dir=working_dir,
                context_schema=AgentContext,
                checkpointer=checkpointer,
                mcp_client=mcp_client,
                llm_config=llm_config,
                skills_dir=runtime_skills_dirs,
                sandbox_bindings=None,
                interrupt_on=interrupt_on,
            )

        self.cached_llm_tools = list(getattr(graph, "_llm_tools", []))
        self.cached_tools_in_catalog = list(
            getattr(graph, "_tools_in_catalog", self.cached_llm_tools)
            or self.tool_factory.get_catalog_tools()
        )
        self.cached_agent_skills = filtered_skills
        self.cached_mcp_server_names = self._resolve_cached_mcp_server_names(
            tools=self.cached_llm_tools,
            mcp_config=mcp_config,
            mcp_module_map=mcp_module_map,
        )

        async def cleanup() -> None:
            await mcp_client.close()
            await checkpointer_ctx.__aexit__(None, None, None)

        return graph, cleanup

    def _resolve_skills_dirs(self, working_dir: Path) -> list[Path]:
        candidates = [
            working_dir / "skills",
            self.skill_factory.get_default_skills_dir(),
            working_dir / CONFIG_SKILLS_DIR,
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
        enabled_servers = {
            name for name, server in mcp_config.servers.items() if server.enabled
        }
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
    def _filter_skills_by_patterns(
        skills: list[Skill], patterns: list[str]
    ) -> list[Skill]:
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
            if not any(
                matches(pattern, category=skill.category, name=skill.name)
                for pattern in positive_patterns
            ):
                continue
            if any(
                matches(pattern, category=skill.category, name=skill.name)
                for pattern in negative_patterns
            ):
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
