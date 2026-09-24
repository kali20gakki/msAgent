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

"""Central registry for loading, saving, and caching configurations."""

from __future__ import annotations

import asyncio
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml
from pydantic import BaseModel

from msagent.configs.agent import (
    AgentConfig,
    BatchAgentConfig,
    BatchSubAgentConfig,
    SubAgentConfig,
)
from msagent.configs.checkpointer import BatchCheckpointerConfig, CheckpointerConfig
from msagent.configs.llm import BatchLLMConfig, LLMConfig
from msagent.configs.mcp import MCPConfig
from msagent.core.constants import (
    CONFIG_AGENTS_DIR,
    CONFIG_AGENTS_FILE_NAME,
    CONFIG_CHECKPOINTERS_DIR,
    CONFIG_CHECKPOINTERS_FILE_NAME,
    CONFIG_LLMS_DIR,
    CONFIG_LLMS_FILE_NAME,
    CONFIG_MCP_FILE_NAME,
    CONFIG_SUBAGENTS_DIR,
    CONFIG_SUBAGENTS_FILE_NAME,
)
from msagent.core.paths import AppPaths
from msagent.tools.internal.memory import ensure_memory_file, is_default_memory_content

T = TypeVar("T")


def _merge_items(base: list[T], overrides: list[T], key: Callable[[T], Any]) -> list[T]:
    """Merge named configuration objects while preserving stable ordering."""
    merged = list(base)
    positions = {key(item): index for index, item in enumerate(merged)}
    for item in overrides:
        item_key = key(item)
        if item_key in positions:
            merged[positions[item_key]] = item
        else:
            positions[item_key] = len(merged)
            merged.append(item)
    return merged


def _overlay_model(base: T, override: T) -> T:
    """Apply only fields explicitly present in a validated override model."""
    if not isinstance(base, BaseModel) or not isinstance(override, BaseModel):
        return override

    updates: dict[str, object] = {}
    for field_name in override.model_fields_set:
        override_value = getattr(override, field_name)
        base_value = getattr(base, field_name, None)
        if isinstance(base_value, BaseModel) and isinstance(override_value, BaseModel):
            updates[field_name] = _overlay_model(base_value, override_value)
        else:
            updates[field_name] = override_value
    return base.model_copy(update=updates)


def _merge_model_overrides(base: list[T], overrides: list[T], key: Callable[[T], Any]) -> list[T]:
    """Merge partial Pydantic config overrides while preserving packaged fields."""
    merged = list(base)
    positions = {key(item): index for index, item in enumerate(merged)}
    for item in overrides:
        item_key = key(item)
        if item_key in positions:
            index = positions[item_key]
            merged[index] = _overlay_model(merged[index], item)
        else:
            positions[item_key] = len(merged)
            merged.append(item)
    return merged


class ConfigRegistry:
    """Central registry for loading, saving, and caching all configurations."""

    def __init__(self, working_dir: Path, app_paths: AppPaths | None = None):
        self.working_dir = working_dir.expanduser().resolve()
        self.app_paths = app_paths or AppPaths.resolve()
        self.project_paths = self.app_paths.for_project(self.working_dir)
        self.config_dir = self.app_paths.config_dir
        self.default_config_dir = Path(str(files("resources") / "configs" / "default"))

        self.llms_file = self.config_dir / CONFIG_LLMS_FILE_NAME.name
        self.llms_dir = self.config_dir / CONFIG_LLMS_DIR.name
        self.checkpointers_file = self.config_dir / CONFIG_CHECKPOINTERS_FILE_NAME.name
        self.checkpointers_dir = self.config_dir / CONFIG_CHECKPOINTERS_DIR.name
        self.agents_file = self.config_dir / CONFIG_AGENTS_FILE_NAME.name
        self.agents_dir = self.config_dir / CONFIG_AGENTS_DIR.name
        self.subagents_file = self.config_dir / CONFIG_SUBAGENTS_FILE_NAME.name
        self.subagents_dir = self.config_dir / CONFIG_SUBAGENTS_DIR.name
        self.mcp_file = self.config_dir / CONFIG_MCP_FILE_NAME.name
        # Lazy-loaded caches
        self._llms: BatchLLMConfig | None = None
        self._checkpointers: BatchCheckpointerConfig | None = None
        self._agents: BatchAgentConfig | None = None
        self._subagents: BatchSubAgentConfig | None = None
        self._mcp: MCPConfig | None = None

    # === Setup ===

    async def ensure_config_dir(self) -> None:
        """Create global directories without materializing packaged defaults."""
        await asyncio.to_thread(self.project_paths.ensure)
        await asyncio.to_thread(ensure_memory_file, state_dir=self.project_paths.root)

    @staticmethod
    def _source_exists(file_path: Path, dir_path: Path) -> bool:
        return file_path.is_file() or (dir_path.is_dir() and any(dir_path.glob("*.yml")))

    # === LLM configs ===

    async def load_llms(self, force_reload: bool = False) -> BatchLLMConfig:
        """Load all LLM configs (cached)."""
        if self._llms is None or force_reload:
            await self.ensure_config_dir()
            defaults = await BatchLLMConfig.from_yaml(
                file_path=self.default_config_dir / CONFIG_LLMS_FILE_NAME.name,
                dir_path=self.default_config_dir / CONFIG_LLMS_DIR.name,
            )
            overrides: list[LLMConfig] = []
            if self._source_exists(self.llms_file, self.llms_dir):
                overrides = (await BatchLLMConfig.from_yaml(self.llms_file, self.llms_dir)).llms
            self._llms = BatchLLMConfig(llms=_merge_items(defaults.llms, overrides, lambda item: item.alias))
        return self._llms

    async def get_llm(self, alias: str) -> LLMConfig:
        """Get single LLM by alias."""
        llms = await self.load_llms()
        llm = llms.get_llm_config(alias)
        if llm:
            return llm
        raise ValueError(f"LLM '{alias}' not found. Available: {llms.llm_names}")

    # === Checkpointer configs ===

    async def load_checkpointers(self, force_reload: bool = False) -> BatchCheckpointerConfig:
        """Load all checkpointer configs (cached)."""
        if self._checkpointers is None or force_reload:
            await self.ensure_config_dir()
            defaults = await BatchCheckpointerConfig.from_yaml(
                file_path=self.default_config_dir / CONFIG_CHECKPOINTERS_FILE_NAME.name,
                dir_path=self.default_config_dir / CONFIG_CHECKPOINTERS_DIR.name,
            )
            overrides: list[CheckpointerConfig] = []
            if self._source_exists(self.checkpointers_file, self.checkpointers_dir):
                overrides = (
                    await BatchCheckpointerConfig.from_yaml(
                        self.checkpointers_file,
                        self.checkpointers_dir,
                    )
                ).checkpointers
            self._checkpointers = BatchCheckpointerConfig(
                checkpointers=_merge_items(defaults.checkpointers, overrides, lambda item: item.type)
            )
        return self._checkpointers

    async def get_checkpointer(self, name: str) -> CheckpointerConfig | None:
        """Get single checkpointer by type name."""
        checkpointers = await self.load_checkpointers()
        return checkpointers.get_checkpointer_config(name)

    # === SubAgent configs ===

    async def load_subagents(self, force_reload: bool = False) -> BatchSubAgentConfig:
        """Load all subagent configs (cached)."""
        if self._subagents is None or force_reload:
            await self.ensure_config_dir()

            llm_config = await self.load_llms()
            defaults = await BatchSubAgentConfig.from_yaml(
                file_path=self.default_config_dir / CONFIG_SUBAGENTS_FILE_NAME.name,
                dir_path=self.default_config_dir / CONFIG_SUBAGENTS_DIR.name,
                batch_llm_config=llm_config,
            )
            overrides: list[SubAgentConfig] = []
            if self._source_exists(self.subagents_file, self.subagents_dir):
                overrides = (
                    await BatchSubAgentConfig.from_yaml(
                        file_path=self.subagents_file,
                        dir_path=self.subagents_dir,
                        prompt_base_path=self.app_paths.home,
                        batch_llm_config=llm_config,
                    )
                ).subagents
            self._subagents = BatchSubAgentConfig(
                subagents=_merge_model_overrides(defaults.subagents, overrides, lambda item: item.name)
            )
        return self._subagents

    async def get_subagent(self, name: str) -> SubAgentConfig | None:
        """Get single subagent by name."""
        subagents = await self.load_subagents()
        return subagents.get_subagent_config(name)

    # === Agent configs ===

    async def load_agents(self, force_reload: bool = False) -> BatchAgentConfig:
        """Load all agent configs with resolved references (cached)."""
        if self._agents is None or force_reload:
            await self.ensure_config_dir()

            llm_config = await self.load_llms()
            checkpointer_config = await self.load_checkpointers()
            subagents_config = await self.load_subagents()

            defaults = await BatchAgentConfig.from_yaml(
                file_path=self.default_config_dir / CONFIG_AGENTS_FILE_NAME.name,
                dir_path=self.default_config_dir / CONFIG_AGENTS_DIR.name,
                batch_llm_config=llm_config,
                batch_checkpointer_config=checkpointer_config,
                batch_subagent_config=subagents_config,
            )
            overrides: list[AgentConfig] = []
            if self._source_exists(self.agents_file, self.agents_dir):
                overrides = (
                    await BatchAgentConfig.from_yaml(
                        file_path=self.agents_file,
                        dir_path=self.agents_dir,
                        prompt_base_path=self.app_paths.home,
                        allow_partial=True,
                        batch_llm_config=llm_config,
                        batch_checkpointer_config=checkpointer_config,
                        batch_subagent_config=subagents_config,
                    )
                ).agents

            base_agents = defaults.agents
            selected_default = next(
                (item.name for item in overrides if "default" in item.model_fields_set and item.default),
                None,
            )
            if selected_default is not None:
                base_agents = [
                    item.model_copy(update={"default": False}) if item.name != selected_default else item
                    for item in base_agents
                ]
            self._agents = BatchAgentConfig(
                agents=_merge_model_overrides(base_agents, overrides, lambda item: item.name)
            )
        return self._agents

    async def get_agent(self, name: str | None = None) -> AgentConfig:
        """Get an explicit Agent, or the workspace selection, or the packaged default."""
        agents = await self.load_agents()
        selected_name = name if name is not None else self.project_paths.get_current_agent()
        agent = agents.get_agent_config(selected_name)
        if agent is None and name is None:
            agent = agents.get_default_agent()
        if agent:
            return agent
        raise ValueError(f"Agent '{selected_name}' not found. Available: {agents.agent_names}")

    # === MCP config ===

    async def load_mcp(self, force_reload: bool = False) -> MCPConfig:
        """Load MCP server config (cached)."""
        if self._mcp is None or force_reload:
            await self.ensure_config_dir()
            defaults = await MCPConfig.from_json(self.default_config_dir / CONFIG_MCP_FILE_NAME.name)
            overrides = await MCPConfig.from_json(self.mcp_file)
            self._mcp = MCPConfig(servers={**defaults.servers, **overrides.servers})
        return self._mcp

    async def save_mcp(self, config: MCPConfig) -> None:
        """Save MCP config to file."""
        config.to_json(self.mcp_file)
        self._mcp = config

    # === User memory ===

    async def load_user_memory(self) -> str:
        """Load user memory from project-specific memory file.

        Returns:
            Formatted user memory string for prompt injection, or empty string if no memory
        """
        memory_path = self.project_paths.memory_file
        if memory_path.exists():
            content = await asyncio.to_thread(memory_path.read_text, encoding="utf-8")
            content = content.strip()
            if content and not is_default_memory_content(content):
                return f"<user-memory>\n{content}\n</user-memory>"
        return ""

    # === Update operations ===

    async def update_agent_llm(self, agent_name: str, llm_alias: str) -> None:
        """Update an agent's LLM reference and persist."""
        await self._ensure_named_override(agent_name, subagent=False)
        await BatchAgentConfig.update_agent_llm(
            file_path=self.agents_file,
            agent_name=agent_name,
            new_llm_name=llm_alias,
            dir_path=self.agents_dir,
        )
        self._agents = None  # Invalidate cache

    async def update_subagent_llm(self, subagent_name: str, llm_alias: str) -> None:
        """Update a subagent's LLM reference and persist."""
        await self._ensure_named_override(subagent_name, subagent=True)
        await BatchAgentConfig.update_agent_llm(
            file_path=self.subagents_file,
            agent_name=subagent_name,
            new_llm_name=llm_alias,
            dir_path=self.subagents_dir,
        )
        self._subagents = None  # Invalidate cache

    async def set_current_agent(self, agent_name: str) -> None:
        """Persist the selected Agent in this workspace's project state."""
        agents = await self.load_agents()
        if agents.get_agent_config(agent_name) is None:
            raise ValueError(f"Agent '{agent_name}' not found. Available: {agents.agent_names}")
        await asyncio.to_thread(self.project_paths.set_current_agent, agent_name)

    async def get_current_model(self, agent_name: str) -> str | None:
        """Return a valid workspace model preference for an Agent."""
        model_name = self.project_paths.get_current_model(agent_name)
        if model_name is None:
            return None
        llms = await self.load_llms()
        return model_name if llms.get_llm_config(model_name) is not None else None

    async def set_current_model(self, agent_name: str, model_name: str) -> None:
        """Persist an Agent's model preference in this workspace's project state."""
        await self.get_agent(agent_name)
        await self.get_llm(model_name)
        await asyncio.to_thread(self.project_paths.set_current_model, agent_name, model_name)

    async def update_default_agent(self, agent_name: str) -> None:
        """Compatibility alias for workspace-scoped Agent selection."""
        await self.set_current_agent(agent_name)

    async def add_agent_skill_pattern(self, agent_name: str, skill_pattern: str) -> bool:
        """Append a skill pattern to an agent's config if it is not already present."""
        await self._ensure_named_override(agent_name, subagent=False, include_fields=("skills",))
        changed = await BatchAgentConfig.add_agent_skill_pattern(
            file_path=self.agents_file,
            agent_name=agent_name,
            skill_pattern=skill_pattern,
            dir_path=self.agents_dir,
        )
        self._agents = None  # Invalidate cache
        return changed

    async def _ensure_named_override(
        self,
        name: str,
        *,
        subagent: bool,
        include_fields: tuple[str, ...] = (),
    ) -> None:
        """Create the smallest valid override before mutating one field."""
        await self.ensure_config_dir()
        target_dir = self.subagents_dir if subagent else self.agents_dir
        target_file = target_dir / f"{name}.yml"
        aggregate_file = self.subagents_file if subagent else self.agents_file
        if target_file.is_file() or aggregate_file.is_file():
            return

        source_dir_name = CONFIG_SUBAGENTS_DIR.name if subagent else CONFIG_AGENTS_DIR.name
        source_dir = self.default_config_dir / source_dir_name
        source_file = source_dir / f"{name}.yml"
        if not source_file.is_file():
            for candidate in source_dir.glob("*.yml"):
                data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                if data.get("name") == name:
                    source_file = candidate
                    break
        if not source_file.is_file():
            raise ValueError(f"Cannot create an override for unknown config item '{name}'")

        source_data = yaml.safe_load(await asyncio.to_thread(source_file.read_text, encoding="utf-8")) or {}
        minimal_override = {
            "version": source_data.get("version"),
            "name": source_data.get("name", name),
            "llm": source_data.get("llm"),
        }
        for field_name in include_fields:
            if field_name in source_data:
                minimal_override[field_name] = source_data[field_name]
        if not minimal_override["llm"]:
            raise ValueError(f"Cannot create an override without an LLM for '{name}'")

        target_dir.mkdir(parents=True, exist_ok=True)
        yaml_content = yaml.safe_dump(
            minimal_override,
            sort_keys=False,
            allow_unicode=True,
        )
        await asyncio.to_thread(target_file.write_text, yaml_content, encoding="utf-8")

    # === Cache management ===

    def invalidate_cache(self) -> None:
        """Clear all cached configs."""
        self._llms = None
        self._checkpointers = None
        self._agents = None
        self._subagents = None
        self._mcp = None
