"""Central registry for loading, saving, and caching configurations."""

from __future__ import annotations

import asyncio
import json
import shutil
from importlib.resources import files
from pathlib import Path

import yaml

from msagent.configs.agent import (
    AgentConfig,
    BatchAgentConfig,
    BatchSubAgentConfig,
    SubAgentConfig,
)
from msagent.configs.approval import ToolApprovalConfig
from msagent.configs.checkpointer import BatchCheckpointerConfig, CheckpointerConfig
from msagent.configs.llm import BatchLLMConfig, LLMConfig
from msagent.configs.mcp import MCPConfig
from msagent.configs.sandbox import BatchSandboxConfig, SandboxConfig
from msagent.core.constants import (
    CONFIG_AGENTS_DIR,
    CONFIG_AGENTS_FILE_NAME,
    CONFIG_APPROVAL_FILE_NAME,
    CONFIG_CHECKPOINTERS_DIR,
    CONFIG_CHECKPOINTERS_FILE_NAME,
    CONFIG_CHECKPOINTS_URL_FILE_NAME,
    CONFIG_DIR_NAME,
    CONFIG_LLMS_DIR,
    CONFIG_LLMS_FILE_NAME,
    CONFIG_MCP_FILE_NAME,
    CONFIG_MEMORY_FILE_NAME,
    CONFIG_SANDBOXES_DIR,
    CONFIG_SUBAGENTS_DIR,
    CONFIG_SUBAGENTS_FILE_NAME,
)


class ConfigRegistry:
    """Central registry for loading, saving, and caching all configurations."""

    def __init__(self, working_dir: Path):
        self.working_dir = working_dir
        self.config_dir = working_dir / CONFIG_DIR_NAME

        # Lazy-loaded caches
        self._llms: BatchLLMConfig | None = None
        self._checkpointers: BatchCheckpointerConfig | None = None
        self._agents: BatchAgentConfig | None = None
        self._subagents: BatchSubAgentConfig | None = None
        self._sandboxes: BatchSandboxConfig | None = None
        self._mcp: MCPConfig | None = None
        self._approval: ToolApprovalConfig | None = None

    # === Setup ===

    async def ensure_config_dir(self) -> None:
        """Ensure config directory exists, copy from template if needed."""
        template_config_dir = Path(str(files("resources") / "configs" / "default"))

        if not self.config_dir.exists():
            await asyncio.to_thread(
                shutil.copytree,
                template_config_dir,
                self.config_dir,
                ignore=shutil.ignore_patterns(
                    CONFIG_CHECKPOINTS_URL_FILE_NAME.name.replace(".db", ".*"),
                ),
            )
        else:
            await self._copy_missing_template_files(template_config_dir)
            await self._normalize_legacy_defaults(template_config_dir)

        # Ensure CONFIG_DIR_NAME is ignored in git (local-only, not committed)
        git_info_exclude = self.working_dir / ".git" / "info" / "exclude"
        if git_info_exclude.parent.exists():
            try:
                existing_content = ""
                if git_info_exclude.exists():
                    existing_content = await asyncio.to_thread(
                        git_info_exclude.read_text
                    )

                ignore_pattern = f"{CONFIG_DIR_NAME}/"
                if ignore_pattern not in existing_content:

                    def write_exclude():
                        with git_info_exclude.open("a") as f:
                            f.write(f"\n# msAgent configuration\n{ignore_pattern}\n")

                    await asyncio.to_thread(write_exclude)
            except Exception:
                pass

    async def _copy_missing_template_files(self, template_config_dir: Path) -> None:
        """Copy newly-added default template files into existing config directories."""

        def copy_missing() -> None:
            for template_path in template_config_dir.rglob("*"):
                if not template_path.is_file():
                    continue
                relative_path = template_path.relative_to(template_config_dir)
                target_path = self.config_dir / relative_path
                if not target_path.exists():
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(template_path, target_path)

        await asyncio.to_thread(copy_missing)

    async def _normalize_legacy_defaults(self, template_config_dir: Path) -> None:
        """Gently upgrade legacy default files without overwriting user customizations."""

        def normalize() -> None:
            prompt_path = self.config_dir / "prompts" / "agents" / "general.md"
            template_prompt_path = template_config_dir / "prompts" / "agents" / "general.md"
            if prompt_path.exists() and template_prompt_path.exists():
                prompt_text = prompt_path.read_text(encoding="utf-8")
                if (
                    "You are a versatile AI assistant" in prompt_text
                    and "Ascend NPU Profiling 性能分析助手" not in prompt_text
                ):
                    shutil.copy2(template_prompt_path, prompt_path)

            mcp_path = self.config_dir / CONFIG_MCP_FILE_NAME.name
            template_mcp_path = template_config_dir / CONFIG_MCP_FILE_NAME.name
            if mcp_path.exists() and template_mcp_path.exists():
                current = json.loads(mcp_path.read_text(encoding="utf-8"))
                template = json.loads(template_mcp_path.read_text(encoding="utf-8"))
                current_servers = current.setdefault("mcpServers", {})
                template_servers = template.get("mcpServers", {})
                for name, server in template_servers.items():
                    current_servers.setdefault(name, server)
                mcp_path.write_text(
                    json.dumps(current, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            general_agent_path = self.config_dir / "agents" / "general.yml"
            template_general_agent_path = template_config_dir / "agents" / "general.yml"
            if general_agent_path.exists() and template_general_agent_path.exists():
                current_agent = yaml.safe_load(
                    general_agent_path.read_text(encoding="utf-8")
                ) or {}
                template_agent = yaml.safe_load(
                    template_general_agent_path.read_text(encoding="utf-8")
                ) or {}
                current_patterns = (
                    current_agent.setdefault("tools", {}).setdefault("patterns", [])
                )
                template_patterns = template_agent.get("tools", {}).get("patterns", [])
                for pattern in template_patterns:
                    if pattern not in current_patterns:
                        current_patterns.append(pattern)
                general_agent_path.write_text(
                    yaml.safe_dump(
                        current_agent, sort_keys=False, allow_unicode=True
                    ),
                    encoding="utf-8",
                )

        await asyncio.to_thread(normalize)

    # === LLM configs ===

    async def load_llms(self, force_reload: bool = False) -> BatchLLMConfig:
        """Load all LLM configs (cached)."""
        if self._llms is None or force_reload:
            await self.ensure_config_dir()
            self._llms = await BatchLLMConfig.from_yaml(
                file_path=self.working_dir / CONFIG_LLMS_FILE_NAME,
                dir_path=self.working_dir / CONFIG_LLMS_DIR,
            )
        return self._llms

    async def get_llm(self, alias: str) -> LLMConfig:
        """Get single LLM by alias."""
        llms = await self.load_llms()
        llm = llms.get_llm_config(alias)
        if llm:
            return llm
        raise ValueError(f"LLM '{alias}' not found. Available: {llms.llm_names}")

    # === Checkpointer configs ===

    async def load_checkpointers(
        self, force_reload: bool = False
    ) -> BatchCheckpointerConfig:
        """Load all checkpointer configs (cached)."""
        if self._checkpointers is None or force_reload:
            await self.ensure_config_dir()
            self._checkpointers = await BatchCheckpointerConfig.from_yaml(
                file_path=self.working_dir / CONFIG_CHECKPOINTERS_FILE_NAME,
                dir_path=self.working_dir / CONFIG_CHECKPOINTERS_DIR,
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

            llm_config = None
            if (self.working_dir / CONFIG_LLMS_FILE_NAME).exists() or (
                self.working_dir / CONFIG_LLMS_DIR
            ).exists():
                llm_config = await self.load_llms()

            self._subagents = await BatchSubAgentConfig.from_yaml(
                file_path=self.working_dir / CONFIG_SUBAGENTS_FILE_NAME,
                dir_path=self.working_dir / CONFIG_SUBAGENTS_DIR,
                batch_llm_config=llm_config,
            )
        return self._subagents

    async def get_subagent(self, name: str) -> SubAgentConfig | None:
        """Get single subagent by name."""
        subagents = await self.load_subagents()
        return subagents.get_subagent_config(name)

    # === Sandbox configs ===

    async def load_sandboxes(self, force_reload: bool = False) -> BatchSandboxConfig:
        """Load all sandbox configs (cached)."""
        if self._sandboxes is None or force_reload:
            await self.ensure_config_dir()
            self._sandboxes = await BatchSandboxConfig.from_yaml(
                dir_path=self.working_dir / CONFIG_SANDBOXES_DIR,
            )
        return self._sandboxes

    async def get_sandbox(self, name: str) -> SandboxConfig:
        """Get single sandbox by name."""
        sandboxes = await self.load_sandboxes()
        sandbox = sandboxes.get_sandbox_config(name)
        if sandbox:
            return sandbox
        raise ValueError(
            f"Sandbox '{name}' not found. Available: {sandboxes.sandbox_names}"
        )

    # === Agent configs ===

    async def load_agents(self, force_reload: bool = False) -> BatchAgentConfig:
        """Load all agent configs with resolved references (cached)."""
        if self._agents is None or force_reload:
            await self.ensure_config_dir()

            llm_config = None
            checkpointer_config = None

            if (self.working_dir / CONFIG_LLMS_FILE_NAME).exists() or (
                self.working_dir / CONFIG_LLMS_DIR
            ).exists():
                llm_config = await self.load_llms()

            if (self.working_dir / CONFIG_CHECKPOINTERS_FILE_NAME).exists() or (
                self.working_dir / CONFIG_CHECKPOINTERS_DIR
            ).exists():
                checkpointer_config = await self.load_checkpointers()

            subagents_config = None
            if (self.working_dir / CONFIG_SUBAGENTS_FILE_NAME).exists() or (
                self.working_dir / CONFIG_SUBAGENTS_DIR
            ).exists():
                subagents_config = await self.load_subagents()

            sandboxes_config = None
            if (self.working_dir / CONFIG_SANDBOXES_DIR).exists():
                sandboxes_config = await self.load_sandboxes()

            self._agents = await BatchAgentConfig.from_yaml(
                file_path=self.working_dir / CONFIG_AGENTS_FILE_NAME,
                dir_path=self.working_dir / CONFIG_AGENTS_DIR,
                batch_llm_config=llm_config,
                batch_checkpointer_config=checkpointer_config,
                batch_subagent_config=subagents_config,
                batch_sandbox_config=sandboxes_config,
            )
        return self._agents

    async def get_agent(self, name: str | None = None) -> AgentConfig:
        """Get single agent by name, or default agent if name is None."""
        agents = await self.load_agents()
        agent = agents.get_agent_config(name)
        if agent:
            return agent
        raise ValueError(f"Agent '{name}' not found. Available: {agents.agent_names}")

    # === MCP config ===

    async def load_mcp(self, force_reload: bool = False) -> MCPConfig:
        """Load MCP server config (cached)."""
        if self._mcp is None or force_reload:
            await self.ensure_config_dir()
            self._mcp = await MCPConfig.from_json(
                self.working_dir / CONFIG_MCP_FILE_NAME
            )
        return self._mcp

    async def save_mcp(self, config: MCPConfig) -> None:
        """Save MCP config to file."""
        config.to_json(self.working_dir / CONFIG_MCP_FILE_NAME)
        self._mcp = config

    # === Approval config ===

    def load_approval(self, force_reload: bool = False) -> ToolApprovalConfig:
        """Load tool approval config (cached)."""
        if self._approval is None or force_reload:
            self._approval = ToolApprovalConfig.from_json_file(
                self.working_dir / CONFIG_APPROVAL_FILE_NAME
            )
        return self._approval

    def save_approval(self, config: ToolApprovalConfig) -> None:
        """Save approval config to file."""
        config.save_to_json_file(self.working_dir / CONFIG_APPROVAL_FILE_NAME)
        self._approval = config

    # === User memory ===

    async def load_user_memory(self) -> str:
        """Load user memory from project-specific memory file.

        Returns:
            Formatted user memory string for prompt injection, or empty string if no memory
        """
        memory_path = self.working_dir / CONFIG_MEMORY_FILE_NAME
        if memory_path.exists():
            content = await asyncio.to_thread(memory_path.read_text)
            content = content.strip()
            if content:
                return f"<user-memory>\n{content}\n</user-memory>"
        return ""

    # === Update operations ===

    async def update_agent_llm(self, agent_name: str, llm_alias: str) -> None:
        """Update an agent's LLM reference and persist."""
        await BatchAgentConfig.update_agent_llm(
            file_path=self.working_dir / CONFIG_AGENTS_FILE_NAME,
            agent_name=agent_name,
            new_llm_name=llm_alias,
            dir_path=self.working_dir / CONFIG_AGENTS_DIR,
        )
        self._agents = None  # Invalidate cache

    async def update_subagent_llm(self, subagent_name: str, llm_alias: str) -> None:
        """Update a subagent's LLM reference and persist."""
        await BatchAgentConfig.update_agent_llm(
            file_path=self.working_dir / CONFIG_SUBAGENTS_FILE_NAME,
            agent_name=subagent_name,
            new_llm_name=llm_alias,
            dir_path=self.working_dir / CONFIG_SUBAGENTS_DIR,
        )
        self._subagents = None  # Invalidate cache

    async def update_default_agent(self, agent_name: str) -> None:
        """Set the default agent and persist."""
        await BatchAgentConfig.update_default_agent(
            file_path=self.working_dir / CONFIG_AGENTS_FILE_NAME,
            agent_name=agent_name,
            dir_path=self.working_dir / CONFIG_AGENTS_DIR,
        )
        self._agents = None  # Invalidate cache

    # === Cache management ===

    def invalidate_cache(self) -> None:
        """Clear all cached configs."""
        self._llms = None
        self._checkpointers = None
        self._agents = None
        self._subagents = None
        self._sandboxes = None
        self._mcp = None
        self._approval = None
