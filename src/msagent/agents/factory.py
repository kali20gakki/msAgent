"""Agent factory using deepagents runtime primitives."""

from __future__ import annotations

import logging
import tempfile
from fnmatch import fnmatch
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware

from msagent.agents.local_context import ensure_local_context_prompt
from msagent.core.constants import CONFIG_CONVERSATION_HISTORY_DIR
from msagent.llms.factory import LLMFactory
from msagent.middlewares.tool_result_eviction import ToolResultEvictionMiddleware
from msagent.tools.catalog import fetch_skills, fetch_tools, get_skill, get_tool, run_tool
from msagent.tools.factory import ToolFactory
from msagent.tools.web_search import web_search
from msagent.utils.deepagents_compat import patch_deepagents_windows_absolute_paths

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from msagent.configs import AgentConfig, LLMConfig


logger = logging.getLogger(__name__)


class _ToolPatternFilterMiddleware(AgentMiddleware[Any, Any, Any]):
    """Filter tool list at model-call time so deepagents defaults are constrained too."""

    def __init__(
        self,
        *,
        filter_tools: Callable[[list[Any]], list[Any]],
    ) -> None:
        self._filter_tools = filter_tools

    def wrap_model_call(self, request, handler):
        filtered_tools = self._filter_tools(list(getattr(request, "tools", []) or []))
        request = request.override(tools=filtered_tools)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        filtered_tools = self._filter_tools(list(getattr(request, "tools", []) or []))
        request = request.override(tools=filtered_tools)
        return await handler(request)


class AgentFactory:
    """Factory for creating deepagents-based graphs."""

    def __init__(
        self,
        *,
        llm_factory: LLMFactory | None = None,
        tool_factory: ToolFactory | None = None,
    ) -> None:
        self.llm_factory = llm_factory or LLMFactory()
        self.tool_factory = tool_factory or ToolFactory()

    async def create(
        self,
        config: AgentConfig,
        working_dir: Path | None = None,
        context_schema: type[Any] | None = None,
        mcp_client: Any | None = None,
        skills_dir: Path | list[Path] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        llm_config: LLMConfig | None = None,
        sandbox_bindings: list[Any] | None = None,
        interrupt_on: dict[str, bool | dict[str, Any]] | None = None,
    ) -> CompiledStateGraph:
        del sandbox_bindings

        patch_deepagents_windows_absolute_paths()
        working_dir = (working_dir or Path.cwd()).resolve()
        resolved_llm = llm_config or config.llm
        retry_cfg = getattr(config, "retry", None)
        model_retry_cfg = getattr(retry_cfg, "model", None) if retry_cfg is not None else None
        tool_retry_cfg = getattr(retry_cfg, "tool", None) if retry_cfg is not None else None
        llm_max_retries: int | None = None
        llm_timeout_seconds: float | None = None
        if retry_cfg is not None and retry_cfg.enabled:
            model_enabled = (
                getattr(model_retry_cfg, "enabled", True)
                if model_retry_cfg is not None
                else True
            )
            if model_enabled:
                llm_max_retries = int(getattr(model_retry_cfg, "max_retries", 0))
                llm_timeout_seconds = (
                    float(getattr(model_retry_cfg, "timeout"))
                    if model_retry_cfg is not None
                    and getattr(model_retry_cfg, "timeout", None) is not None
                    else None
                )
            else:
                llm_max_retries = 0
        elif retry_cfg is not None and not retry_cfg.enabled:
            llm_max_retries = 0

        model = self.llm_factory.create(
            resolved_llm,
            max_retries=llm_max_retries,
            timeout_seconds=llm_timeout_seconds,
        )

        runtime_tools: list[BaseTool] = [
            fetch_tools,
            get_tool,
            run_tool,
            fetch_skills,
            get_skill,
            web_search,
        ]
        mcp_tools: list[BaseTool] = []
        mcp_module_map: dict[str, str] = {}
        if mcp_client is not None:
            loaded = await mcp_client.tools()
            mcp_tools = list(loaded or [])
            mcp_module_map = dict(getattr(mcp_client, "module_map", {}) or {})

        tool_patterns = list(config.tools.patterns or []) if config.tools is not None else []
        mcp_servers = self._collect_mcp_servers(mcp_client, mcp_module_map)
        positive_patterns, negative_patterns = self._compile_tool_patterns(tool_patterns)

        if config.tools is not None:
            runtime_tools, mcp_tools = self._filter_tools_by_patterns(
                runtime_tools=runtime_tools,
                mcp_tools=mcp_tools,
                positive_patterns=positive_patterns,
                negative_patterns=negative_patterns,
                mcp_module_map=mcp_module_map,
                mcp_servers=mcp_servers,
            )

        tool_timeout = None
        if config.tools is not None:
            tool_timeout = config.tools.execution_timeout_seconds
        if tool_timeout:
            mcp_tools = self.tool_factory.wrap_tools_with_timeout(
                mcp_tools,
                timeout_seconds=float(tool_timeout),
                source="agent",
            )
            runtime_tools = self.tool_factory.wrap_tools_with_timeout(
                runtime_tools,
                timeout_seconds=float(tool_timeout),
                source="agent",
            )

        all_tools = [*runtime_tools, *mcp_tools]

        skills_sources = self._resolve_existing_paths(skills_dir)
        memory_sources = self._ensure_memory_file(working_dir)
        enable_skills_middleware = self._should_enable_skills_middleware(
            config=config,
            skills_sources=skills_sources,
        )

        def _filter_request_tools(tools: list[BaseTool]) -> list[BaseTool]:
            return self._filter_tool_objects_by_patterns(
                tools=tools,
                positive_patterns=positive_patterns,
                negative_patterns=negative_patterns,
                mcp_module_map=mcp_module_map,
                mcp_servers=mcp_servers,
            )

        middleware: list[AgentMiddleware[Any, Any, Any]] = []
        agent_backend = self._build_composite_backend(working_dir)
        metadata_backend = FilesystemBackend(virtual_mode=False)
        if memory_sources:
            middleware.append(
                MemoryMiddleware(
                    backend=metadata_backend,
                    sources=memory_sources,
                )
            )
        if enable_skills_middleware:
            middleware.append(
                SkillsMiddleware(
                    backend=metadata_backend,
                    sources=skills_sources,
                )
            )
        tool_output_max_tokens = (
            int(getattr(config.tools, "output_max_tokens", 0))
            if config.tools is not None
            and getattr(config.tools, "output_max_tokens", None) is not None
            else None
        )
        if tool_output_max_tokens is not None and tool_output_max_tokens > 0:
            middleware.append(
                ToolResultEvictionMiddleware(
                    backend=agent_backend,
                    tool_token_limit_before_evict=tool_output_max_tokens,
                )
            )

        raw_system_prompt = config.prompt
        if isinstance(raw_system_prompt, list):
            raw_system_prompt = "\n\n".join(str(item) for item in raw_system_prompt)
        else:
            raw_system_prompt = str(raw_system_prompt)
        system_prompt = ensure_local_context_prompt(raw_system_prompt)
        kwargs: dict[str, Any] = {
            "model": model,
            "tools": all_tools,
            "system_prompt": system_prompt,
            "backend": agent_backend,
            "checkpointer": checkpointer,
            "name": config.name,
            "middleware": middleware,
        }
        if interrupt_on:
            kwargs["interrupt_on"] = interrupt_on
        if (
            retry_cfg is not None
            and retry_cfg.enabled
            and (getattr(tool_retry_cfg, "enabled", True) if tool_retry_cfg is not None else True)
            and int(getattr(tool_retry_cfg, "max_retries", 0)) > 0
        ):
            tool_names = (
                list(getattr(tool_retry_cfg, "tools"))
                if tool_retry_cfg is not None
                and getattr(tool_retry_cfg, "tools", None) is not None
                else None
            )
            retry_on = self._resolve_retry_on_exceptions(
                list(getattr(tool_retry_cfg, "retry_on"))
                if tool_retry_cfg is not None
                and getattr(tool_retry_cfg, "retry_on", None) is not None
                else []
            )
            tool_retry_kwargs: dict[str, Any] = {
                "max_retries": int(getattr(tool_retry_cfg, "max_retries")),
                "tools": tool_names,
                "on_failure": getattr(tool_retry_cfg, "on_failure", "continue"),
                "backoff_factor": float(
                    getattr(tool_retry_cfg, "backoff_factor", 2.0)
                ),
                "initial_delay": float(getattr(tool_retry_cfg, "initial_delay", 1.0)),
                "max_delay": float(getattr(tool_retry_cfg, "max_delay", 60.0)),
                "jitter": bool(getattr(tool_retry_cfg, "jitter", True)),
            }
            if retry_on is not None:
                tool_retry_kwargs["retry_on"] = retry_on
            middleware.append(
                ToolRetryMiddleware(**tool_retry_kwargs)
            )

        middleware.append(_ToolPatternFilterMiddleware(filter_tools=_filter_request_tools))
        if context_schema is not None:
            kwargs["context_schema"] = context_schema

        graph = create_deep_agent(**kwargs)

        # Keep CLI-compatible metadata caches for /tools and runtime context.
        setattr(graph, "_agent_backend", agent_backend)
        setattr(graph, "_llm_tools", all_tools)
        setattr(graph, "_tools_in_catalog", list(all_tools))
        return graph

    def _filter_tools_by_patterns(
        self,
        *,
        runtime_tools: list[BaseTool],
        mcp_tools: list[BaseTool],
        positive_patterns: list[tuple[str, str, str]],
        negative_patterns: list[tuple[str, str, str]],
        mcp_module_map: dict[str, str],
        mcp_servers: set[str],
    ) -> tuple[list[BaseTool], list[BaseTool]]:
        filtered_runtime: list[BaseTool] = []
        for tool in runtime_tools:
            tool_name = self._tool_name(tool)
            modules = self._runtime_modules_for_tool(tool_name)
            names = self._runtime_names_for_tool(tool_name)
            if self._tool_matches_patterns(
                positive_patterns=positive_patterns,
                negative_patterns=negative_patterns,
                category="impl",
                modules=modules,
                names=names,
            ):
                filtered_runtime.append(tool)

        filtered_mcp: list[BaseTool] = []
        for tool in mcp_tools:
            tool_name = self._tool_name(tool)
            module, raw_name = self._resolve_mcp_tool_identity(
                tool_name=tool_name,
                mcp_module_map=mcp_module_map,
                mcp_servers=mcp_servers,
            )
            names = {tool_name}
            if raw_name:
                names.add(raw_name)
            if self._tool_matches_patterns(
                positive_patterns=positive_patterns,
                negative_patterns=negative_patterns,
                category="mcp",
                modules={module},
                names=names,
            ):
                filtered_mcp.append(tool)

        return filtered_runtime, filtered_mcp

    def _filter_tool_objects_by_patterns(
        self,
        *,
        tools: list[BaseTool],
        positive_patterns: list[tuple[str, str, str]],
        negative_patterns: list[tuple[str, str, str]],
        mcp_module_map: dict[str, str],
        mcp_servers: set[str],
    ) -> list[BaseTool]:
        filtered: list[BaseTool] = []
        for tool in tools:
            tool_name = self._tool_name(tool)
            module, raw_name = self._resolve_mcp_tool_identity(
                tool_name=tool_name,
                mcp_module_map=mcp_module_map,
                mcp_servers=mcp_servers,
            )

            if module != "unknown":
                names = {tool_name}
                if raw_name:
                    names.add(raw_name)
                if self._tool_matches_patterns(
                    positive_patterns=positive_patterns,
                    negative_patterns=negative_patterns,
                    category="mcp",
                    modules={module},
                    names=names,
                ):
                    filtered.append(tool)
                continue

            if self._tool_matches_patterns(
                positive_patterns=positive_patterns,
                negative_patterns=negative_patterns,
                category="impl",
                modules=self._runtime_modules_for_tool(tool_name),
                names=self._runtime_names_for_tool(tool_name),
            ):
                filtered.append(tool)
        return filtered

    @staticmethod
    def _tool_name(tool: BaseTool) -> str:
        return str(getattr(tool, "name", "") or "")

    @staticmethod
    def _runtime_modules_for_tool(tool_name: str) -> set[str]:
        del tool_name
        return {"deepagents"}

    @staticmethod
    def _runtime_names_for_tool(tool_name: str) -> set[str]:
        return {tool_name}

    @staticmethod
    def _collect_mcp_servers(
        mcp_client: Any | None,
        mcp_module_map: dict[str, str],
    ) -> set[str]:
        servers = {
            module_ref.split(":", 1)[1]
            for module_ref in mcp_module_map.values()
            if module_ref.startswith("mcp:") and ":" in module_ref
        }
        config = getattr(mcp_client, "config", None)
        config_servers = getattr(config, "servers", None)
        if isinstance(config_servers, dict):
            servers.update(str(name) for name in config_servers.keys())
        return servers

    def _resolve_mcp_tool_identity(
        self,
        *,
        tool_name: str,
        mcp_module_map: dict[str, str],
        mcp_servers: set[str],
    ) -> tuple[str, str]:
        module_ref = mcp_module_map.get(tool_name, "")
        module = ""
        if module_ref.startswith("mcp:") and ":" in module_ref:
            module = module_ref.split(":", 1)[1]

        if not module:
            parsed_module, parsed_name = self._parse_mcp_prefixed_tool_name(
                tool_name=tool_name,
                mcp_servers=mcp_servers,
            )
            if parsed_module:
                return parsed_module, parsed_name
            return "unknown", tool_name

        parsed_module, parsed_name = self._parse_mcp_prefixed_tool_name(
            tool_name=tool_name,
            mcp_servers={module},
        )
        if parsed_module:
            return module, parsed_name
        return module, tool_name

    @staticmethod
    def _parse_mcp_prefixed_tool_name(
        *,
        tool_name: str,
        mcp_servers: set[str],
    ) -> tuple[str | None, str]:
        for server in sorted(mcp_servers, key=len, reverse=True):
            for separator in ("__", "_"):
                prefix = f"{server}{separator}"
                if tool_name.startswith(prefix):
                    return server, tool_name[len(prefix) :]
        return None, tool_name

    @staticmethod
    def _compile_tool_patterns(
        patterns: list[str],
    ) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
        positives: list[tuple[str, str, str]] = []
        negatives: list[tuple[str, str, str]] = []

        for raw_pattern in patterns:
            if not raw_pattern:
                continue
            is_negative = raw_pattern.startswith("!")
            pattern = raw_pattern[1:] if is_negative else raw_pattern
            parts = pattern.split(":")
            if len(parts) != 3:
                logger.warning(
                    "Ignoring invalid tool pattern '%s'. Expected format 'category:module:name'.",
                    raw_pattern,
                )
                continue
            entry = (parts[0], parts[1], parts[2])
            if is_negative:
                negatives.append(entry)
            else:
                positives.append(entry)
        return positives, negatives

    @staticmethod
    def _tool_matches_patterns(
        *,
        positive_patterns: list[tuple[str, str, str]],
        negative_patterns: list[tuple[str, str, str]],
        category: str,
        modules: set[str],
        names: set[str],
    ) -> bool:
        if not positive_patterns:
            return False

        def _match(pattern: tuple[str, str, str]) -> bool:
            category_p, module_p, name_p = pattern
            if not fnmatch(category, category_p):
                return False
            if not any(fnmatch(module, module_p) for module in modules):
                return False
            return any(fnmatch(name, name_p) for name in names)

        return any(_match(p) for p in positive_patterns) and not any(
            _match(p) for p in negative_patterns
        )

    @staticmethod
    def _resolve_existing_paths(paths: Path | list[Path] | None) -> list[str]:
        if paths is None:
            return []
        candidates = [paths] if isinstance(paths, Path) else list(paths)
        return [str(path) for path in candidates if path.exists()]

    @staticmethod
    def _should_enable_skills_middleware(
        *,
        config: AgentConfig,
        skills_sources: list[str],
    ) -> bool:
        if not skills_sources:
            return False
        skills_config = getattr(config, "skills", None)
        if skills_config is None:
            return False
        patterns = list(getattr(skills_config, "patterns", []) or [])
        return any(pattern and not pattern.startswith("!") for pattern in patterns)

    @staticmethod
    def _ensure_memory_file(working_dir: Path) -> list[str]:
        from msagent.tools.internal.memory import ensure_memory_file

        memory_file = ensure_memory_file(working_dir)
        return [str(memory_file)]

    @staticmethod
    def _build_composite_backend(working_dir: Path) -> CompositeBackend:
        local_backend = LocalShellBackend(
            root_dir=str(working_dir),
            inherit_env=True,
        )
        large_results_backend = FilesystemBackend(
            root_dir=tempfile.mkdtemp(prefix="msagent_large_tool_results_"),
            virtual_mode=True,
        )
        conversation_history_backend = FilesystemBackend(
            root_dir=working_dir / CONFIG_CONVERSATION_HISTORY_DIR,
            virtual_mode=True,
        )
        return CompositeBackend(
            default=local_backend,
            routes={
                "/large_tool_results/": large_results_backend,
                "/conversation_history/": conversation_history_backend,
            },
        )

    @staticmethod
    def _build_skills_text(skills: list[Any], use_catalog: bool) -> str:
        if not skills:
            return ""

        lines = [
            "Available skills:",
            "Always call `get_skill(name, category)` before using a skill.",
        ]
        for skill in skills:
            scripts: list[str] = getattr(
                skill, "get_script_relative_paths", lambda: []
            )()
            scripts_text = (
                f" (scripts: {', '.join(f'`{p}`' for p in scripts)})"
                if scripts
                else ""
            )
            lines.append(
                f"- {getattr(skill, 'display_name', getattr(skill, 'name', 'unknown'))}: "
                f"{getattr(skill, 'description', '')}{scripts_text}"
            )

        if not use_catalog:
            lines.append(
                "When scripts are present under `scripts/`, prefer running those scripts."
            )
        return "\n".join(lines)

    @staticmethod
    def _resolve_retry_on_exceptions(
        names: list[str],
    ) -> tuple[type[Exception], ...] | None:
        if not names:
            return None

        resolved: list[type[Exception]] = []
        for raw_name in names:
            name = str(raw_name).strip()
            if not name:
                continue

            exc_type: type[Exception] | None = None
            if "." in name:
                module_name, _, attr_name = name.rpartition(".")
                try:
                    module = import_module(module_name)
                    candidate = getattr(module, attr_name, None)
                except Exception:
                    candidate = None
            else:
                import builtins

                candidate = getattr(builtins, name, None)

            if isinstance(candidate, type) and issubclass(candidate, Exception):
                exc_type = candidate

            if exc_type is None:
                logger.warning(
                    "Ignoring retry.tool.retry_on entry '%s': not a resolvable Exception subclass.",
                    name,
                )
                continue

            resolved.append(exc_type)

        return tuple(resolved) if resolved else None
