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

"""Agent factory using deepagents runtime primitives."""

from __future__ import annotations

import logging
import string
import os
import tempfile
from fnmatch import fnmatch
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware
import httpx
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware

from msagent.agents.local_context import ensure_local_context_prompt
from msagent.configs import AgentConfig, BaseAgentConfig, RetryPolicyConfig, SubAgentConfig
from msagent.core.constants import CONFIG_CONVERSATION_HISTORY_DIR
from msagent.llms.factory import LLMFactory
from msagent.middlewares.tool_result_eviction import ToolResultEvictionMiddleware
from msagent.tools.catalog import (
    fetch_skills,
    fetch_tools,
    get_skill,
    get_tool,
    run_tool,
)
from msagent.tools.factory import ToolFactory
from msagent.tools.web_search import web_search
from msagent.utils.deepagents_compat import patch_deepagents_windows_absolute_paths

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from msagent.configs import LLMConfig


logger = logging.getLogger(__name__)

_TAVILY_SERVER_KEYWORDS = ("tavily",)
_TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
_TAVILY_VALIDATE_URL = "https://api.tavily.com/usage"
_TAVILY_VALIDATE_TIMEOUT_SECONDS = 5.0
_TAVILY_KEY_VALIDATION_CACHE: dict[str, bool] = {}
_SEARCH_TOOL_NAME_KEYWORDS = ("search", "web_search")
_SEARCH_TOOL_DESCRIPTION_KEYWORDS = ("search", "web", "internet", "query")


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


class _SystemMessageMiddleware(AgentMiddleware[Any, Any, Any]):
    """Populate system message placeholders from runtime AgentContext.template_vars."""

    class _SafeTemplateFormatter(string.Formatter):
        """String formatter that leaves unknown placeholders unchanged."""

        def __init__(self, context: dict[str, Any]) -> None:
            super().__init__()
            self._context = context

        def get_value(self, key: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if isinstance(key, str):
                if key in self._context:
                    return self._context[key]
                return "{" + key + "}"
            return super().get_value(key, args, kwargs)

    @classmethod
    def _safe_render_templates(cls, data: Any, context: dict[str, Any] | None) -> Any:
        formatter = cls._SafeTemplateFormatter(context or {})
        if isinstance(data, str):
            return cls._safe_render_text(data, formatter)
        if isinstance(data, list):
            return [cls._render_text_block(item, formatter) for item in data]
        return data

    @staticmethod
    def _safe_render_text(text: str, formatter: string.Formatter) -> str:
        try:
            return formatter.vformat(text, (), {})
        except ValueError:
            return text

    @classmethod
    def _render_text_block(cls, item: Any, formatter: string.Formatter) -> Any:
        if not isinstance(item, dict) or item.get("type") != "text" or not isinstance(item.get("text"), str):
            return item

        rendered_text = cls._safe_render_text(item["text"], formatter)
        if rendered_text == item["text"]:
            return item

        rendered_item = dict(item)
        rendered_item["text"] = rendered_text
        return rendered_item

    @staticmethod
    def _render_request_system_message(request):
        system_message = getattr(request, "system_message", None)
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None) if runtime is not None else None
        template_vars = getattr(context, "template_vars", None) if context is not None else None
        if system_message is None or not template_vars:
            return request

        rendered_content = _SystemMessageMiddleware._safe_render_templates(
            system_message.content,
            template_vars,
        )
        if rendered_content == system_message.content:
            return request

        return request.override(system_message=system_message.model_copy(update={"content": rendered_content}))

    def wrap_model_call(self, request, handler):
        request = self._render_request_system_message(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request = self._render_request_system_message(request)
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
        tool_retry_cfg = getattr(retry_cfg, "tool", None) if retry_cfg is not None else None
        llm_max_retries, llm_timeout_seconds = self._resolve_llm_retry_for_policy(retry_cfg)

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

            if await self._should_prefer_search_mcp(
                mcp_client,
                mcp_tools=mcp_tools,
                mcp_module_map=mcp_module_map,
            ):
                runtime_tools = [t for t in runtime_tools if self._tool_name(t) != "web_search"]
        catalog_runtime_tools = list(runtime_tools)
        catalog_mcp_tools = list(mcp_tools)

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
        deepagents_subagents = self._build_deepagents_subagent_specs(
            config=config,
            agent_backend=agent_backend,
            skills_sources=skills_sources,
            catalog_runtime_tools=catalog_runtime_tools,
            catalog_mcp_tools=catalog_mcp_tools,
            mcp_module_map=mcp_module_map,
            mcp_servers=mcp_servers,
            tool_timeout=tool_timeout,
        )
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
            if config.tools is not None and getattr(config.tools, "output_max_tokens", None) is not None
            else None
        )
        if tool_output_max_tokens is not None and tool_output_max_tokens > 0:
            middleware.append(
                ToolResultEvictionMiddleware(
                    backend=agent_backend,
                    tool_token_limit_before_evict=tool_output_max_tokens,
                )
            )
        middleware.append(_SystemMessageMiddleware())

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
        if deepagents_subagents:
            kwargs["subagents"] = deepagents_subagents
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
                if tool_retry_cfg is not None and getattr(tool_retry_cfg, "tools", None) is not None
                else None
            )
            retry_on = self._resolve_retry_on_exceptions(
                list(getattr(tool_retry_cfg, "retry_on"))
                if tool_retry_cfg is not None and getattr(tool_retry_cfg, "retry_on", None) is not None
                else []
            )
            tool_retry_kwargs: dict[str, Any] = {
                "max_retries": int(getattr(tool_retry_cfg, "max_retries")),
                "tools": tool_names,
                "on_failure": getattr(tool_retry_cfg, "on_failure", "continue"),
                "backoff_factor": float(getattr(tool_retry_cfg, "backoff_factor", 2.0)),
                "initial_delay": float(getattr(tool_retry_cfg, "initial_delay", 1.0)),
                "max_delay": float(getattr(tool_retry_cfg, "max_delay", 60.0)),
                "jitter": bool(getattr(tool_retry_cfg, "jitter", True)),
            }
            if retry_on is not None:
                tool_retry_kwargs["retry_on"] = retry_on
            middleware.append(ToolRetryMiddleware(**tool_retry_kwargs))

        middleware.append(_ToolPatternFilterMiddleware(filter_tools=_filter_request_tools))
        if context_schema is not None:
            kwargs["context_schema"] = context_schema

        graph = create_deep_agent(**kwargs)

        # Keep CLI-compatible metadata caches for /tools and runtime context.
        setattr(graph, "_agent_backend", agent_backend)
        setattr(graph, "_llm_tools", all_tools)
        setattr(graph, "_tools_in_catalog", list(all_tools))
        return graph

    @staticmethod
    def _resolve_llm_retry_for_policy(
        retry_cfg: RetryPolicyConfig | None,
    ) -> tuple[int | None, float | None]:
        model_retry_cfg = getattr(retry_cfg, "model", None) if retry_cfg is not None else None
        llm_max_retries: int | None = None
        llm_timeout_seconds: float | None = None
        if retry_cfg is not None and retry_cfg.enabled:
            model_enabled = getattr(model_retry_cfg, "enabled", True) if model_retry_cfg is not None else True
            if model_enabled:
                llm_max_retries = int(getattr(model_retry_cfg, "max_retries", 0))
                llm_timeout_seconds = (
                    float(getattr(model_retry_cfg, "timeout"))
                    if model_retry_cfg is not None and getattr(model_retry_cfg, "timeout", None) is not None
                    else None
                )
            else:
                llm_max_retries = 0
        elif retry_cfg is not None and not retry_cfg.enabled:
            llm_max_retries = 0
        return llm_max_retries, llm_timeout_seconds

    @staticmethod
    def _agent_system_prompt_text(prompt: str | list[str]) -> str:
        if isinstance(prompt, list):
            return "\n\n".join(str(item) for item in prompt)
        return str(prompt)

    def _build_deepagents_subagent_specs(
        self,
        *,
        config: AgentConfig,
        agent_backend: Any,
        skills_sources: list[str],
        catalog_runtime_tools: list[Any],
        catalog_mcp_tools: list[Any],
        mcp_module_map: dict[str, str],
        mcp_servers: set[str],
        tool_timeout: float | None,
    ) -> list[dict[str, Any]]:
        raw = getattr(config, "subagents", None) or []
        if not raw:
            return []

        main_retry = getattr(config, "retry", None)
        specs: list[dict[str, Any]] = []
        for sub in raw:
            if not isinstance(sub, SubAgentConfig):
                continue
            if sub.name == "general-purpose":
                continue

            sub_retry = sub.retry if sub.retry is not None else main_retry
            sub_max_r, sub_timeout = self._resolve_llm_retry_for_policy(sub_retry)
            sub_model = self.llm_factory.create(
                sub.llm,
                max_retries=sub_max_r,
                timeout_seconds=sub_timeout,
            )

            system_prompt = ensure_local_context_prompt(self._agent_system_prompt_text(sub.prompt))
            spec: dict[str, Any] = {
                "name": sub.name,
                "description": sub.description or f"Subagent {sub.name}",
                "system_prompt": system_prompt,
                "model": sub_model,
            }

            stools = sub.tools
            if stools is not None and stools.patterns:
                pos, neg = self._compile_tool_patterns(list(stools.patterns))
                rt, mcp = self._filter_tools_by_patterns(
                    runtime_tools=catalog_runtime_tools,
                    mcp_tools=catalog_mcp_tools,
                    positive_patterns=pos,
                    negative_patterns=neg,
                    mcp_module_map=mcp_module_map,
                    mcp_servers=mcp_servers,
                )
                sub_t_timeout = tool_timeout
                if stools.execution_timeout_seconds is not None:
                    sub_t_timeout = float(stools.execution_timeout_seconds)
                if sub_t_timeout:
                    mcp = self.tool_factory.wrap_tools_with_timeout(
                        mcp,
                        timeout_seconds=float(sub_t_timeout),
                        source="subagent",
                    )
                    rt = self.tool_factory.wrap_tools_with_timeout(
                        rt,
                        timeout_seconds=float(sub_t_timeout),
                        source="subagent",
                    )
                spec["tools"] = [*rt, *mcp]

            if self._should_enable_skills_middleware(config=sub, skills_sources=skills_sources):
                spec["skills"] = list(skills_sources)

            extra_mw = self._subagent_extra_middleware(
                sub=sub,
                agent_backend=agent_backend,
                fallback_retry=main_retry,
            )
            if extra_mw:
                spec["middleware"] = extra_mw

            specs.append(spec)
        return specs

    def _subagent_extra_middleware(
        self,
        *,
        sub: SubAgentConfig,
        agent_backend: Any,
        fallback_retry: RetryPolicyConfig | None,
    ) -> list[AgentMiddleware[Any, Any, Any]]:
        extra: list[AgentMiddleware[Any, Any, Any]] = []
        retry_cfg = sub.retry if sub.retry is not None else fallback_retry
        tool_retry_cfg = getattr(retry_cfg, "tool", None) if retry_cfg is not None else None
        if (
            retry_cfg is not None
            and retry_cfg.enabled
            and (getattr(tool_retry_cfg, "enabled", True) if tool_retry_cfg is not None else True)
            and int(getattr(tool_retry_cfg, "max_retries", 0)) > 0
        ):
            tool_names = (
                list(getattr(tool_retry_cfg, "tools"))
                if tool_retry_cfg is not None and getattr(tool_retry_cfg, "tools", None) is not None
                else None
            )
            retry_on = self._resolve_retry_on_exceptions(
                list(getattr(tool_retry_cfg, "retry_on"))
                if tool_retry_cfg is not None and getattr(tool_retry_cfg, "retry_on", None) is not None
                else []
            )
            tool_retry_kwargs: dict[str, Any] = {
                "max_retries": int(getattr(tool_retry_cfg, "max_retries")),
                "tools": tool_names,
                "on_failure": getattr(tool_retry_cfg, "on_failure", "continue"),
                "backoff_factor": float(getattr(tool_retry_cfg, "backoff_factor", 2.0)),
                "initial_delay": float(getattr(tool_retry_cfg, "initial_delay", 1.0)),
                "max_delay": float(getattr(tool_retry_cfg, "max_delay", 60.0)),
                "jitter": bool(getattr(tool_retry_cfg, "jitter", True)),
            }
            if retry_on is not None:
                tool_retry_kwargs["retry_on"] = retry_on
            extra.append(ToolRetryMiddleware(**tool_retry_kwargs))

        sub_tools = sub.tools
        if (
            sub_tools is not None
            and getattr(sub_tools, "output_max_tokens", None) is not None
            and int(sub_tools.output_max_tokens) > 0
        ):
            extra.append(
                ToolResultEvictionMiddleware(
                    backend=agent_backend,
                    tool_token_limit_before_evict=int(sub_tools.output_max_tokens),
                )
            )
        return extra

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

    @staticmethod
    async def _should_prefer_search_mcp(
        mcp_client: Any,
        *,
        mcp_tools: list[Any],
        mcp_module_map: dict[str, str],
    ) -> bool:
        config = getattr(mcp_client, "config", None)
        servers = getattr(config, "servers", None)
        if not isinstance(servers, dict):
            return False

        for name, server in servers.items():
            if not getattr(server, "enabled", False):
                continue

            server_tools = AgentFactory._collect_server_tools(
                server_name=str(name),
                mcp_tools=mcp_tools,
                mcp_module_map=mcp_module_map,
            )
            if not AgentFactory._has_search_tool(server_tools):
                continue

            normalized_name = str(name).lower()
            if any(kw in normalized_name for kw in _TAVILY_SERVER_KEYWORDS):
                if await AgentFactory._has_valid_tavily_api_key(server):
                    return True
                continue

            return True

        return False

    @staticmethod
    def _collect_server_tools(
        *,
        server_name: str,
        mcp_tools: list[Any],
        mcp_module_map: dict[str, str],
    ) -> list[Any]:
        server_prefix = f"mcp:{server_name}"
        return [
            tool
            for tool in mcp_tools
            if str(mcp_module_map.get(AgentFactory._tool_name(tool), "") or "") == server_prefix
        ]

    @staticmethod
    def _has_search_tool(mcp_tools: list[Any]) -> bool:
        for tool in mcp_tools:
            tool_name = AgentFactory._tool_name(tool).lower()
            if any(keyword in tool_name for keyword in _SEARCH_TOOL_NAME_KEYWORDS):
                return True
            description = str(getattr(tool, "description", "") or "").lower()
            if description and all(keyword in description for keyword in _SEARCH_TOOL_DESCRIPTION_KEYWORDS):
                return True
        return False

    @staticmethod
    async def _has_valid_tavily_api_key(server: Any) -> bool:
        api_key = AgentFactory._resolve_tavily_api_key(server)
        if not api_key:
            return False
        if not api_key.startswith("tvly-"):
            logger.warning("Ignoring Tavily MCP preference because TAVILY_API_KEY does not look valid.")
            return False
        return await AgentFactory._probe_tavily_api_key(api_key)

    @staticmethod
    def _resolve_tavily_api_key(server: Any) -> str:
        env = getattr(server, "env", None)
        if isinstance(env, dict):
            explicit_key = str(env.get(_TAVILY_API_KEY_ENV, "") or "").strip()
            if explicit_key.startswith("${") and explicit_key.endswith("}"):
                env_name = explicit_key[2:-1].strip()
                return str(os.environ.get(env_name, "") or "").strip()
            if explicit_key:
                return explicit_key

        return str(os.environ.get(_TAVILY_API_KEY_ENV, "") or "").strip()

    @staticmethod
    async def _probe_tavily_api_key(api_key: str) -> bool:
        cached = _TAVILY_KEY_VALIDATION_CACHE.get(api_key)
        if cached is not None:
            return cached

        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=_TAVILY_VALIDATE_TIMEOUT_SECONDS, follow_redirects=True) as client:
                response = await client.get(_TAVILY_VALIDATE_URL, headers=headers)
        except httpx.HTTPError as exc:
            logger.warning("Unable to validate Tavily API key; keeping built-in web_search fallback: %s", exc)
            return False

        if response.status_code == 200:
            _TAVILY_KEY_VALIDATION_CACHE[api_key] = True
            return True

        if response.status_code in {401, 403}:
            logger.warning("Tavily API key validation failed with HTTP %s.", response.status_code)
            _TAVILY_KEY_VALIDATION_CACHE[api_key] = False
            return False

        logger.warning(
            "Unable to confirm Tavily API key validity (HTTP %s); keeping built-in web_search fallback.",
            response.status_code,
        )
        return False

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

        return any(_match(p) for p in positive_patterns) and not any(_match(p) for p in negative_patterns)

    @staticmethod
    def _resolve_existing_paths(paths: Path | list[Path] | None) -> list[str]:
        if paths is None:
            return []
        candidates = [paths] if isinstance(paths, Path) else list(paths)
        return [str(path) for path in candidates if path.exists()]

    @staticmethod
    def _should_enable_skills_middleware(
        *,
        config: BaseAgentConfig,
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
            scripts: list[str] = getattr(skill, "get_script_relative_paths", lambda: [])()
            scripts_text = f" (scripts: {', '.join(f'`{p}`' for p in scripts)})" if scripts else ""
            lines.append(
                f"- {getattr(skill, 'display_name', getattr(skill, 'name', 'unknown'))}: "
                f"{getattr(skill, 'description', '')}{scripts_text}"
            )

        if not use_catalog:
            lines.append("When scripts are present under `scripts/`, prefer running those scripts.")
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
