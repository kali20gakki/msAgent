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

"""Agent factory using deepagents runtime primitives."""

from __future__ import annotations

import logging
import string
import os
import tempfile
from functools import partial
from fnmatch import fnmatch
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware import MemoryMiddleware, SkillsMiddleware
from deepagents.middleware.summarization import SummarizationMiddleware
import httpx
from langchain.agents.middleware import ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware

from msagent.agents.local_context import ensure_local_context_prompt
from msagent.configs import AgentConfig, BaseAgentConfig, CompressionConfig, RetryPolicyConfig, SubAgentConfig
from msagent.core.constants import CONFIG_CONVERSATION_HISTORY_DIR
from msagent.llms.factory import LLMFactory
from msagent.middlewares.model_retry import LoggingModelRetryMiddleware
from msagent.middlewares.tool_result_eviction import ToolResultEvictionMiddleware
from msagent.skills.factory import DEFAULT_SKILL_CATEGORY
from msagent.tools.catalog import (
    fetch_skills,
    fetch_tools,
    get_skill,
    get_tool,
    run_tool,
)
from msagent.tools.factory import ToolFactory
from msagent.tools.internal.memory import is_default_memory_content, read_memory_file
from msagent.tools.web_search import web_search
from msagent.utils.deepagents_compat import patch_deepagents_windows_absolute_paths
from msagent.utils.offload import render_compression_summary_prompt

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
_TAVILY_VALIDATE_MAX_REDIRECTS = 3
_TAVILY_KEY_VALIDATION_CACHE: dict[str, bool] = {}
_SEARCH_TOOL_NAME_KEYWORDS = ("search", "web_search")
_SEARCH_TOOL_DESCRIPTION_KEYWORDS = ("search", "web", "internet", "query")
_THIRD_PARTY_PROMPTS_PATCHED = False
_DEFAULT_MODEL_RETRY_ON_EXCEPTIONS = (
    "httpx.ReadTimeout",
    "asyncio.TimeoutError",
)
_FILTERED_SKILLS_SYSTEM_PROMPT_PREFIX = """

## Skills

Use only the skills listed below. Call `get_skill(name, category)` before using a skill, then follow the SKILL.md workflow.
"""
_COMPACT_BASE_AGENT_PROMPT = """You are a tool-using agent. Be concise, accurate, and action-oriented.

- Read relevant context before editing or concluding.
- Prefer acting directly over narrating intent.
- Verify important results against the user's request.
- Ask only when ambiguity or risk is material."""
_COMPACT_TODO_SYSTEM_PROMPT = """## `write_todos`

Use `write_todos` only for complex multi-step work or when the user explicitly asks for todo tracking.
- Never call `write_todos` more than once in the same turn.
- Mark active work as `in_progress`.
- Mark tasks `completed` immediately after finishing them."""
_COMPACT_FILESYSTEM_SYSTEM_PROMPT = """## Filesystem Tools

Use filesystem tools instead of shell for reading, searching, and editing files.
- Read a file before editing it.
- Use pagination (`offset` / `limit`) for large files.
- All file paths must be absolute and start with `/`."""
_COMPACT_EXECUTION_SYSTEM_PROMPT = """## Execute Tool `execute`

Use `execute` for shell commands, scripts, tests, and builds.
- Prefer dedicated filesystem tools over `cat`, `grep`, or `find`.
- Use absolute paths and quote paths that contain spaces."""
_COMPACT_TASK_SYSTEM_PROMPT = """## `task`

Use `task` to delegate independent multi-step work or parallelizable investigations to subagents.
- Avoid `task` for trivial work.
- Reconcile subagent results in the main thread."""
_COMPACT_MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
Persist only durable preferences, role instructions, and reusable task context.
- Do not store transient details or secrets.
- Ask for missing identifiers before acting.
- If the user explicitly asks to remember durable information, update memory promptly.
</memory_guidelines>
"""


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


class _FilteredSkillsMiddleware(SkillsMiddleware):
    """Load skills metadata, then keep only the skills allowed by the current agent patterns."""

    def __init__(self, *, backend: Any, sources: list[str], allowed_skills: list[Any] | None) -> None:
        super().__init__(backend=backend, sources=sources)
        self._default_allowed_skill_names = {
            str(getattr(skill, "name", "")).strip()
            for skill in (allowed_skills or [])
            if str(getattr(skill, "name", "")).strip()
        }
        self.system_prompt_prefix = _FILTERED_SKILLS_SYSTEM_PROMPT_PREFIX

    def _allowed_skill_names(self, runtime: Any | None = None) -> set[str]:
        context = getattr(runtime, "context", None)
        runtime_skills = getattr(context, "skill_catalog", None)
        if not isinstance(runtime_skills, list) or not runtime_skills:
            return set(self._default_allowed_skill_names)
        return {
            str(getattr(skill, "name", "")).strip()
            for skill in runtime_skills  # pylint: disable=not-an-iterable
            if str(getattr(skill, "name", "")).strip()
        }

    def _filter_skills_metadata(
        self,
        skills_metadata: list[dict[str, Any]],
        runtime: Any | None = None,
    ) -> list[dict[str, Any]]:
        allowed_skill_names = self._allowed_skill_names(runtime)
        if not allowed_skill_names:
            return []
        return [skill for skill in skills_metadata if str(skill.get("name", "")).strip() in allowed_skill_names]

    def _sorted_filtered_skills(
        self,
        skills_metadata: list[dict[str, Any]],
        runtime: Any | None = None,
    ) -> list[dict[str, Any]]:
        return sorted(
            self._filter_skills_metadata(skills_metadata, runtime),
            key=lambda skill: (
                str(skill.get("category", "default")).casefold(),
                str(skill.get("name", "")).casefold(),
            ),
        )

    def _format_skills_list(self, skills_metadata: list[dict[str, Any]]) -> str:
        sorted_skills = self._sorted_filtered_skills(skills_metadata)
        if not sorted_skills:
            return "- No skills enabled for the current agent."

        lines: list[str] = []
        for skill in sorted_skills:
            category = str(skill.get("category", "default"))
            display_name = skill["name"] if category == DEFAULT_SKILL_CATEGORY else f"{category}/{skill['name']}"
            lines.append(f"- `{display_name}`: {skill['description']}")
            lines.append(f"  Read `{skill['path']}` for full instructions.")
        return "\n".join(lines)

    def modify_request(self, request):
        skills_metadata = list(request.state.get("skills_metadata", []))
        skills_list = self._format_skills_list(
            self._sorted_filtered_skills(skills_metadata, getattr(request, "runtime", None))
        )
        skills_section = f"{self.system_prompt_prefix}\n{skills_list}\n"
        system_message = getattr(request, "system_message", None)
        if system_message is None:
            return request
        from deepagents.middleware._utils import append_to_system_message

        return request.override(system_message=append_to_system_message(system_message, skills_section))

    def before_agent(self, state, runtime, config):
        # Refresh on every turn so skills added to the configured directories
        # become available without restarting the current agent session.
        load_state = dict(state)
        load_state.pop("skills_metadata", None)
        update = super().before_agent(load_state, runtime, config)
        if update is None:
            return None
        return {"skills_metadata": self._sorted_filtered_skills(list(update.get("skills_metadata", [])), runtime)}

    async def abefore_agent(self, state, runtime, config):
        load_state = dict(state)
        load_state.pop("skills_metadata", None)
        update = await super().abefore_agent(load_state, runtime, config)
        if update is None:
            return None
        return {"skills_metadata": self._sorted_filtered_skills(list(update.get("skills_metadata", [])), runtime)}


def patch_third_party_prompt_defaults() -> None:
    """Patch verbose third-party prompts with compact project defaults."""
    global _THIRD_PARTY_PROMPTS_PATCHED
    if _THIRD_PARTY_PROMPTS_PATCHED:
        return

    import deepagents.graph as deepagents_graph
    import deepagents.middleware.filesystem as deepagents_filesystem
    import deepagents.middleware.memory as deepagents_memory
    import deepagents.middleware.subagents as deepagents_subagents
    from langchain.agents.middleware.todo import TodoListMiddleware, WRITE_TODOS_TOOL_DESCRIPTION

    filesystem_system_prompt = "\n\n".join(
        (
            _COMPACT_FILESYSTEM_SYSTEM_PROMPT,
            _COMPACT_EXECUTION_SYSTEM_PROMPT,
        )
    )

    for module, attr, prompt in (
        (deepagents_graph, "BASE_AGENT_PROMPT", _COMPACT_BASE_AGENT_PROMPT),
        (deepagents_filesystem, "FILESYSTEM_SYSTEM_PROMPT", _COMPACT_FILESYSTEM_SYSTEM_PROMPT),
        (deepagents_filesystem, "EXECUTION_SYSTEM_PROMPT", _COMPACT_EXECUTION_SYSTEM_PROMPT),
        (deepagents_memory, "MEMORY_SYSTEM_PROMPT", _COMPACT_MEMORY_SYSTEM_PROMPT),
    ):
        setattr(module, attr, prompt)

    deepagents_graph.FilesystemMiddleware = partial(
        deepagents_filesystem.FilesystemMiddleware,
        system_prompt=filesystem_system_prompt,
    )
    deepagents_graph.TodoListMiddleware = partial(
        TodoListMiddleware,
        system_prompt=_COMPACT_TODO_SYSTEM_PROMPT,
        tool_description=WRITE_TODOS_TOOL_DESCRIPTION,
    )
    deepagents_graph.SubAgentMiddleware = partial(
        deepagents_subagents.SubAgentMiddleware,
        system_prompt=_COMPACT_TASK_SYSTEM_PROMPT,
    )

    _THIRD_PARTY_PROMPTS_PATCHED = True


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
        project_state_dir: Path | None = None,
        context_schema: type[Any] | None = None,
        mcp_client: Any | None = None,
        skills_dir: Path | list[Path] | None = None,
        allowed_skills: list[Any] | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        llm_config: LLMConfig | None = None,
        interrupt_on: dict[str, bool | dict[str, Any]] | None = None,
    ) -> CompiledStateGraph:
        patch_third_party_prompt_defaults()
        patch_deepagents_windows_absolute_paths()
        working_dir = (working_dir or Path.cwd()).resolve()
        state_dir = project_state_dir.resolve() if project_state_dir is not None else working_dir
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

        tool_patterns = list(config.tools.patterns or []) if config.tools is not None else []
        positive_patterns, negative_patterns = self._compile_tool_patterns(tool_patterns)

        if mcp_client is not None:
            loaded = await mcp_client.tools()
            mcp_tools = list(loaded or [])
            mcp_module_map = dict(getattr(mcp_client, "module_map", {}) or {})

            if await self._should_prefer_search_mcp(
                mcp_client,
                mcp_tools=mcp_tools,
                mcp_module_map=mcp_module_map,
                positive_patterns=positive_patterns,
            ):
                runtime_tools = [t for t in runtime_tools if self._tool_name(t) != "web_search"]
        catalog_runtime_tools = list(runtime_tools)
        catalog_mcp_tools = list(mcp_tools)

        mcp_servers = self._collect_mcp_servers(mcp_client, mcp_module_map)

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
        memory_sources = self._resolve_memory_sources(state_dir)
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

        tool_output_max_tokens = (
            int(getattr(config.tools, "output_max_tokens", 0))
            if config.tools is not None and getattr(config.tools, "output_max_tokens", None) is not None
            else None
        )
        needs_large_results = tool_output_max_tokens is not None and tool_output_max_tokens > 0
        if not needs_large_results:
            for sub in list(getattr(config, "subagents", None) or []):
                sub_tools = getattr(sub, "tools", None)
                if (
                    sub_tools is not None
                    and getattr(sub_tools, "output_max_tokens", None) is not None
                    and int(sub_tools.output_max_tokens) > 0
                ):
                    needs_large_results = True
                    break
        needs_conversation_history = getattr(config, "compression", None) is not None

        middleware: list[AgentMiddleware[Any, Any, Any]] = []
        agent_backend = self._build_agent_backend(
            working_dir,
            conversation_history_dir=state_dir / "conversation_history",
            enable_large_results=needs_large_results,
            enable_conversation_history=needs_conversation_history,
        )
        deepagents_subagents = self._build_deepagents_subagent_specs(
            config=config,
            agent_backend=agent_backend,
            skills_sources=skills_sources,
            allowed_skills=allowed_skills,
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
                _FilteredSkillsMiddleware(
                    backend=metadata_backend,
                    sources=skills_sources,
                    allowed_skills=allowed_skills,
                )
            )
        if tool_output_max_tokens is not None and tool_output_max_tokens > 0:
            middleware.append(
                ToolResultEvictionMiddleware(
                    backend=agent_backend,
                    tool_token_limit_before_evict=tool_output_max_tokens,
                )
            )
        model_retry_middleware = self._build_model_retry_middleware(retry_cfg)
        if model_retry_middleware is not None:
            middleware.append(model_retry_middleware)

        middleware.append(_SystemMessageMiddleware())

        compression_middleware = None
        if needs_conversation_history:
            compression_middleware = self._build_compression_middleware(
                compression_config=getattr(config, "compression"),
                main_llm_config=resolved_llm,
                agent_backend=agent_backend,
                working_dir=working_dir,
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

        if compression_middleware is not None:
            # Reuse the single custom SummarizationMiddleware for the main agent.
            # deepagents hard-codes its own SummarizationMiddleware in the
            # standard stack, so without patching we would end up with two
            # instances sharing the same name and langchain would reject the
            # graph. The patch also keeps auto and manual compaction aligned on
            # the exact same instance.
            import deepagents.graph as deepagents_graph

            original_create_summarization = deepagents_graph.create_summarization_middleware
            patch_hit = False

            def create_summarization_for_msagent(model_, backend_):
                nonlocal patch_hit
                if model_ is model:
                    patch_hit = True
                    return compression_middleware
                return original_create_summarization(model_, backend_)

            deepagents_graph.create_summarization_middleware = create_summarization_for_msagent
            try:
                graph = create_deep_agent(**kwargs)
            finally:
                deepagents_graph.create_summarization_middleware = original_create_summarization

            if not patch_hit:
                logger.warning(
                    "Custom compression patch was never applied: deepagents did "
                    "not route summarization-middleware creation through the "
                    "patched factory (the SDK version likely changed). The custom "
                    "compression configuration may be silently ignored."
                )
        else:
            graph = create_deep_agent(**kwargs)

        # Keep CLI-compatible metadata caches for /tools and runtime context.
        setattr(graph, "_agent_backend", agent_backend)
        setattr(graph, "_llm_tools", all_tools)
        setattr(graph, "_tools_in_catalog", list(all_tools))
        setattr(graph, "_compression_middleware", compression_middleware)
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

    def _build_model_retry_middleware(
        self,
        retry_cfg: RetryPolicyConfig | None,
    ) -> LoggingModelRetryMiddleware | None:
        """Build a `ModelRetryMiddleware` covering the full model call lifecycle.

        Unlike the SDK-level `max_retries` (which only retries before the first token),
        this middleware wraps the entire async model call — including streaming body
        reads — so mid-stream timeouts (`httpx.ReadTimeout` / `openai.APITimeoutError`)
        are retried with the same exponential-backoff policy.
        """
        model_retry_cfg = getattr(retry_cfg, "model", None) if retry_cfg is not None else None
        if (
            retry_cfg is None
            or not retry_cfg.enabled
            or model_retry_cfg is None
            or not getattr(model_retry_cfg, "enabled", True)
            or int(getattr(model_retry_cfg, "max_retries", 0)) <= 0
        ):
            return None

        retry_on_names = getattr(model_retry_cfg, "retry_on", None)
        if retry_on_names is None:
            retry_on_names = list(_DEFAULT_MODEL_RETRY_ON_EXCEPTIONS)
        retry_on = self._resolve_retry_on_exceptions(list(retry_on_names))

        kwargs: dict[str, Any] = {
            "max_retries": int(getattr(model_retry_cfg, "max_retries")),
            "on_failure": getattr(model_retry_cfg, "on_failure", "error"),
        }
        if retry_on is not None:
            kwargs["retry_on"] = retry_on
        return LoggingModelRetryMiddleware(**kwargs)

    def _build_compression_middleware(
        self,
        *,
        compression_config: CompressionConfig,
        main_llm_config: LLMConfig,
        agent_backend: Any,
        working_dir: Path,
    ) -> SummarizationMiddleware:
        """Build the resident summarization middleware for auto and manual compaction."""
        compression_llm_config = compression_config.llm or main_llm_config
        compression_llm = self.llm_factory.create(compression_llm_config)

        summary_prompt = render_compression_summary_prompt(
            compression_config.prompt,
            working_dir,
        )

        trigger: Any = None
        if compression_config.auto_compress_enabled:
            context_window = getattr(main_llm_config, "context_window", None)
            if context_window and context_window > 0:
                trigger = (
                    "tokens",
                    int(context_window * compression_config.auto_compress_threshold),
                )

        return SummarizationMiddleware(
            model=compression_llm,
            backend=agent_backend,
            trigger=trigger,
            keep=("messages", compression_config.messages_to_keep),
            trim_tokens_to_summarize=None,
            summary_prompt=summary_prompt or "Summarize the conversation so far.",
        )

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
        allowed_skills: list[Any] | None,
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

            extra_mw = self._subagent_extra_middleware(
                sub=sub,
                agent_backend=agent_backend,
                fallback_retry=main_retry,
                skills_sources=skills_sources,
                allowed_skills=self._filter_skills_by_patterns(
                    allowed_skills,
                    patterns=list(getattr(getattr(sub, "skills", None), "patterns", []) or []),
                ),
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
        skills_sources: list[str],
        allowed_skills: list[Any] | None,
    ) -> list[AgentMiddleware[Any, Any, Any]]:
        extra: list[AgentMiddleware[Any, Any, Any]] = []
        retry_cfg = sub.retry if sub.retry is not None else fallback_retry
        model_retry_middleware = self._build_model_retry_middleware(retry_cfg)
        if model_retry_middleware is not None:
            extra.append(model_retry_middleware)
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
        if self._should_enable_skills_middleware(config=sub, skills_sources=skills_sources):
            extra.append(
                _FilteredSkillsMiddleware(
                    backend=FilesystemBackend(virtual_mode=False),
                    sources=skills_sources,
                    allowed_skills=allowed_skills,
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
        positive_patterns: list[tuple[str, str, str]],
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

            # Only prefer an MCP search server when the current agent's tool
            # patterns actually expose its tools. Otherwise dropping the
            # built-in web_search would silently remove generic web search
            # from agents that cannot even call the server's tools.
            if AgentFactory._mcp_server_allowed_by_patterns(positive_patterns, str(name)):
                return True

        return False

    @staticmethod
    def _mcp_server_allowed_by_patterns(
        positive_patterns: list[tuple[str, str, str]],
        server_name: str,
    ) -> bool:
        """Whether the agent's positive tool patterns allow an MCP server."""
        for category_p, module_p, _name_p in positive_patterns:
            if fnmatch("mcp", category_p) and fnmatch(server_name, module_p):
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
            async with httpx.AsyncClient(
                timeout=_TAVILY_VALIDATE_TIMEOUT_SECONDS,
                follow_redirects=True,
                max_redirects=_TAVILY_VALIDATE_MAX_REDIRECTS,
                verify=True,
            ) as client:
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
        sources: list[str] = []
        for path in candidates:
            if not path.is_dir():
                continue

            # DeepAgents scans only ``source/<skill>/SKILL.md``. Built-in
            # skills are grouped as ``skills/<category>/<skill>/SKILL.md``,
            # so expose each category as an individual source.
            child_dirs = sorted(
                (child for child in path.iterdir() if child.is_dir()),
                key=lambda item: item.name,
            )
            if any((child / "SKILL.md").is_file() for child in child_dirs):
                sources.append(str(path))
            sources.extend(
                str(child)
                for child in child_dirs
                if any((skill_dir / "SKILL.md").is_file() for skill_dir in child.iterdir() if skill_dir.is_dir())
            )
        return sources

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
    def _resolve_memory_sources(state_dir: Path) -> list[str]:
        memory_content = read_memory_file(state_dir=state_dir)
        if not memory_content or is_default_memory_content(memory_content):
            return []

        from msagent.tools.internal.memory import ensure_memory_file

        memory_file = ensure_memory_file(state_dir=state_dir)
        return [str(memory_file)]

    @staticmethod
    def _build_agent_backend(
        working_dir: Path,
        *,
        conversation_history_dir: Path | None = None,
        enable_large_results: bool,
        enable_conversation_history: bool,
    ) -> CompositeBackend | LocalShellBackend:
        local_backend = LocalShellBackend(
            root_dir=str(working_dir),
            inherit_env=True,
            virtual_mode=False,
        )
        routes: dict[str, Any] = {}
        if enable_large_results:
            routes["/large_tool_results/"] = FilesystemBackend(
                root_dir=tempfile.mkdtemp(prefix="msagent_large_tool_results_"),
                virtual_mode=True,
            )
        if enable_conversation_history:
            routes["/conversation_history/"] = FilesystemBackend(
                root_dir=conversation_history_dir or working_dir / CONFIG_CONVERSATION_HISTORY_DIR,
                virtual_mode=True,
            )
        if not routes:
            return local_backend
        return CompositeBackend(default=local_backend, routes=routes)

    @staticmethod
    def _filter_skills_by_patterns(skills: list[Any] | None, patterns: list[str]) -> list[Any]:
        if not skills:
            return []
        if not patterns:
            return list(skills)

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

        filtered: list[Any] = []
        for skill in skills:
            category = str(getattr(skill, "category", "default"))
            name = str(getattr(skill, "name", ""))
            if not any(matches(pattern, category=category, name=name) for pattern in positive_patterns):
                continue
            if any(matches(pattern, category=category, name=name) for pattern in negative_patterns):
                continue
            filtered.append(skill)
        return filtered

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
