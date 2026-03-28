from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from msagent.agents.react_agent import create_react_agent
from msagent.middlewares import (
    CircuitBreakerRetryMiddleware,
    RetryConfig,
    RetryMiddleware,
)
from msagent.tools.subagents.task import SubAgent, create_task_tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore

    from msagent.agents import StateSchemaType
    from msagent.configs import LLMConfig, RetryPolicyConfig
    from msagent.sandboxes import SandboxBackend


def _create_retry_middleware(
    retry_config: RetryPolicyConfig | None,
) -> RetryMiddleware | None:
    """Create retry middleware from agent retry configuration.

    Args:
        retry_config: Retry policy configuration from agent config

    Returns:
        Configured RetryMiddleware or None if retry is disabled
    """
    if retry_config is None:
        # Use default retry config
        return RetryMiddleware()

    if not retry_config.enabled:
        return None

    llm_config = RetryConfig(
        max_retries=retry_config.llm_max_retries,
        base_delay=retry_config.llm_base_delay,
        max_delay=retry_config.llm_max_delay,
    )

    if retry_config.enable_circuit_breaker:
        return CircuitBreakerRetryMiddleware(
            llm_config=llm_config,
            enable_llm_retry=True,
            enable_tool_retry=False,
            failure_threshold=retry_config.circuit_breaker_threshold,
            recovery_timeout=retry_config.circuit_breaker_recovery,
        )

    return RetryMiddleware(
        llm_config=llm_config,
        enable_llm_retry=True,
        enable_tool_retry=False,
    )


def create_deep_agent(
    tools: list[BaseTool],
    prompt: str,
    llm_config: LLMConfig,
    model_provider: Callable[[LLMConfig], BaseChatModel],
    subagents: list[SubAgent] | None = None,
    state_schema: StateSchemaType | None = None,
    context_schema: type[Any] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    internal_tools: list[BaseTool] | None = None,
    store: BaseStore | None = None,
    name: str | None = None,
    tool_sandbox_map: dict[str, SandboxBackend | None] | None = None,
    retry_config: RetryPolicyConfig | None = None,
) -> CompiledStateGraph:
    """Create a deep agent with retry support.

    Args:
        tools: List of tools available to the agent
        prompt: System prompt for the agent
        llm_config: LLM configuration
        model_provider: Function to create LLM model from config
        subagents: Optional list of subagents for delegation
        state_schema: Optional state schema for the graph
        context_schema: Optional context schema for the graph
        checkpointer: Optional checkpoint saver for persistence
        internal_tools: Optional internal tools (not exposed to LLM)
        store: Optional store for memory
        name: Optional agent name
        tool_sandbox_map: Optional mapping of tools to sandbox backends
        retry_config: Optional retry configuration from agent config

    Returns:
        Compiled state graph
    """
    model = model_provider(llm_config)
    all_tools = (internal_tools or []) + tools
    if subagents:
        task_tool = create_task_tool(
            subagents,
            model_provider,
            state_schema,
        )
        all_tools = all_tools + [task_tool]

    # Create retry middleware from config
    retry_middleware = _create_retry_middleware(retry_config)

    return create_react_agent(
        model,
        prompt=prompt,
        tools=all_tools,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        name=name,
        tool_sandbox_map=tool_sandbox_map,
        retry_middleware=retry_middleware,
    )
