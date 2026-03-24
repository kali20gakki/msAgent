from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent

from msagent.middlewares import (
    ApprovalMiddleware,
    CompressToolOutputMiddleware,
    PendingToolResultMiddleware,
    ReturnDirectMiddleware,
    RetryConfig,
    RetryMiddleware,
    SandboxMiddleware,
    TokenCostMiddleware,
    ToolRetryConfig,
    create_dynamic_prompt_middleware,
)
from msagent.tools.internal.memory import read_memory_file

if TYPE_CHECKING:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

    from msagent.agents import ContextSchemaType, StateSchemaType
    from msagent.sandboxes import SandboxBackend


def create_react_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    prompt: str,
    state_schema: StateSchemaType | None = None,
    context_schema: ContextSchemaType | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    name: str | None = None,
    tool_sandbox_map: dict[str, SandboxBackend | None] | None = None,
    retry_middleware: RetryMiddleware | None = None,
):
    """Create a ReAct agent using LangChain's create_agent.

    Args:
        model: The language model to use
        tools: List of tools available to the agent
        prompt: System prompt for the agent
        state_schema: Optional custom state schema
        context_schema: Optional custom context schema
        checkpointer: Optional checkpoint saver for persistence
        store: Optional store for memory
        name: Optional agent name
        tool_sandbox_map: Optional mapping of tools to sandbox backends
        retry_middleware: Optional retry middleware for LLM/tool calls.
            If not provided, a default RetryMiddleware is used.
            Set to None explicitly to disable retry.

    Returns:
        Compiled state graph agent
    """
    has_read_memory = read_memory_file in tools

    # Use default retry middleware if not provided
    # Set to None explicitly to disable
    if retry_middleware is None:
        retry_middleware = RetryMiddleware()

    # Middleware execution order:
    # - before_* hooks: First to last
    # - after_* hooks: Last to first (reverse)
    # - wrap_* hooks: Nested (first middleware wraps all others)

    # Group 0: Retry - Outermost wrapper for model/tool calls
    # Must be first to wrap all subsequent middleware
    retry_group: list[AgentMiddleware[Any, Any]] = []
    if retry_middleware:
        retry_group.append(retry_middleware)

    # Group 1: Dynamic prompt - Render template with runtime context
    dynamic_prompt: list[AgentMiddleware[Any, Any]] = [
        create_dynamic_prompt_middleware(prompt),
    ]

    # Group 2: afterModel - After each model response
    after_model: list[AgentMiddleware[Any, Any]] = [
        TokenCostMiddleware(),  # Extract token usage for ctx/token display
    ]

    # Group 3: wrapToolCall - Around each tool call
    wrap_tool_call: list[AgentMiddleware[Any, Any]] = [
        ApprovalMiddleware(),  # Check approval before executing tools
    ]
    # Add sandbox AFTER approval
    if tool_sandbox_map:
        wrap_tool_call.append(SandboxMiddleware(tool_sandbox_map))
    if has_read_memory:
        wrap_tool_call.append(
            CompressToolOutputMiddleware(model)  # Compress large tool outputs
        )

    # Group 4: beforeAgent - Before each agent invocation
    before_agent: list[AgentMiddleware[Any, Any]] = [
        PendingToolResultMiddleware(),  # Repair missing tool results after interrupts
    ]

    # Group 5: beforeModel - Before each model call
    before_model: list[AgentMiddleware[Any, Any]] = [
        ReturnDirectMiddleware(),  # Check for return_direct and terminate if needed
    ]

    # Combine all middleware
    # Order matters: retry is outermost (first), then others
    middlewares: list[AgentMiddleware[Any, Any]] = (
        retry_group
        + dynamic_prompt
        + after_model
        + wrap_tool_call
        + before_agent
        + before_model
    )

    return create_agent(
        model=model,
        tools=tools,
        state_schema=state_schema,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        name=name,
        middleware=middlewares,
    )
