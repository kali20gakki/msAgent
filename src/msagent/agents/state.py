"""Agent state definition."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langchain.agents.middleware.types import AgentState as BaseAgentState
from typing_extensions import NotRequired, TypedDict


class Todo(TypedDict):
    """Todo item for tracking tasks."""

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge two file dictionaries, with right taking precedence."""
    if left is None:
        return right or {}
    elif right is None:
        return left
    else:
        return {**left, **right}


def add_reducer(left: int | None, right: int | None) -> int:
    """Add two integers, treating None as 0."""
    return (left or 0) + (right or 0)


def replace_reducer(left: int | None, right: int | None) -> int:
    """Replace with new value, treating None as 0."""
    return right if right is not None else (left or 0)


class AgentState(BaseAgentState):
    """Extended agent state with additional fields."""
    
    # Token tracking
    input_tokens: Annotated[NotRequired[int], lambda x, y: y if y is not None else x]
    output_tokens: Annotated[NotRequired[int], lambda x, y: y if y is not None else x]
    
    # Interrupt handling
    interrupts: Annotated[NotRequired[list[dict]], lambda x, y: y if y is not None else x]
    
    # File storage (for backward compatibility)
    files: Annotated[dict[str, Any] | None, file_reducer]
    
    # Todo list
    todos: Annotated[list[Todo] | None, lambda x, y: y if y is not None else x]
    
    # Current token tracking
    current_input_tokens: Annotated[int | None, replace_reducer]
    current_output_tokens: Annotated[int | None, add_reducer]
