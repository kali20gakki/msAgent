"""Middleware - now provided by deepagents."""

# Deepagents provides:
# - FilesystemMiddleware (ls, read_file, write_file, edit_file, glob, grep, execute)
# - SkillsMiddleware (skill loading)
# - MemoryMiddleware (AGENTS.md memory)
# - TodoListMiddleware (todos tool)

# Re-export from deepagents for convenience.
from deepagents.middleware import (
    FilesystemMiddleware,
    MemoryMiddleware,
    SkillsMiddleware,
)
from langchain.agents.middleware import TodoListMiddleware
from msagent.middlewares.tool_result_eviction import ToolResultEvictionMiddleware

__all__ = [
    "FilesystemMiddleware",
    "TodoListMiddleware",
    "SkillsMiddleware",
    "MemoryMiddleware",
    "ToolResultEvictionMiddleware",
]
