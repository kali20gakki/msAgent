"""Slash command parser and handlers."""

import shlex
import uuid
from collections.abc import Callable

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.handlers import (
    AgentHandler,
    CompressionHandler,
    MCPHandler,
    ModelHandler,
    SkillsHandler,
    ToolOutputHandler,
    ThreadsHandler,
    ToolsHandler,
)
from msagent.cli.theme import console
from msagent.core.logging import get_logger

logger = get_logger(__name__)


class CommandDispatcher:
    """Dispatch slash commands."""

    def __init__(self, session) -> None:
        """Initialize with reference to CLI session."""
        self.session = session
        self.commands = self._register_commands()
        self.agent_handler = AgentHandler(session)
        self.model_handler = ModelHandler(session)
        self.mcp_handler = MCPHandler(session)
        self.tools_handler = ToolsHandler(session)
        self.skills_handler = SkillsHandler(session)
        self.threads_handler = ThreadsHandler(session)
        self.compression_handler = CompressionHandler(session)
        self.tool_output_handler = ToolOutputHandler(session)

    def _register_commands(self) -> dict[str, Callable]:
        """Register all available commands."""
        return {
            "/help": self.cmd_help,
            "/hotkeys": self.cmd_hotkeys,
            "/agents": self.cmd_agents,
            "/model": self.cmd_model,
            "/threads": self.cmd_threads,
            "/tools": self.cmd_tools,
            "/skills": self.cmd_skills,
            "/mcp": self.cmd_mcp,
            "/offload": self.cmd_offload,
            "/tool-output": self.cmd_tool_output,
            "/clear": self.cmd_clear,
            "/exit": self.cmd_exit,
        }

    async def dispatch(self, command_line: str) -> None:
        """Dispatch a slash command."""
        if not command_line.startswith("/"):
            console.print_error("Commands must start with '/'")
            console.print("")
            return

        try:
            parts = shlex.split(command_line)
            if not parts:
                console.print_error("Empty command")
                console.print("")
                return

            command = parts[0].lower()
            args = parts[1:] if len(parts) > 1 else []

            if command in self.commands:
                await self.commands[command](args)
            else:
                handled = await self.skills_handler.handle_shortcut(
                    initializer.cached_agent_skills,
                    command.removeprefix("/"),
                    args,
                    raw_input=command_line,
                )
                if handled:
                    return
                console.print_error(f"Unknown command: {command}")
                console.print("")
                await self.cmd_help([])

        except Exception as e:
            console.print_error(f"Command error: {e}")
            console.print("")

    async def cmd_help(self, args: list[str]) -> None:
        """Show help information."""
        self.session.renderer.render_help(self.commands)

    async def cmd_hotkeys(self, args: list[str]) -> None:
        """Show keyboard shortcuts."""
        self.session.renderer.render_hotkeys(self.session.prompt.hotkeys)

    async def cmd_agents(self, args: list[str]) -> None:
        """Handle agents command with interactive selector."""
        await self.agent_handler.handle()

    async def cmd_model(self, args: list[str]) -> None:
        """Handle model command with interactive selector."""
        await self.model_handler.handle()

    async def cmd_threads(self, args: list[str]) -> None:
        """Browse and restore previous conversation threads."""
        await self.threads_handler.handle()

    async def cmd_tools(self, args: list[str]) -> None:
        """Handle tools command with interactive selector."""
        await self.tools_handler.handle(initializer.cached_llm_tools)

    async def cmd_skills(self, args: list[str]) -> None:
        """Browse skills or run a specific skill via `/skills <skill> [task...]`."""
        await self.skills_handler.handle(initializer.cached_agent_skills, args)

    async def cmd_mcp(self, args: list[str]) -> None:
        """Handle MCP management command."""
        await self.mcp_handler.handle()

    async def cmd_offload(self, args: list[str]) -> None:
        """Summarize older messages and offload raw history to backend storage."""
        await self.compression_handler.handle()

    async def cmd_tool_output(self, args: list[str]) -> None:
        """Expand or collapse the latest tool output in an interactive viewer."""
        await self.tool_output_handler.handle()

    async def cmd_clear(self, args: list[str]) -> None:
        """Clear the screen and start a new thread."""
        new_thread_id = str(uuid.uuid4())
        self.session.update_context(
            thread_id=new_thread_id,
            current_input_tokens=None,
            current_output_tokens=None,
        )
        self.session.clear_tool_output()
        logger.info(f"Thread ID: {new_thread_id}")
        console.clear()

    async def cmd_exit(self, args: list[str]) -> None:
        """Exit the application."""
        self.session.running = False
