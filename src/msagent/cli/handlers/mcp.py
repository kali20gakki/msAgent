"""MCP server management for chat sessions."""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import FormattedTextControl

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console, theme
from msagent.cli.ui.shared import (
    SelectorState,
    create_instruction,
    create_selector_application,
)
from msagent.core.logging import get_logger
from msagent.core.settings import settings

logger = get_logger(__name__)


class MCPHandler:
    """Handles MCP server operations like toggling enabled/disabled."""

    def __init__(self, session) -> None:
        """Initialize with reference to CLI session."""
        self.session = session

    async def handle(self) -> None:
        """Show interactive MCP server selector and toggle enabled/disabled."""
        try:
            # Load current MCP config using existing method
            mcp_config = await initializer.load_mcp_config(
                self.session.context.working_dir
            )

            if not mcp_config.servers:
                console.print_error("No MCP servers configured")
                console.print("")
                return

            # Show interactive selector
            modified = await self._get_mcp_selection(mcp_config.servers)

            if modified:
                # Save changes back to file
                await initializer.save_mcp_config(
                    mcp_config, self.session.context.working_dir
                )
                # Trigger reload
                self.session.needs_reload = True
                self.session.running = False

        except Exception as e:
            console.print_error(f"Error managing MCP servers: {e}")
            console.print("")
            logger.debug("MCP management error", exc_info=True)

    async def _get_mcp_selection(self, mcp_servers) -> bool:
        """Get MCP server selection and toggle enabled/disabled.

        Args:
            mcp_servers: Dictionary of MCPServerConfig objects

        Returns:
            True if any changes were made, False otherwise
        """
        if not mcp_servers:
            return False

        server_names = list(mcp_servers.keys())
        state = SelectorState()
        modified = False

        # Create text control with formatted text
        text_control = FormattedTextControl(
            text=lambda: self._format_server_list(
                mcp_servers, server_names, state.index
            ),
            focusable=True,
            show_cursor=False,
        )

        # Create key bindings
        kb = KeyBindings()

        @kb.add(Keys.Up)
        def _(event):
            state.move_cyclic(-1, size=len(server_names))

        @kb.add(Keys.Down)
        def _(event):
            state.move_cyclic(1, size=len(server_names))

        @kb.add(" ")
        def _(event):
            nonlocal modified
            server_name = server_names[state.index]
            server_config = mcp_servers[server_name]
            # Toggle enabled state
            server_config.enabled = not server_config.enabled
            modified = True

        @kb.add(Keys.Enter)
        def _(event):
            event.app.exit()

        @kb.add(Keys.ControlC)
        def _(event):
            event.app.exit()

        # Create application
        context = self.session.context
        app = create_selector_application(
            context=context,
            text_control=text_control,
            key_bindings=kb,
            header_windows=create_instruction("Space: toggle, Enter: save"),
        )

        try:
            await app.run_async()

            return modified

        except (KeyboardInterrupt, EOFError):
            return False

    @staticmethod
    def _format_server_list(mcp_servers, server_names: list, selected_index: int):
        """Format the MCP server list with checkboxes and highlighting.

        Args:
            mcp_servers: Dictionary of MCPServerConfig objects
            server_names: List of server names
            selected_index: Index of currently selected server

        Returns:
            FormattedText with styled lines
        """
        prompt_symbol = settings.cli.prompt_style.strip()
        lines = []
        for i, name in enumerate(server_names):
            server = mcp_servers[name]
            enabled = server.enabled
            checkbox = "[x]" if enabled else "[ ]"

            if i == selected_index:
                # Use direct color code for selected line
                lines.append(
                    (f"{theme.selection_color}", f"{prompt_symbol} {checkbox} {name}")
                )
            else:
                lines.append(("", f"  {checkbox} {name}"))

            if i < len(server_names) - 1:
                lines.append(("", "\n"))

        return FormattedText(lines)
