"""CLI bootstrap for starting the LangGraph web server."""

from __future__ import annotations

from pathlib import Path

from msagent.cli.theme import console
from msagent.web.launcher import launch_langgraph_dev_server


async def handle_web_command(args) -> int:
    """Handle the `msagent web` command."""
    try:
        return await launch_langgraph_dev_server(
            host=str(args.host),
            port=int(args.port),
            ui_port=int(args.ui_port),
            start_ui=not bool(args.no_ui),
            open_browser_on_start=not bool(args.no_open),
            working_dir=Path(args.working_dir),
            agent=args.agent,
            model=args.model,
        )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        console.print_error(f"Error starting web server: {exc}")
        console.print("")
        return 1
