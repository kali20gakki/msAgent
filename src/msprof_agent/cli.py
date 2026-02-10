"""CLI interface for MSProf Agent."""

import asyncio
import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .config import LLMConfig, MCPConfig, config_manager
from .tui import run_tui
from .version import __version__

app = typer.Typer(
    name="msagent",
    help="🚀 msAgent - AI Assistant with MCP Support",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Show version information."""
    if value:
        console.print(f"[bold cyan]🚀 msAgent[/bold cyan] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
) -> None:
    """msAgent - AI Assistant with MCP Support."""
    pass


@app.command(name="chat")
def chat_command(
    message: Optional[str] = typer.Argument(None, help="Message to send"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output"),
    tui: bool = typer.Option(False, "--tui", "-t", help="Launch TUI interface"),
) -> None:
    """💬 Start a chat session with msAgent."""
    if tui:
        run_tui()
        return
    
    async def do_chat():
        agent = Agent()
        initialized = await agent.initialize()
        
        if not initialized:
            console.print(Panel(
                agent.error_message,
                title="[yellow]⚠️ Configuration Required[/yellow]",
                border_style="yellow"
            ))
            return
        
        try:
            if message:
                # Single message mode
                console.print(f"[bold cyan]👤 You:[/bold cyan] {message}\n")
                console.print("[bold green]🤖 msAgent:[/bold green] ", end="")
                
                if stream:
                    async for chunk in agent.chat_stream(message):
                        console.print(chunk, end="")
                    console.print()
                else:
                    response = await agent.chat(message)
                    console.print(response)
            else:
                # Interactive mode
                console.print(Panel(
                    "[bold green]🤖 msAgent[/bold green] - Interactive Mode\n"
                    "Type your message and press Enter. Use [bold]/help[/bold] for commands.",
                    border_style="green"
                ))
                
                while True:
                    try:
                        user_input = console.input("[bold cyan]👤 You:[/bold cyan] ")
                        user_input = user_input.strip()
                        
                        if not user_input:
                            continue
                        
                        if user_input.lower() in ["/exit", "/quit", "/q"]:
                            console.print("[dim]Goodbye! 👋[/dim]")
                            break
                        
                        if user_input.lower() == "/help":
                            console.print(Panel(
                                "[bold]Available Commands:[/bold]\n"
                                "  [cyan]/help[/cyan]  - Show this help message\n"
                                "  [cyan]/clear[/cyan] - Clear chat history\n"
                                "  [cyan]/exit[/cyan]  - Exit the chat\n",
                                title="Help",
                                border_style="blue"
                            ))
                            continue
                        
                        if user_input.lower() == "/clear":
                            agent.clear_history()
                            console.print("[dim]Chat history cleared.[/dim]")
                            continue
                        
                        console.print("[bold green]🤖 msAgent:[/bold green] ", end="")
                        
                        if stream:
                            async for chunk in agent.chat_stream(user_input):
                                console.print(chunk, end="")
                            console.print()
                        else:
                            response = await agent.chat(user_input)
                            console.print(response)
                            
                    except KeyboardInterrupt:
                        console.print("\n[dim]Goodbye! 👋[/dim]")
                        break
                    except EOFError:
                        break
        finally:
            await agent.shutdown()
    
    asyncio.run(do_chat())


@app.command(name="config")
def config_command(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    llm_provider: Optional[str] = typer.Option(None, "--llm-provider", help="LLM provider (openai/anthropic/gemini/custom)"),
    llm_api_key: Optional[str] = typer.Option(None, "--llm-api-key", help="LLM API key"),
    llm_base_url: Optional[str] = typer.Option(None, "--llm-base-url", help="Custom base URL"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", "-m", help="Model name"),
) -> None:
    """⚙️ Configure msAgent settings."""
    if show:
        # ... (omitted similar logic)
        config = config_manager.get_config()
        
        table = Table(title="⚙️ Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("LLM Provider", config.llm.provider)
        table.add_row("API Key", "✓ Set" if config.llm.api_key else "✗ Not set")
        table.add_row("Base URL", config.llm.base_url or "Default")
        table.add_row("Model", config.llm.model)
        table.add_row("Temperature", str(config.llm.temperature))
        table.add_row("Max Tokens", str(config.llm.max_tokens))
        table.add_row("Theme", config.theme)
        table.add_row("MCP Servers", str(len(config.mcp_servers)))
        
        console.print(table)
        
        if config.mcp_servers:
            mcp_table = Table(title="🔌 MCP Servers")
            mcp_table.add_column("Name", style="cyan")
            mcp_table.add_column("Command", style="green")
            mcp_table.add_column("Status", style="yellow")
            
            for server in config.mcp_servers:
                status = "✓ Enabled" if server.enabled else "✗ Disabled"
                mcp_table.add_row(server.name, f"{server.command} {' '.join(server.args)}", status)
            
            console.print(mcp_table)
        
        console.print(f"\n[dim]Config file: {config_manager.CONFIG_FILE}[/dim]")
        return
    
    # Update configuration
    config = config_manager.get_config()
    
    if llm_provider:
        config.llm.provider = llm_provider
    if llm_api_key:
        config.llm.api_key = llm_api_key
    if llm_base_url:
        config.llm.base_url = llm_base_url
    if llm_model:
        config.llm.model = llm_model
    
    config_manager.save_config(config)
    console.print("[green]✓ Configuration saved successfully![/green]")


@app.command(name="mcp")
def mcp_command(
    action: str = typer.Argument(..., help="Action: add, remove, list"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Server name"),
    command: Optional[str] = typer.Option(None, "--command", "-c", help="Server command"),
    args: Optional[str] = typer.Option(None, "--args", "-a", help="Command arguments (comma-separated)"),
) -> None:
    """🔌 Manage MCP (Model Context Protocol) servers."""
    if action == "list":
        config = config_manager.get_config()
        
        if not config.mcp_servers:
            console.print("[yellow]⚠ No MCP servers configured.[/yellow]")
            console.print("Use [cyan]msagent mcp add --name <name> --command <cmd>[/cyan] to add one.")
            return
        
        table = Table(title="🔌 MCP Servers")
        table.add_column("Name", style="cyan")
        table.add_column("Command", style="green")
        table.add_column("Arguments", style="blue")
        table.add_column("Status", style="yellow")
        
        for server in config.mcp_servers:
            status = "✓ Enabled" if server.enabled else "✗ Disabled"
            args_str = " ".join(server.args) if server.args else "None"
            table.add_row(server.name, server.command, args_str, status)
        
        console.print(table)
    
    elif action == "add":
        if not name or not command:
            console.print("[red]❌ Error: --name and --command are required[/red]")
            console.print("Usage: msagent mcp add --name <name> --command <cmd> [--args <args>]")
            raise typer.Exit(1)
        
        args_list = args.split(",") if args else []
        
        mcp_config = MCPConfig(
            name=name,
            command=command,
            args=args_list,
        )
        
        config_manager.add_mcp_server(mcp_config)
        console.print(f"[green]✓ MCP server '{name}' added successfully![/green]")
    
    elif action == "remove":
        if not name:
            console.print("[red]❌ Error: --name is required[/red]")
            raise typer.Exit(1)
        
        if config_manager.remove_mcp_server(name):
            console.print(f"[green]✓ MCP server '{name}' removed successfully![/green]")
        else:
            console.print(f"[yellow]⚠ MCP server '{name}' not found.[/yellow]")
    
    else:
        console.print(f"[red]❌ Unknown action: {action}[/red]")
        console.print("Available actions: add, remove, list")
        raise typer.Exit(1)


@app.command(name="ask")
def ask_command(
    question: str = typer.Argument(..., help="Question to ask"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream output"),
) -> None:
    """❓ Ask a single question and get an answer."""
    # ... (logic same as before, no text change needed inside async function except variable names which are internal)
    async def do_ask():
        agent = Agent()
        initialized = await agent.initialize()
        
        if not initialized:
            console.print(Panel(
                agent.error_message,
                title="[yellow]⚠️ Configuration Required[/yellow]",
                border_style="yellow"
            ))
            return
        
        try:
            if stream:
                async for chunk in agent.chat_stream(question):
                    console.print(chunk, end="")
                console.print()
            else:
                response = await agent.chat(question)
                console.print(response)
        finally:
            await agent.shutdown()
    
    asyncio.run(do_ask())


@app.command(name="info")
def info_command() -> None:
    """ℹ️ Show information about msAgent."""
    info_text = """
[bold cyan]🚀 msAgent[/bold cyan] - AI Assistant with MCP Support

[bold]Features:[/bold]
  • 💬 Interactive chat with AI models
  • 🔌 MCP (Model Context Protocol) support
  • 🎨 Beautiful TUI interface
  • 🌊 Streaming responses
  • ⚙️ Flexible configuration

[bold]Supported LLM Providers:[/bold]
  • OpenAI (GPT-4, GPT-3.5)
  • Anthropic (Claude)
  • Google (Gemini)
  • Custom OpenAI-compatible APIs

[bold]Configuration:[/bold]
  Config file: ~/.config/msprof-agent/config.json
  
  Environment variables:
    • OPENAI_API_KEY / OPENAI_MODEL
    • ANTHROPIC_API_KEY / ANTHROPIC_MODEL
    • GEMINI_API_KEY / GEMINI_MODEL
    • CUSTOM_API_KEY / CUSTOM_BASE_URL / CUSTOM_MODEL

[bold]Quick Start:[/bold]
  1. Set your API key: export OPENAI_API_KEY="your-key"
  2. Start chatting: msagent chat
  3. Or use TUI: msagent chat --tui

[bold]Documentation:[/bold]
  Use [cyan]msagent --help[/cyan] for command reference
    """
    
    console.print(Panel(info_text, border_style="cyan"))


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
