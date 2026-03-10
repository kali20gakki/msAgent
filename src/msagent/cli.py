"""CLI interface for msagent."""

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent
from .application import ChatApplicationService
from .config import (
    MCPConfig,
    config_manager,
    get_default_api_key_env,
)
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
        service = ChatApplicationService(Agent())
        
        # Initialize agent with spinner
        with console.status("[bold green]Initializing agent and loading MCP servers...[/bold green]", spinner="dots"):
            initialized = await service.initialize()
        
        if not initialized:
            console.print(Panel(
                service.get_status().error_message,
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
                    async for chunk in service.chat_stream(message):
                        console.print(chunk, end="")
                    console.print()
                else:
                    response = await service.chat(message)
                    console.print(response)
            else:
                # Interactive mode
                mcp_servers = list(service.get_status().connected_servers)
                if mcp_servers:
                    server_list = ", ".join([f"[cyan]{s}[/cyan]" for s in mcp_servers])
                    mcp_msg = f"\n\n[dim]🔌 Connected MCP Servers: {server_list}[/dim]"
                else:
                    mcp_msg = "\n\n[dim]⚠️ No MCP servers connected[/dim]"

                console.print(Panel(
                    "[bold green]🤖 msAgent[/bold green] - Interactive Mode\n"
                    "Type your message and press Enter. Use [bold]/help[/bold] for commands." + mcp_msg,
                    border_style="green"
                ))
                
                while True:
                    try:
                        user_input = console.input("[bold cyan]👤 You:[/bold cyan] ")
                        intent = service.resolve_user_input(user_input)
                        if intent.type == "ignore":
                            continue

                        normalized = " ".join(user_input.strip().lower().split())
                        if normalized == "/help":
                            console.print(Panel(
                                "[bold]Available Commands:[/bold]\n"
                                "  [cyan]/help[/cyan]  - Show this help message\n"
                                "  [cyan]/clear[/cyan] - Clear chat history\n"
                                "  [cyan]/exit[/cyan]  - Exit the chat\n",
                                title="Help",
                                border_style="blue"
                            ))
                            continue

                        if intent.type == "exit":
                            console.print("[dim]Goodbye! 👋[/dim]")
                            break
                        if intent.type == "clear":
                            service.clear_history()
                            console.print("[dim]Chat history cleared.[/dim]")
                            continue
                        if intent.type == "new_session":
                            session_num = service.start_new_session()
                            console.print(f"[dim]Started new session #{session_num}.[/dim]")
                            continue
                        if intent.type != "chat":
                            continue
                        
                        console.print("[bold green]🤖 msAgent:[/bold green] ", end="")
                        
                        if stream:
                            async for chunk in service.chat_stream(intent.message):
                                console.print(chunk, end="")
                            console.print()
                        else:
                            response = await service.chat(intent.message)
                            console.print(response)
                            
                    except KeyboardInterrupt:
                        console.print("\n[dim]Goodbye! 👋[/dim]")
                        break
                    except EOFError:
                        break
        finally:
            await service.shutdown()
    
    asyncio.run(do_chat())


@app.command(name="config")
def config_command(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    llm_provider: Optional[str] = typer.Option(None, "--llm-provider", help="LLM provider (openai/anthropic/gemini/custom)"),
    llm_api_key: Optional[str] = typer.Option(None, "--llm-api-key", help="LLM API key"),
    llm_api_key_env: Optional[str] = typer.Option(
        None,
        "--llm-api-key-env",
        help="Environment variable name used to resolve API key",
    ),
    llm_max_tokens: Optional[int] = typer.Option(
        None,
        "--llm-max-tokens",
        help="Max output tokens (<=0 means auto by model)",
    ),
    llm_base_url: Optional[str] = typer.Option(None, "--llm-base-url", help="Custom base URL"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", "-m", help="Model name"),
) -> None:
    """⚙️ Configure msAgent settings."""
    if not isinstance(llm_api_key_env, str):
        llm_api_key_env = None
    if not isinstance(llm_max_tokens, int):
        llm_max_tokens = None

    if show:
        # ... (omitted similar logic)
        config = config_manager.get_config()
        
        table = Table(title="⚙️ Current Configuration")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("LLM Provider", config.llm.provider)
        table.add_row("API Key", "✓ Set" if config.llm.is_configured() else "✗ Not set")
        table.add_row("API Key Env", config.llm.api_key_env or get_default_api_key_env(config.llm.provider) or "Not configured")
        table.add_row("Base URL", config.llm.base_url or "Default")
        table.add_row("Model", config.llm.model)
        table.add_row("Temperature", str(config.llm.temperature))
        resolved_max_tokens = config.llm.resolve_max_tokens()
        max_tokens_display = (
            "Auto (provider/model default)"
            if config.llm.is_max_tokens_auto()
            else str(config.llm.max_tokens)
        )
        table.add_row("Max Tokens", max_tokens_display)
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
    security_hints: list[str] = []
    model_changed = False
    
    if llm_provider:
        config.llm.provider = llm_provider
    if llm_api_key_env:
        config.llm.api_key_env = llm_api_key_env.strip()
    if llm_api_key:
        config.llm.api_key = llm_api_key.strip()
        if not config.llm.api_key_env:
            default_api_key_env = get_default_api_key_env(config.llm.provider)
            if default_api_key_env:
                config.llm.api_key_env = default_api_key_env
        if config.llm.api_key_env:
            security_hints.append(
                f"API Key 不会写入配置文件，请在环境变量 {config.llm.api_key_env} 中维护密钥。"
            )
    if llm_base_url:
        config.llm.base_url = llm_base_url
    if llm_model:
        next_model = llm_model.strip()
        if next_model:
            model_changed = next_model != config.llm.model
            config.llm.model = next_model
    if llm_max_tokens is not None:
        config.llm.max_tokens = max(0, llm_max_tokens)
    elif model_changed:
        config.llm.max_tokens = 0
    
    config_manager.save_config(config)
    console.print("[green]✓ Configuration saved successfully![/green]")
    for hint in security_hints:
        console.print(f"[yellow]⚠ {hint}[/yellow]")


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
        service = ChatApplicationService(Agent())
        initialized = await service.initialize()
        
        if not initialized:
            console.print(Panel(
                service.get_status().error_message,
                title="[yellow]⚠️ Configuration Required[/yellow]",
                border_style="yellow"
            ))
            return
        
        try:
            if stream:
                async for chunk in service.chat_stream(question):
                    console.print(chunk, end="")
                console.print()
            else:
                response = await service.chat(question)
                console.print(response)
        finally:
            await service.shutdown()
    
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
  Config file:
    • Local: ./config.json
    • Global (Linux / macOS): ~/.config/msagent/config.json
    • Global (Windows): %USERPROFILE%\.config\msagent\config.json
  (API keys are not stored in this file)
  
  Environment variables:
    • OPENAI_API_KEY / OPENAI_MODEL
    • ANTHROPIC_API_KEY / ANTHROPIC_MODEL
    • GEMINI_API_KEY / GEMINI_MODEL
    • CUSTOM_API_KEY / CUSTOM_BASE_URL / CUSTOM_MODEL

[bold]Quick Start:[/bold]
  1. Set your API key:
       Linux / macOS -> export OPENAI_API_KEY="your-key"
       Windows PowerShell -> $env:OPENAI_API_KEY = "your-key"
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
