"""Core agent logic for msagent."""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator

from rich.console import Console

from .config import AppConfig, config_manager
from .llm import Message, create_llm_client
from .mcp_client import mcp_manager

console = Console()


class Agent:
    """msagent - Core agent implementation."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or config_manager.get_config()
        self.llm_client = None
        self.messages: list[Message] = []
        self._initialized = False
        self._error_message = ""

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def error_message(self) -> str:
        return self._error_message

    async def initialize(self) -> bool:
        """Initialize the agent."""
        try:
            if not self.config.llm.is_configured():
                self._error_message = (
                    "⚠️ LLM not configured. Please set up your API key:\n"
                    "   • Environment: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY\n"
                    "   • Config file: ~/.config/msagent/config.json\n"
                    "   • Use: msagent config --help"
                )
                return False

            self.llm_client = create_llm_client(self.config.llm)

            for mcp_config in self.config.mcp_servers:
                if mcp_config.enabled:
                    await mcp_manager.add_server(mcp_config)

            self._initialized = True
            return True

        except Exception as e:
            self._error_message = f"❌ Failed to initialize agent: {e}"
            return False

    def get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        mcp_servers = mcp_manager.get_connected_servers()
        prompt = (
            "You are msagent, a helpful AI assistant that can use tools to help users.\n\n"
            "When you need to use a tool, call the tool directly.\n"
            "When you receive tool results, incorporate them into your response naturally.\n\n"
            f"Available MCP servers: {', '.join(mcp_servers) if mcp_servers else 'None'}\n\n"
            "Be concise, helpful, and friendly in your responses."
        )
        return prompt

    async def chat(self, user_input: str) -> str:
        """Process a chat message and return the response."""
        if not self._initialized or not self.llm_client:
            return "Error: Agent not initialized. Please check your configuration."

        self.messages.append(Message("user", user_input))
        all_messages = [Message("system", self.get_system_prompt())] + self.messages
        tools = mcp_manager.get_all_tools()

        timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
        try:
            console.print("[dim]⏳ Waiting for LLM response...[/dim]")
            t0 = time.monotonic()
            response = await asyncio.wait_for(
                self.llm_client.chat(all_messages, tools=tools if tools else None),
                timeout=timeout_s,
            )
            dt = time.monotonic() - t0
            console.print(f"[grey50]⚡ LLM response took {dt:.2f}s[/grey50]")

            self.messages.append(Message("assistant", response))
            self._print_usage()
            return response
        except asyncio.TimeoutError:
            return f"❌ Error: LLM response timed out after {timeout_s:.0f}s"
        except Exception as e:
            return f"❌ Error: {e}"

    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Process a chat message and stream the response."""
        if not self._initialized or not self.llm_client:
            yield "Error: Agent not initialized. Please check your configuration."
            return

        self.messages.append(Message("user", user_input))
        all_messages = [Message("system", self.get_system_prompt())] + self.messages
        tools = mcp_manager.get_all_tools()

        timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
        start = time.monotonic()
        full_response = ""

        try:
            stream = self.llm_client.chat_stream(all_messages, tools=tools if tools else None)

            while True:
                elapsed = time.monotonic() - start
                remaining = max(timeout_s - elapsed, 0.001)
                try:
                    chunk = await asyncio.wait_for(stream.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    try:
                        await stream.aclose()
                    except Exception:
                        pass
                    yield f"❌ Error: LLM stream timed out after {timeout_s:.0f}s"
                    return

                if chunk:
                    full_response += chunk
                    yield chunk

            if not full_response:
                fallback = await asyncio.wait_for(
                    self.llm_client.chat(all_messages, tools=tools if tools else None),
                    timeout=timeout_s,
                )
                if fallback:
                    full_response = fallback
                    yield fallback
                else:
                    last_msgs = [m.to_dict() for m in all_messages[-3:]]
                    yield (
                        "❌ Error: LLM returned empty response.\n"
                        f"Last messages: {json.dumps(last_msgs, ensure_ascii=False)}"
                    )
                    return

            dt = time.monotonic() - start
            yield f"\n\n⚡ LLM response took {dt:.2f}s\n"
            self.messages.append(Message("assistant", full_response))
        except Exception as e:
            yield f"❌ Error: {e}"

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()

    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self.messages.copy()

    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        await mcp_manager.disconnect_all()
        self._initialized = False

    def _print_usage(self) -> None:
        usage = getattr(self.llm_client, "last_usage", None)
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        if all(isinstance(v, int) for v in (prompt, completion, total)):
            console.print(
                "[dim]🧮 Tokens used: "
                f"prompt={prompt} completion={completion} total={total}[/dim]"
            )
