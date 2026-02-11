"""Core agent logic for msagent."""

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator
from typing import Any

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
            # Check LLM configuration
            if not self.config.llm.is_configured():
                self._error_message = (
                    "⚠️ LLM not configured. Please set up your API key:\n"
                    "   • Environment: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY\n"
                    "   • Config file: ~/.config/msagent/config.json\n"
                    "   • Use: msagent config --help"
                )

                return False
            
            # Initialize LLM client
            self.llm_client = create_llm_client(self.config.llm)
            
            # Initialize MCP servers
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
        
        prompt = """You are msagent, a helpful AI assistant that can use tools to help users.

When you need to use a tool, respond with a tool call in the appropriate format.
When you receive tool results, incorporate them into your response naturally.
Do NOT output DSML or any tool-call markup in the message content. If you need a tool, use the tool_calls field only.

Available MCP servers: """ + (", ".join(mcp_servers) if mcp_servers else "None") + """

Be concise, helpful, and friendly in your responses."""
        
        return prompt
    
    async def chat(self, user_input: str) -> str:
        """Process a chat message and return the response."""
        if not self._initialized or not self.llm_client:
            return "Error: Agent not initialized. Please check your configuration."
        
        # Add user message
        self.messages.append(Message("user", user_input))
        
        # Prepare messages with system prompt
        all_messages = [Message("system", self.get_system_prompt())] + self.messages
        
        # Get available tools
        tools = mcp_manager.get_all_tools()
        
        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        def _add_usage() -> None:
            usage = getattr(self.llm_client, "last_usage", None)
            if isinstance(usage, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    val = usage.get(key)
                    if isinstance(val, int):
                        total_usage[key] += val

        try:
            if tools:
                # Use tool-enabled chat
                console.print("[dim]⏳ Waiting for LLM tool decision...[/dim]")
                timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
                t0 = time.monotonic()
                try:
                    response = await asyncio.wait_for(
                        self.llm_client.chat_with_tools(all_messages, tools),
                        timeout=timeout_s,
                    )
                except asyncio.TimeoutError:
                    return f"❌ Error: LLM tool decision timed out after {timeout_s:.0f}s"
                dt = time.monotonic() - t0
                _add_usage()
                console.print(f"[dim]⏲️ Tool decision took {dt:.2f}s[/dim]")
                
                # Check if tool calls are needed
                tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
                if tool_calls:
                    # Add assistant message with tool calls
                    self.messages.append(Message(
                        "assistant",
                        response.get("content") or "",
                        tool_calls=tool_calls,
                    ))
                    
                    # Execute tool calls
                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        console.print(f"[dim]🔧 Calling tool: {tool_name}[/dim]")
                        t_tool = time.monotonic()
                        result = await mcp_manager.call_tool(tool_name, arguments)
                        t_tool_dt = time.monotonic() - t_tool
                        console.print(f"[dim]✅ Tool finished: {tool_name}[/dim]")
                        console.print(f"[dim]⏲️ Tool {tool_name} took {t_tool_dt:.2f}s[/dim]")
                        
                        # Add tool result to messages (truncate if too large)
                        max_chars = int(os.getenv("MSAGENT_TOOL_RESULT_MAX_CHARS", "12000"))
                        result_text = str(result)
                        if len(result_text) > max_chars:
                            result_text = (
                                result_text[:max_chars]
                                + f"\n\n...[truncated {len(str(result)) - max_chars} chars]"
                            )
                        self.messages.append(Message(
                            "tool",
                            result_text,
                            tool_call_id=tool_call.get("id")
                        ))
                    
                    # Get final response after tool execution
                    all_messages = [Message("system", self.get_system_prompt())] + self.messages
                    console.print("[dim]⏳ Waiting for LLM response...[/dim]")
                    timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
                    force_stream = self._should_force_stream()
                    t1 = time.monotonic()
                    if force_stream:
                        final_response = await self._collect_stream_response(all_messages, timeout_s)
                        if final_response is None:
                            return f"❌ Error: LLM stream timed out after {timeout_s:.0f}s"
                    else:
                        try:
                            final_response = await asyncio.wait_for(
                                self.llm_client.chat(all_messages),
                                timeout=timeout_s,
                            )
                        except asyncio.TimeoutError:
                            return f"❌ Error: LLM response timed out after {timeout_s:.0f}s"
                    dt2 = time.monotonic() - t1
                    _add_usage()
                    console.print(f"[dim]⏲️ LLM response took {dt2:.2f}s[/dim]")
                    self.messages.append(Message("assistant", final_response))
                    if total_usage["total_tokens"] > 0:
                        console.print(
                            f"[dim]🧮 Tokens used: prompt={total_usage['prompt_tokens']} "
                            f"completion={total_usage['completion_tokens']} total={total_usage['total_tokens']}[/dim]"
                        )
                    return final_response
                else:
                    content = response.get("content", "")
                    self.messages.append(Message("assistant", content))
                    if total_usage["total_tokens"] > 0:
                        console.print(
                            f"[dim]🧮 Tokens used: prompt={total_usage['prompt_tokens']} "
                            f"completion={total_usage['completion_tokens']} total={total_usage['total_tokens']}[/dim]"
                        )
                    return content
            else:
                # Simple chat without tools
                console.print("[dim]⏳ Waiting for LLM response...[/dim]")
                timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
                force_stream = self._should_force_stream()
                t0 = time.monotonic()
                if force_stream:
                    response = await self._collect_stream_response(all_messages, timeout_s)
                    if response is None:
                        return f"❌ Error: LLM stream timed out after {timeout_s:.0f}s"
                else:
                    try:
                        response = await asyncio.wait_for(
                            self.llm_client.chat(all_messages),
                            timeout=timeout_s,
                        )
                    except asyncio.TimeoutError:
                        return f"❌ Error: LLM response timed out after {timeout_s:.0f}s"
                dt = time.monotonic() - t0
                _add_usage()
                console.print(f"[dim]⏲️ LLM response took {dt:.2f}s[/dim]")
                self.messages.append(Message("assistant", response))
                if total_usage["total_tokens"] > 0:
                    console.print(
                        f"[dim]🧮 Tokens used: prompt={total_usage['prompt_tokens']} "
                        f"completion={total_usage['completion_tokens']} total={total_usage['total_tokens']}[/dim]"
                    )
                return response
                
        except Exception as e:

            return f"❌ Error: {e}"
    
    async def chat_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Process a chat message and stream the response."""
        if not self._initialized or not self.llm_client:
            yield "Error: Agent not initialized. Please check your configuration."
            return
        
        # Add user message
        self.messages.append(Message("user", user_input))
        
        # Prepare messages with system prompt
        all_messages = [Message("system", self.get_system_prompt())] + self.messages
        
        # Get available tools
        tools = mcp_manager.get_all_tools()
        
        total_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        def _add_usage() -> None:
            usage = getattr(self.llm_client, "last_usage", None)
            if isinstance(usage, dict):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    val = usage.get(key)
                    if isinstance(val, int):
                        total_usage[key] += val

        try:
            if tools:
                # Check if we need to use tools (non-streaming for tool detection)
                console.print("[dim]⏳ Waiting for LLM tool decision...[/dim]")
                timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
                t0 = time.monotonic()
                try:
                    response = await asyncio.wait_for(
                        self.llm_client.chat_with_tools(all_messages, tools),
                        timeout=timeout_s,
                    )
                except asyncio.TimeoutError:
                    yield f"❌ Error: LLM tool decision timed out after {timeout_s:.0f}s"
                    return
                dt = time.monotonic() - t0
                _add_usage()
                yield f"⏲️ Tool decision took {dt:.2f}s\n\n"
                
                # Check if tool calls are needed
                tool_calls = response.get("tool_calls") if isinstance(response, dict) else None
                if tool_calls:
                    # Add assistant message with tool calls
                    self.messages.append(Message(
                        "assistant",
                        response.get("content") or "",
                        tool_calls=tool_calls,
                    ))
                    
                    # Execute tool calls
                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        yield f"🔧 Calling tool: {tool_name}...\n\n"
                        t_tool = time.monotonic()
                        result = await mcp_manager.call_tool(tool_name, arguments)
                        t_tool_dt = time.monotonic() - t_tool
                        yield f"✅ Tool finished: {tool_name}\n\n"
                        yield f"⏲️ Tool {tool_name} took {t_tool_dt:.2f}s\n\n"
                        
                        # Add tool result to messages (truncate if too large)
                        max_chars = int(os.getenv("MSAGENT_TOOL_RESULT_MAX_CHARS", "12000"))
                        result_text = str(result)
                        if len(result_text) > max_chars:
                            result_text = (
                                result_text[:max_chars]
                                + f"\n\n...[truncated {len(str(result)) - max_chars} chars]"
                            )
                        self.messages.append(Message(
                            "tool",
                            result_text,
                            tool_call_id=tool_call.get("id")
                        ))
                    
                    # Stream final response after tool execution
                    all_messages = [Message("system", self.get_system_prompt())] + self.messages
                    console.print("[dim]⏳ Waiting for LLM response...[/dim]")
                    timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
                    t1 = time.monotonic()
                    full_response = ""
                    async for chunk in self._yield_stream_response(all_messages, timeout_s):
                        if chunk is None:
                            yield f"❌ Error: LLM stream timed out after {timeout_s:.0f}s"
                            return
                        full_response += chunk
                        yield chunk
                    if not full_response:
                        yield "❌ Error: LLM returned empty response"
                        return
                    dt2 = time.monotonic() - t1
                    _add_usage()
                    yield f"\n\n⏲️ LLM response took {dt2:.2f}s\n"
                    if total_usage["total_tokens"] > 0:
                        yield (
                            f"🧮 Tokens used: prompt={total_usage['prompt_tokens']} "
                            f"completion={total_usage['completion_tokens']} total={total_usage['total_tokens']}\n"
                        )
                    self.messages.append(Message("assistant", full_response))
                else:
                    content = response.get("content", "")
                    self.messages.append(Message("assistant", content))
                    if total_usage["total_tokens"] > 0:
                        yield (
                            f"\n\n🧮 Tokens used: prompt={total_usage['prompt_tokens']} "
                            f"completion={total_usage['completion_tokens']} total={total_usage['total_tokens']}\n"
                        )
                    yield content
            else:
                # Stream without tools
                full_response = ""
                timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
                t0 = time.monotonic()
                async for chunk in self._yield_stream_response(all_messages, timeout_s):
                    if chunk is None:
                        yield f"❌ Error: LLM stream timed out after {timeout_s:.0f}s"
                        return
                    full_response += chunk
                    yield chunk
                if not full_response:
                    yield "❌ Error: LLM returned empty response"
                    return
                dt = time.monotonic() - t0
                _add_usage()
                yield f"\n\n⏲️ LLM response took {dt:.2f}s\n"
                if total_usage["total_tokens"] > 0:
                    yield (
                        f"🧮 Tokens used: prompt={total_usage['prompt_tokens']} "
                        f"completion={total_usage['completion_tokens']} total={total_usage['total_tokens']}\n"
                    )
                self.messages.append(Message("assistant", full_response))
                
        except Exception as e:

            yield f"❌ Error: {e}"
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


    def _should_force_stream(self) -> bool:
        force_env = os.getenv("MSAGENT_FORCE_STREAM", "1").lower() in {"1", "true", "yes"}
        if force_env:
            return True
        base_url = (self.config.llm.base_url or "").lower()
        model = (self.config.llm.model or "").lower()
        return "deepseek" in base_url or "deepseek-reasoner" in model

    async def _collect_stream_response(self, messages: list[Message], timeout_s: float) -> str | None:
        full_response = ""
        got_chunk = False
        async for chunk in self._yield_stream_response(messages, timeout_s):
            if chunk is None:
                return None
            got_chunk = True
            full_response += chunk
        if not got_chunk:
            return None
        return full_response

    async def _yield_stream_response(self, messages: list[Message], timeout_s: float):
        stream = self.llm_client.chat_stream(messages)
        start = time.monotonic()
        while True:
            if time.monotonic() - start > timeout_s:
                try:
                    await stream.aclose()
                except Exception:
                    pass
                yield None
                return
            try:
                chunk = await asyncio.wait_for(stream.__anext__(), timeout=timeout_s)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                try:
                    await stream.aclose()
                except Exception:
                    pass
                yield None
                return
            yield chunk
    
    def get_history(self) -> list[Message]:
        """Get conversation history."""
        return self.messages.copy()
    
    async def shutdown(self) -> None:
        """Shutdown the agent and cleanup resources."""
        await mcp_manager.disconnect_all()
        self._initialized = False
