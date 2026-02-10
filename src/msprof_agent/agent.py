"""Core agent logic for MSProf Agent."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from rich.console import Console

from .config import AppConfig, config_manager
from .llm import Message, create_llm_client
from .mcp_client import mcp_manager

console = Console()



class Agent:
    """MSProf Agent - Core agent implementation."""
    
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
                    "   • Config file: ~/.config/msprof-agent/config.json\n"
                    "   • Use: msprof config --help"
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
        
        prompt = """You are MSProf Agent, a helpful AI assistant that can use tools to help users.

When you need to use a tool, respond with a tool call in the appropriate format.
When you receive tool results, incorporate them into your response naturally.

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
        
        try:
            if tools:
                # Use tool-enabled chat
                response = await self.llm_client.chat_with_tools(all_messages, tools)
                
                # Check if tool calls are needed
                if "tool_calls" in response and response["tool_calls"]:
                    # Add assistant message with tool calls
                    self.messages.append(Message(
                        "assistant",
                        response.get("content") or f"Using tool: {response['tool_calls'][0]['function']['name']}"
                    ))
                    
                    # Execute tool calls
                    for tool_call in response["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        console.print(f"[dim]🔧 Calling tool: {tool_name}[/dim]")
                        result = await mcp_manager.call_tool(tool_name, arguments)
                        
                        # Add tool result to messages
                        self.messages.append(Message(
                            "tool",
                            f"Tool '{tool_name}' result: {result}"
                        ))
                    
                    # Get final response after tool execution
                    all_messages = [Message("system", self.get_system_prompt())] + self.messages
                    final_response = await self.llm_client.chat(all_messages)
                    self.messages.append(Message("assistant", final_response))
                    return final_response
                else:
                    content = response.get("content", "")
                    self.messages.append(Message("assistant", content))
                    return content
            else:
                # Simple chat without tools
                response = await self.llm_client.chat(all_messages)
                self.messages.append(Message("assistant", response))
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
        
        try:
            if tools:
                # Check if we need to use tools (non-streaming for tool detection)
                response = await self.llm_client.chat_with_tools(all_messages, tools)
                
                # Check if tool calls are needed
                if "tool_calls" in response and response["tool_calls"]:
                    # Add assistant message with tool calls
                    self.messages.append(Message(
                        "assistant",
                        response.get("content") or f"Using tool: {response['tool_calls'][0]['function']['name']}"
                    ))
                    
                    # Execute tool calls
                    for tool_call in response["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        try:
                            arguments = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError:
                            arguments = {}
                        
                        yield f"🔧 Calling tool: {tool_name}...\n\n"
                        result = await mcp_manager.call_tool(tool_name, arguments)
                        
                        # Add tool result to messages
                        self.messages.append(Message(
                            "tool",
                            f"Tool '{tool_name}' result: {result}"
                        ))
                    
                    # Stream final response after tool execution
                    all_messages = [Message("system", self.get_system_prompt())] + self.messages
                    full_response = ""
                    async for chunk in self.llm_client.chat_stream(all_messages):
                        full_response += chunk
                        yield chunk
                    self.messages.append(Message("assistant", full_response))
                else:
                    content = response.get("content", "")
                    self.messages.append(Message("assistant", content))
                    yield content
            else:
                # Stream without tools
                full_response = ""
                async for chunk in self.llm_client.chat_stream(all_messages):
                    full_response += chunk
                    yield chunk
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
