"""LLM client for msagent."""

import json
import os
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types as genai_types
from rich.console import Console

from .config import LLMConfig

console = Console()


class Message:
    """Represents a chat message."""
    
    def __init__(
        self,
        role: str,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls
    
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "assistant" and self.tool_calls:
            data["tool_calls"] = self.tool_calls
        if self.role == "tool" and self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        return data


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.last_usage: dict[str, Any] | None = None
    
    @abstractmethod
    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> str:
        """Send a chat request and return the response."""
        pass
    
    @abstractmethod
    async def chat_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncGenerator[str, None]:
        """Send a chat request and stream the response."""
        pass
    
    @abstractmethod
    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> dict[str, Any]:
        """Send a chat request with tool support."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI-compatible API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self._debug_enabled = os.getenv("MSAGENT_DEBUG_LLM", "1").lower() in {"1", "true", "yes"}
        self._request_timeout_s = float(os.getenv("MSAGENT_LLM_TIMEOUT", "120"))
        self._log_path = Path(
            os.getenv("MSAGENT_LLM_LOG_PATH", str(Path.home() / ".config" / "msagent" / "llm_errors.log"))
        )
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=self.base_url,
            timeout=self._request_timeout_s,
        )

    def _debug(self, message: str) -> None:
        if self._debug_enabled:
            console.print(f"[dim]LLM DEBUG:[/dim] {message}")

    def _log_http_error(self, err: Exception, payload: dict[str, Any], body: str) -> None:
        safe_payload = {
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "stream": payload.get("stream", False),
            "tool_choice": payload.get("tool_choice"),
            "tools": bool(payload.get("tools")),
            "messages_count": len(payload.get("messages", [])),
        }
        log_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "status": getattr(err, "status_code", None),
            "url": self.base_url,
            "payload": safe_payload,
            "response": body[:5000],
        }
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as log_err:
            console.print(f"[yellow]LLM log write failed:[/yellow] {log_err}")
        console.print(
            "[red]LLM request failed[/red] "
            f"status={getattr(err, 'status_code', None)} url={self.base_url} "
            f"payload={safe_payload} response={body}"
        )

    def _format_http_error(self, err: Exception, payload: dict[str, Any], body: str) -> str:
        safe_payload = {
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "stream": payload.get("stream", False),
            "tool_choice": payload.get("tool_choice"),
            "tools": bool(payload.get("tools")),
            "messages_count": len(payload.get("messages", [])),
        }
        return (
            f"LLM request failed: status={getattr(err, 'status_code', None)} url={self.base_url} "
            f"payload={safe_payload} response={body}"
        )
    
    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> str:
        """Send a chat request."""
        request_messages = self._prepare_messages(messages)
        payload = {
            "model": self.config.model,
            "messages": request_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        self._debug(
            f"POST /chat/completions model={self.config.model} "
            f"messages={len(payload['messages'])} tools={bool(tools)}"
        )
        console.print(f"[dim]⏳ Waiting for LLM response (timeout {self._request_timeout_s:.0f}s)...[/dim]")
        try:
            response = await self.client.chat.completions.create(**payload)
        except Exception as err:
            body = str(err)
            self._log_http_error(err, payload, body)
            raise RuntimeError(self._format_http_error(err, payload, body)) from err

        self.last_usage = getattr(response, "usage", None)
        if response.choices:
            message = response.choices[0].message
            console.print("[dim]✅ LLM response received[/dim]")
            content = getattr(message, "content", "") or ""
            if not content:
                content = getattr(message, "reasoning_content", "") or ""
            return content
        console.print("[dim]✅ LLM response received (empty)[/dim]")
        return ""
    
    async def chat_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        self.last_usage = None
        request_messages = self._prepare_messages(messages)
        payload = {
            "model": self.config.model,
            "messages": request_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        self._debug(
            f"POST /chat/completions stream model={self.config.model} "
            f"messages={len(payload['messages'])} tools={bool(tools)}"
        )
        console.print(f"[dim]⏳ Waiting for LLM stream (timeout {self._request_timeout_s:.0f}s)...[/dim]")
        try:
            stream = await self.client.chat.completions.create(
                **payload,
                stream=True,
                stream_options={"include_usage": True},
            )
        except Exception as err:
            body = str(err)
            self._log_http_error(err, payload, body)
            raise RuntimeError(self._format_http_error(err, payload, body)) from err

        first_chunk = True
        async for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage:
                self.last_usage = chunk.usage
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "") or ""
            if not content:
                content = getattr(delta, "reasoning_content", "") or ""
            if content:
                if first_chunk:
                    console.print("[dim]✅ LLM stream started[/dim]")
                    first_chunk = False
                yield content
        console.print("[dim]✅ LLM stream finished[/dim]")
    
    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> dict[str, Any]:
        """Send a chat request with tool support."""
        request_messages = self._prepare_messages(messages)
        payload = {
            "model": self.config.model,
            "messages": request_messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "tools": tools,
            "tool_choice": "auto",
        }
        self._debug(
            f"POST /chat/completions tools model={self.config.model} "
            f"messages={len(payload['messages'])} tools={bool(tools)}"
        )
        console.print(f"[dim]⏳ Waiting for LLM tool decision (timeout {self._request_timeout_s:.0f}s)...[/dim]")
        try:
            response = await self.client.chat.completions.create(**payload)
        except Exception as err:
            body = str(err)
            self._log_http_error(err, payload, body)
            raise RuntimeError(self._format_http_error(err, payload, body)) from err

        self.last_usage = getattr(response, "usage", None)
        if response.choices:
            console.print("[dim]✅ LLM tool decision received[/dim]")
            message = response.choices[0].message
            return message.model_dump()
        console.print("[dim]✅ LLM tool decision received (empty)[/dim]")
        return {}

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        request_messages = [m.to_dict() for m in messages]
        if self._requires_reasoning_content():
            for item in request_messages:
                if item.get("role") == "assistant" and "reasoning_content" not in item:
                    item["reasoning_content"] = ""
        return request_messages

    def _requires_reasoning_content(self) -> bool:
        model = (self.config.model or "").lower()
        base_url = (self.base_url or "").lower()
        return "deepseek-reasoner" in model or "api.deepseek.com" in base_url


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com"
        self.client = AsyncAnthropic(
            api_key=config.api_key,
            base_url=self.base_url,
        )
    
    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """Convert messages to Anthropic format."""
        system = ""
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })
        
        return system, anthropic_messages
    
    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> str:
        """Send a chat request."""
        system, anthropic_messages = self._convert_messages(messages)
        
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools

        response = await self.client.messages.create(**payload)
        self.last_usage = getattr(response, "usage", None)
        content = getattr(response, "content", [])
        if content:
            for block in content:
                if getattr(block, "type", None) == "text":
                    return getattr(block, "text", "")
        return ""
    
    async def chat_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        system, anthropic_messages = self._convert_messages(messages)
        
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
        
        async with self.client.messages.stream(**payload) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text
    
    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> dict[str, Any]:
        """Send a chat request with tool support."""
        system, anthropic_messages = self._convert_messages(messages)
        
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "tools": tools,
        }
        if system:
            payload["system"] = system
        
        response = await self.client.messages.create(**payload)
        self.last_usage = getattr(response, "usage", None)
        content = getattr(response, "content", [])
        if content:
            # Check for tool use
            for block in content:
                if getattr(block, "type", None) == "tool_use":
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": getattr(block, "id", ""),
                            "type": "function",
                            "function": {
                                "name": getattr(block, "name", ""),
                                "arguments": json.dumps(getattr(block, "input", {})),
                            }
                        }]
                    }
            # Regular text response
            if getattr(content[0], "type", None) == "text":
                return {
                    "role": "assistant",
                    "content": getattr(content[0], "text", ""),
                }
        return {"role": "assistant", "content": ""}


class GeminiClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com"
        http_options = genai_types.HttpOptions(base_url=self.base_url)
        self.client = genai.Client(api_key=self.api_key, http_options=http_options)
        self.aclient = self.client.aio
    
    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert messages to Gemini format."""
        gemini_messages = []
        for msg in messages:
            if msg.role == "system":
                # Gemini doesn't have system role, treat as user
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": f"System: {msg.content}"}],
                })
            elif msg.role == "user":
                gemini_messages.append({
                    "role": "user",
                    "parts": [{"text": msg.content}],
                })
            elif msg.role == "assistant":
                gemini_messages.append({
                    "role": "model",
                    "parts": [{"text": msg.content}],
                })
        return gemini_messages
    
    def _convert_tools(self, tools: list[dict]) -> list[genai_types.Tool]:
        """Convert tools to Gemini format."""
        function_declarations = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                function_declarations.append(
                    genai_types.FunctionDeclaration(
                        name=func.get("name", ""),
                        description=func.get("description", ""),
                        parameters=func.get("parameters", {}),
                    )
                )
        return [genai_types.Tool(function_declarations=function_declarations)]
    
    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> str:
        """Send a chat request."""
        gemini_messages = self._convert_messages(messages)
        
        config: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }
        request: dict[str, Any] = {
            "model": self.config.model,
            "contents": gemini_messages,
            "config": config,
        }
        if tools:
            request["tools"] = self._convert_tools(tools)
        
        response = await self.aclient.models.generate_content(**request)
        self.last_usage = getattr(response, "usage", None)
        text = getattr(response, "text", "")
        if text:
            return text
        return ""
    
    async def chat_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        gemini_messages = self._convert_messages(messages)
        
        config: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }
        request: dict[str, Any] = {
            "model": self.config.model,
            "contents": gemini_messages,
            "config": config,
        }
        if tools:
            request["tools"] = self._convert_tools(tools)
        
        async for chunk in self.aclient.models.generate_content_stream(**request):
            text = getattr(chunk, "text", "")
            if text:
                yield text
    
    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> dict[str, Any]:
        """Send a chat request with tool support."""
        gemini_messages = self._convert_messages(messages)
        
        config: dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }
        request = {
            "model": self.config.model,
            "contents": gemini_messages,
            "config": config,
            "tools": self._convert_tools(tools),
        }
        
        response = await self.aclient.models.generate_content(**request)
        self.last_usage = getattr(response, "usage", None)
        if hasattr(response, "function_calls") and response.function_calls:
            calls = []
            for idx, call in enumerate(response.function_calls):
                calls.append({
                    "id": f"gemini_call_{idx}",
                    "type": "function",
                    "function": {
                        "name": getattr(call, "name", ""),
                        "arguments": json.dumps(getattr(call, "args", {})),
                    },
                })
            return {"role": "assistant", "content": None, "tool_calls": calls}
        text = getattr(response, "text", "")
        if text:
            return {"role": "assistant", "content": text}
        return {"role": "assistant", "content": ""}


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory function to create appropriate LLM client."""
    if config.provider == "openai":
        return OpenAIClient(config)
    elif config.provider == "anthropic":
        return AnthropicClient(config)
    elif config.provider == "gemini":
        return GeminiClient(config)
    elif config.provider == "custom":
        # Custom uses OpenAI-compatible format
        return OpenAIClient(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")
