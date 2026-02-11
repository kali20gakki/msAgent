"""LLM client for msagent."""

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from rich.console import Console

from .config import LLMConfig

console = Console()


class Message:
    """Represents a chat message."""
    
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
    
    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
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
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=60.0,
        )
    
    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> str:
        """Send a chat request."""
        payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            return message.get("content", "")
        return ""
    
    async def chat_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        async with self.client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
    
    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> dict[str, Any]:
        """Send a chat request with tool support."""
        payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "tools": tools,
            "tool_choice": "auto",
        }
        
        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        
        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0].get("message", {})
        return {}


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=60.0,
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
        
        response = await self.client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()
        
        content = data.get("content", [])
        if content and len(content) > 0:
            return content[0].get("text", "")
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
        
        async with self.client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    try:
                        import json
                        event = json.loads(data)
                        if event.get("type") == "content_block_delta":
                            delta = event.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text
                    except json.JSONDecodeError:
                        continue
    
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
        
        response = await self.client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()
        
        content = data.get("content", [])
        if content:
            # Check for tool use
            for block in content:
                if block.get("type") == "tool_use":
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            }
                        }]
                    }
            # Regular text response
            if content[0].get("type") == "text":
                return {
                    "role": "assistant",
                    "content": content[0].get("text", ""),
                }
        return {"role": "assistant", "content": ""}


class GeminiClient(LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_key = config.api_key
        self.base_url = config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60.0,
        )
    
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
    
    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert tools to Gemini format."""
        gemini_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                gemini_tools.append({
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
        return [{"function_declarations": gemini_tools}]
    
    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> str:
        """Send a chat request."""
        gemini_messages = self._convert_messages(messages)
        
        payload: dict[str, Any] = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        if tools:
            payload["tools"] = self._convert_tools(tools)
        
        url = f"/models/{self.config.model}:generateContent?key={self.api_key}"
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        candidates = data.get("candidates", [])
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                return parts[0].get("text", "")
        return ""
    
    async def chat_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat response."""
        gemini_messages = self._convert_messages(messages)
        
        payload: dict[str, Any] = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        if tools:
            payload["tools"] = self._convert_tools(tools)
        
        url = f"/models/{self.config.model}:streamGenerateContent?key={self.api_key}"
        async with self.client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        import json
                        chunk = json.loads(line)
                        candidates = chunk.get("candidates", [])
                        if candidates and len(candidates) > 0:
                            content = candidates[0].get("content", {})
                            parts = content.get("parts", [])
                            if parts and len(parts) > 0:
                                text = parts[0].get("text", "")
                                if text:
                                    yield text
                    except json.JSONDecodeError:
                        continue
    
    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> dict[str, Any]:
        """Send a chat request with tool support."""
        gemini_messages = self._convert_messages(messages)
        
        payload = {
            "contents": gemini_messages,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            },
            "tools": self._convert_tools(tools),
        }
        
        url = f"/models/{self.config.model}:generateContent?key={self.api_key}"
        response = await self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        candidates = data.get("candidates", [])
        if candidates and len(candidates) > 0:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if parts and len(parts) > 0:
                part = parts[0]
                # Check for function call
                if "functionCall" in part:
                    func_call = part["functionCall"]
                    return {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": func_call.get("name", ""),
                            "type": "function",
                            "function": {
                                "name": func_call.get("name", ""),
                                "arguments": json.dumps(func_call.get("args", {})),
                            }
                        }]
                    }
                # Regular text response
                return {
                    "role": "assistant",
                    "content": part.get("text", ""),
                }
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
