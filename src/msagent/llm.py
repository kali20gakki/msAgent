"""LLM client for msagent."""

import json
import os
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

import httpx
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
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(
                connect=10.0,
                read=self._request_timeout_s,
                write=30.0,
                pool=30.0,
            ),
        )

    def _debug(self, message: str) -> None:
        if self._debug_enabled:
            console.print(f"[dim]LLM DEBUG:[/dim] {message}")


    async def _read_response_body(self, response: httpx.Response) -> str:
        try:
            await response.aread()
            return response.text
        except Exception as e:
            return f"<unreadable: {e}>"

    def _log_http_error(self, err: httpx.HTTPStatusError, payload: dict[str, Any], body: str) -> None:
        request = err.request
        response = err.response
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
            "status": response.status_code,
            "url": str(request.url),
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
            f"status={response.status_code} url={request.url} "
            f"payload={safe_payload} response={body}"
        )

    def _format_http_error(self, err: httpx.HTTPStatusError, payload: dict[str, Any], body: str) -> str:
        request = err.request
        response = err.response
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
            f"LLM request failed: status={response.status_code} url={request.url} "
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
        response = await self.client.post(
            "/chat/completions",
            json=payload,
            timeout=self._request_timeout_s,
        )
        if response.status_code >= 400:
            body = await self._read_response_body(response)
            err = httpx.HTTPStatusError(
                "Error response",
                request=response.request,
                response=response,
            )
            self._log_http_error(err, payload, body)
            raise RuntimeError(self._format_http_error(err, payload, body)) from err
        data = response.json()
        self.last_usage = data.get("usage")
        
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            console.print("[dim]✅ LLM response received[/dim]")
            content = message.get("content", "")
            if not content:
                content = message.get("reasoning_content", "")
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
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        self._debug(
            f"POST /chat/completions stream model={self.config.model} "
            f"messages={len(payload['messages'])} tools={bool(tools)}"
        )
        console.print(f"[dim]⏳ Waiting for LLM stream (timeout {self._request_timeout_s:.0f}s)...[/dim]")
        async with self.client.stream(
            "POST",
            "/chat/completions",
            json=payload,
            timeout=self._request_timeout_s,
        ) as response:
            if response.status_code >= 400:
                body = await self._read_response_body(response)
                err = httpx.HTTPStatusError(
                    "Error response",
                    request=response.request,
                    response=response,
                )
                self._log_http_error(err, payload, body)
                raise RuntimeError(self._format_http_error(err, payload, body)) from err
            first_chunk = True
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        console.print("[dim]✅ LLM stream finished[/dim]")
                        break
                    try:
                        import json
                        chunk = json.loads(data)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if not content:
                                content = delta.get("reasoning_content", "")
                            if "usage" in chunk and isinstance(chunk["usage"], dict):
                                self.last_usage = chunk["usage"]
                            if content:
                                if first_chunk:
                                    console.print("[dim]✅ LLM stream started[/dim]")
                                    first_chunk = False
                                yield content
                    except json.JSONDecodeError:
                        continue
    
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
        response = await self.client.post(
            "/chat/completions",
            json=payload,
            timeout=self._request_timeout_s,
        )
        if response.status_code >= 400:
            body = await self._read_response_body(response)
            err = httpx.HTTPStatusError(
                "Error response",
                request=response.request,
                response=response,
            )
            self._log_http_error(err, payload, body)
            raise RuntimeError(self._format_http_error(err, payload, body)) from err
        data = response.json()
        self.last_usage = data.get("usage")
        
        if "choices" in data and len(data["choices"]) > 0:
            console.print("[dim]✅ LLM tool decision received[/dim]")
            return data["choices"][0].get("message", {})
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
