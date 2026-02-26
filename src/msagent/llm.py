"""LLM client for msagent (deepagents-based)."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from deepagents import create_deep_agent
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from pydantic import create_model

from .config import LLMConfig
from .mcp_client import mcp_manager


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


class DeepAgentsClient:
    """deepagents-powered LLM client."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        skills: list[str] | None = None,
        memory: list[str] | None = None,
    ):
        self.config = config
        self.last_usage: dict[str, Any] | None = None
        self._model = self._build_model(config)
        self._agent_cache: dict[str, Any] = {}
        self._skills = skills or []
        self._memory = memory or []

    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> str:
        system_prompt, input_messages = self._split_messages(messages)
        agent = self._get_agent(system_prompt, tools or [])
        self.last_usage = None
        result = await agent.ainvoke({"messages": input_messages})
        self.last_usage = self._extract_usage_from_result(result)
        content = self._extract_assistant_content(result)
        return content or ""

    async def chat_stream(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> AsyncGenerator[str, None]:
        system_prompt, input_messages = self._split_messages(messages)
        agent = self._get_agent(system_prompt, tools or [])
        self.last_usage = None
        got_chunk = False

        async for event in agent.astream_events(
            {"messages": input_messages},
            version="v2",
        ):
            usage = self._extract_usage_from_event(event)
            if usage:
                self.last_usage = usage
            if event.get("event") != "on_chat_model_stream":
                continue
            data = event.get("data", {})
            chunk = data.get("chunk")
            text = self._extract_chunk_text(chunk)
            if text:
                got_chunk = True
                yield text

        if not got_chunk:
            # Fallback: if streaming path returns no token events, return one-shot output.
            text = await self.chat(messages, tools)
            if text:
                yield text

    async def chat_with_tools(
        self, messages: list[Message], tools: list[dict]
    ) -> dict[str, Any]:
        content = await self.chat(messages, tools=tools)
        return {"role": "assistant", "content": content}

    def _get_agent(self, system_prompt: str, tools: list[dict]) -> Any:
        cache_key = (
            f"{system_prompt}\n----\n"
            f"{json.dumps(tools, ensure_ascii=False, sort_keys=True)}"
        )
        cached = self._agent_cache.get(cache_key)
        if cached is not None:
            return cached

        deep_tools = [self._build_structured_tool(tool) for tool in tools]
        agent = create_deep_agent(
            model=self._model,
            system_prompt=system_prompt,
            tools=deep_tools,
            skills=self._skills or None,
            memory=self._memory or None,
        )
        self._agent_cache[cache_key] = agent
        return agent

    def _build_model(self, config: LLMConfig):
        provider = (config.provider or "").lower()

        if provider in {"openai", "custom"}:
            kwargs: dict[str, Any] = {
                "model": config.model,
                "api_key": config.api_key,
                "temperature": config.temperature,
                "max_completion_tokens": config.max_tokens,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOpenAI(**kwargs)

        if provider == "anthropic":
            kwargs = {
                "model_name": config.model,
                "api_key": config.api_key,
                "temperature": config.temperature,
                "max_tokens_to_sample": config.max_tokens,
            }
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatAnthropic(**kwargs)

        if provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=config.model,
                api_key=config.api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        raise ValueError(f"Unsupported provider: {config.provider}")

    def _split_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        system_prompt = "\n\n".join(system_parts)
        input_messages = [m.to_dict() for m in messages if m.role != "system"]
        return system_prompt, input_messages

    def _build_structured_tool(self, tool_spec: dict[str, Any]) -> StructuredTool:
        func = tool_spec.get("function", {})
        tool_name = func.get("name", "unknown_tool")
        description = func.get("description") or f"MCP tool: {tool_name}"
        parameters = func.get("parameters") or {}
        args_schema = self._json_schema_to_model(tool_name, parameters)

        async def _runner(**kwargs: Any) -> str:
            return await mcp_manager.call_tool(tool_name, kwargs)

        return StructuredTool.from_function(
            coroutine=_runner,
            name=tool_name,
            description=description,
            args_schema=args_schema,
        )

    def _json_schema_to_model(self, tool_name: str, schema: dict[str, Any]):
        properties = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        fields: dict[str, tuple[type[Any], Any]] = {}

        for key, prop in properties.items():
            field_type = self._json_type_to_python(prop)
            default = ... if key in required else None
            fields[key] = (field_type, default)

        model_name = "".join(ch if ch.isalnum() else "_" for ch in tool_name).strip("_") or "Tool"
        return create_model(f"{model_name}Args", **fields)

    def _json_type_to_python(self, schema: dict[str, Any]) -> type[Any]:
        schema_type = schema.get("type")
        if schema_type == "string":
            return str
        if schema_type == "integer":
            return int
        if schema_type == "number":
            return float
        if schema_type == "boolean":
            return bool
        if schema_type == "array":
            return list[Any]
        if schema_type == "object":
            return dict[str, Any]
        return Any

    def _extract_assistant_content(self, result: Any) -> str | None:
        if isinstance(result, str):
            return result
        if not isinstance(result, dict):
            return None

        messages = result.get("messages")
        if not isinstance(messages, list):
            return None

        for msg in reversed(messages):
            role = getattr(msg, "type", None) or getattr(msg, "role", None)
            if role in {"ai", "assistant"}:
                return self._extract_text(getattr(msg, "content", None))
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return self._extract_text(msg.get("content"))
        return None

    def _extract_chunk_text(self, chunk: Any) -> str:
        if chunk is None:
            return ""
        content = getattr(chunk, "content", None)
        if content is None and isinstance(chunk, dict):
            content = chunk.get("content")
        return self._extract_text(content) or ""

    def _extract_text(self, content: Any) -> str | None:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts) if parts else None

        return None

    def _extract_usage_from_result(self, result: Any) -> dict[str, int] | None:
        if not isinstance(result, dict):
            return None
        messages = result.get("messages")
        if not isinstance(messages, list):
            return None

        for msg in reversed(messages):
            usage = self._extract_usage_from_message_like(msg)
            if usage:
                return usage
        return None

    def _extract_usage_from_event(self, event: dict[str, Any]) -> dict[str, int] | None:
        if not isinstance(event, dict):
            return None
        data = event.get("data")
        if not isinstance(data, dict):
            return None
        direct = self._extract_usage_from_obj(data)
        if direct:
            return direct
        for key in ("output", "chunk", "input"):
            usage = self._extract_usage_from_message_like(data.get(key))
            if usage:
                return usage
        return None

    def _extract_usage_from_message_like(self, msg: Any) -> dict[str, int] | None:
        if msg is None:
            return None

        usage_metadata = getattr(msg, "usage_metadata", None)
        if isinstance(msg, dict) and usage_metadata is None:
            usage_metadata = msg.get("usage_metadata")
        usage = self._normalize_usage_dict(usage_metadata)
        if usage:
            return usage

        response_metadata = getattr(msg, "response_metadata", None)
        if isinstance(msg, dict) and response_metadata is None:
            response_metadata = msg.get("response_metadata")
        if isinstance(response_metadata, dict):
            usage = self._normalize_usage_dict(response_metadata.get("token_usage"))
            if usage:
                return usage
        return None

    def _normalize_usage_dict(self, raw: Any) -> dict[str, int] | None:
        if not isinstance(raw, dict):
            return None

        prompt = raw.get("prompt_tokens")
        if not isinstance(prompt, int):
            prompt = raw.get("input_tokens")

        completion = raw.get("completion_tokens")
        if not isinstance(completion, int):
            completion = raw.get("output_tokens")

        total = raw.get("total_tokens")
        if not isinstance(total, int):
            if isinstance(prompt, int) and isinstance(completion, int):
                total = prompt + completion

        if not all(isinstance(v, int) for v in (prompt, completion, total)):
            return None
        return {
            "prompt_tokens": int(prompt),
            "completion_tokens": int(completion),
            "total_tokens": int(total),
        }

    def _extract_usage_from_obj(self, obj: Any) -> dict[str, int] | None:
        if obj is None:
            return None
        if isinstance(obj, dict):
            usage = self._normalize_usage_dict(obj.get("token_usage"))
            if usage:
                return usage
            usage = self._normalize_usage_dict(obj.get("usage_metadata"))
            if usage:
                return usage
            for key in ("response_metadata", "llm_output", "output"):
                usage = self._extract_usage_from_obj(obj.get(key))
                if usage:
                    return usage
            return None

        llm_output = getattr(obj, "llm_output", None)
        usage = self._extract_usage_from_obj(llm_output)
        if usage:
            return usage
        response_metadata = getattr(obj, "response_metadata", None)
        usage = self._extract_usage_from_obj(response_metadata)
        if usage:
            return usage
        usage_metadata = getattr(obj, "usage_metadata", None)
        return self._normalize_usage_dict(usage_metadata)


def create_llm_client(
    config: LLMConfig,
    *,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
) -> DeepAgentsClient:
    """Factory function to create deepagents client."""
    return DeepAgentsClient(config, skills=skills, memory=memory)
