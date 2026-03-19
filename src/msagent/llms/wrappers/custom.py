from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable


class ChatCustomHTTP(BaseChatModel):
    """Minimal HTTP-backed chat model for custom endpoints."""

    url: str
    api_key: SecretStr = Field(default=SecretStr(""), exclude=True, repr=False)
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    streaming: bool = False
    disable_streaming: bool = True
    timeout: float = 60.0
    http_client: httpx.Client | None = Field(default=None, exclude=True)
    http_async_client: httpx.AsyncClient | None = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "custom_http"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | dict[str, Any] | bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        tool_names = [
            tool["function"]["name"]
            for tool in formatted_tools
            if isinstance(tool, dict)
            and tool.get("type") == "function"
            and isinstance(tool.get("function"), dict)
            and tool["function"].get("name")
        ]

        normalized_choice = tool_choice
        if isinstance(normalized_choice, str):
            if normalized_choice in tool_names:
                normalized_choice = {
                    "type": "function",
                    "function": {"name": normalized_choice},
                }
            elif normalized_choice == "any":
                normalized_choice = "required"
        elif isinstance(normalized_choice, bool) and normalized_choice:
            normalized_choice = "required"

        if normalized_choice is not None:
            kwargs["tool_choice"] = normalized_choice

        return super().bind(tools=formatted_tools, **kwargs)

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        api_key = self.api_key.get_secret_value().strip()
        if api_key:
            headers["Authorization"] = (
                api_key
                if api_key.lower().startswith("bearer ")
                else f"Bearer {api_key}"
            )
        return headers

    @staticmethod
    def _message_to_dict(message: BaseMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"content": message.content}

        if message.name:
            payload["name"] = message.name

        if isinstance(message, HumanMessage):
            payload["role"] = "user"
        elif isinstance(message, AIMessage):
            payload["role"] = "assistant"
            if message.tool_calls or message.invalid_tool_calls:
                payload["tool_calls"] = message.additional_kwargs.get(
                    "tool_calls",
                    [
                        {
                            "id": tool_call["id"],
                            "type": "function",
                            "function": {
                                "name": tool_call["name"],
                                "arguments": (
                                    tool_call["args"]
                                    if isinstance(tool_call["args"], str)
                                    else json.dumps(tool_call["args"])
                                ),
                            },
                        }
                        for tool_call in message.tool_calls
                    ],
                )
                payload["content"] = payload["content"] or None
            elif "function_call" in message.additional_kwargs:
                payload["function_call"] = message.additional_kwargs["function_call"]
                payload["content"] = payload["content"] or None
        elif isinstance(message, SystemMessage):
            payload["role"] = message.additional_kwargs.get(
                "__openai_role__", "system"
            )
        elif isinstance(message, ToolMessage):
            payload["role"] = "tool"
            payload["tool_call_id"] = message.tool_call_id
            if message.name:
                payload["name"] = message.name
        elif isinstance(message, FunctionMessage):
            payload["role"] = "function"
        elif isinstance(message, ChatMessage):
            payload["role"] = message.role
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

        return payload

    def _normalize_tools(self, tools: Any) -> list[dict[str, Any]]:
        if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
            raise TypeError("tools must be a sequence of tool definitions")
        return [
            tool if isinstance(tool, dict) else convert_to_openai_tool(tool)
            for tool in tools
        ]

    def _build_payload(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_to_dict(message) for message in messages],
        }

        if self.temperature not in (None, ""):
            payload["temperature"] = self.temperature
        if self.max_tokens not in (None, "", 0):
            payload["max_tokens"] = self.max_tokens
        if stop:
            payload["stop"] = stop

        extra_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        extra_kwargs.pop("stream", None)

        if "tools" in extra_kwargs:
            extra_kwargs["tools"] = self._normalize_tools(extra_kwargs["tools"])

        payload.update(extra_kwargs)
        return payload

    @staticmethod
    def _extract_message_payload(data: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(data, str):
            return {"role": "assistant", "content": data}, {}

        if not isinstance(data, dict):
            raise ValueError(f"Unsupported custom LLM response type: {type(data)}")

        usage = data.get("usage") or data.get("token_usage")
        model_name = data.get("model")

        if isinstance(data.get("message"), dict):
            return data["message"], {
                "usage": usage,
                "model_name": model_name,
                "finish_reason": data.get("finish_reason"),
            }

        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                if isinstance(choice.get("message"), dict):
                    return choice["message"], {
                        "usage": usage,
                        "model_name": model_name,
                        "finish_reason": choice.get("finish_reason"),
                    }
                if "text" in choice:
                    return {"role": "assistant", "content": choice.get("text", "")}, {
                        "usage": usage,
                        "model_name": model_name,
                        "finish_reason": choice.get("finish_reason"),
                    }

        if any(key in data for key in ("content", "tool_calls", "function_call")):
            return data, {
                "usage": usage,
                "model_name": model_name,
                "finish_reason": data.get("finish_reason"),
            }

        if isinstance(data.get("data"), dict):
            return ChatCustomHTTP._extract_message_payload(data["data"])

        raise ValueError("Unable to extract assistant message from custom LLM response")

    @staticmethod
    def _response_to_message(data: Any) -> tuple[AIMessage, dict[str, Any]]:
        message_payload, metadata = ChatCustomHTTP._extract_message_payload(data)

        content = message_payload.get("content", "")
        if content is None:
            content = ""

        additional_kwargs: dict[str, Any] = {}
        if "function_call" in message_payload:
            additional_kwargs["function_call"] = message_payload["function_call"]

        tool_calls = []
        invalid_tool_calls = []
        raw_tool_calls = message_payload.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    parsed = parse_tool_call(raw_tool_call, return_id=True)
                    if parsed is not None:
                        tool_calls.append(parsed)
                except Exception as exc:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(exc))
                    )

        if reasoning := message_payload.get("reasoning_content"):
            additional_kwargs["thinking"] = {"text": reasoning}

        response_metadata = {
            key: value
            for key, value in metadata.items()
            if key != "usage" and value is not None
        }
        if metadata.get("usage") is not None:
            response_metadata["usage"] = metadata["usage"]

        return (
            AIMessage(
                content=content,
                id=message_payload.get("id"),
                name=message_payload.get("name"),
                additional_kwargs=additional_kwargs,
                tool_calls=tool_calls,
                invalid_tool_calls=invalid_tool_calls,
                response_metadata=response_metadata,
            ),
            metadata,
        )

    def _request_sync(self, payload: dict[str, Any]) -> Any:
        if self.http_client is not None:
            response = self.http_client.post(
                self.url, json=payload, headers=self._build_headers()
            )
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return response.text

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.url, json=payload, headers=self._build_headers())
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return response.text

    async def _request_async(self, payload: dict[str, Any]) -> Any:
        if self.http_async_client is not None:
            response = await self.http_async_client.post(
                self.url, json=payload, headers=self._build_headers()
            )
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return response.text

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.url, json=payload, headers=self._build_headers()
            )
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return response.text

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, kwargs)
        data = self._request_sync(payload)
        message, metadata = self._response_to_message(data)
        generation = ChatGeneration(
            message=message,
            generation_info={
                key: value
                for key, value in metadata.items()
                if key in {"finish_reason", "model_name", "usage"} and value is not None
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "token_usage": metadata.get("usage", {}),
                "model_name": metadata.get("model_name", self.model),
            },
        )

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = self._build_payload(messages, stop, kwargs)
        data = await self._request_async(payload)
        message, metadata = self._response_to_message(data)
        generation = ChatGeneration(
            message=message,
            generation_info={
                key: value
                for key, value in metadata.items()
                if key in {"finish_reason", "model_name", "usage"} and value is not None
            },
        )
        return ChatResult(
            generations=[generation],
            llm_output={
                "token_usage": metadata.get("usage", {}),
                "model_name": metadata.get("model_name", self.model),
            },
        )
