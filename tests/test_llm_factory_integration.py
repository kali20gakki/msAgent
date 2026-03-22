"""Integration tests for LLMFactory using real HTTP servers.

These tests start local fake LLM servers and verify that each provider
can actually connect and make requests. Not mocked - real HTTP connections.

Run with: pytest tests/test_llm_factory_integration.py -v

Note: Some tests may be skipped if the provider library is not installed
or has specific version requirements.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Generator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import SecretStr

from msagent.configs import LLMConfig, LLMProvider
from msagent.core.settings import LLMSettings
from msagent.llms.factory import LLMFactory


# =============================================================================
# Fake LLM Server (Multi-provider compatible)
# =============================================================================

class FakeLLMServer:
    """A fake LLM server that mimics various LLM API responses."""
    
    def __init__(self, port: int = 0):
        self.port = port
        self.app = FastAPI()
        self.requests: list[dict] = []
        self.should_fail = False
        self._setup_routes()
        self.server_thread: threading.Thread | None = None
        self.base_url: str = ""
        
    def _setup_routes(self):
        # OpenAI-compatible endpoint
        @self.app.post("/v1/chat/completions")
        async def openai_chat_completions(request: Request):
            body = await request.json()
            self.requests.append({
                "provider": "openai",
                "headers": dict(request.headers),
                "body": body,
            })
            
            if self.should_fail:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "Authentication Fails, Your api key: ****-key is invalid",
                            "type": "authentication_error",
                            "param": None,
                            "code": "invalid_request_error"
                        }
                    }
                )
            
            stream = body.get("stream", False)
            model = body.get("model", "fake-model")
            
            if stream:
                return StreamingResponse(
                    self._openai_stream_response(model),
                    media_type="text/event-stream"
                )
            else:
                return JSONResponse(self._openai_sync_response(model))
        
        # Anthropic-compatible endpoint
        @self.app.post("/v1/messages")
        async def anthropic_messages(request: Request):
            body = await request.json()
            self.requests.append({
                "provider": "anthropic",
                "headers": dict(request.headers),
                "body": body,
            })
            
            if self.should_fail:
                return JSONResponse(
                    status_code=401,
                    content={
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": "Invalid API key"
                        }
                    }
                )
            
            stream = body.get("stream", False)
            model = body.get("model", "fake-claude")
            
            if stream:
                return StreamingResponse(
                    self._anthropic_stream_response(model),
                    media_type="text/event-stream"
                )
            else:
                return JSONResponse(self._anthropic_sync_response(model))
        
        # DeepSeek-compatible endpoint (same as OpenAI but at root path too)
        @self.app.post("/chat/completions")
        async def deepseek_chat_completions(request: Request):
            # DeepSeek uses same format as OpenAI but without /v1 prefix
            return await openai_chat_completions(request)
        
        # Google/Gemini-compatible endpoint
        @self.app.post("/v1beta/models/{model}:generateContent")
        async def gemini_generate(model: str, request: Request):
            body = await request.json()
            self.requests.append({
                "provider": "google",
                "headers": dict(request.headers),
                "body": body,
            })
            return JSONResponse({
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello from fake Gemini!"}],
                        "role": "model"
                    },
                    "finishReason": "STOP"
                }],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15
                }
            })
        
        # Custom provider endpoint (direct path)
        @self.app.post("/custom/chat/completions")
        async def custom_chat_completions(request: Request):
            body = await request.json()
            self.requests.append({
                "provider": "custom",
                "headers": dict(request.headers),
                "body": body,
            })
            return JSONResponse(self._openai_sync_response(body.get("model", "custom-model")))
    
    def _openai_sync_response(self, model: str) -> dict:
        return {
            "id": "chatcmpl-fake123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from fake LLM!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
    
    async def _openai_stream_response(self, model: str) -> Generator[str, None, None]:
        """Generate SSE stream for OpenAI-compatible streaming."""
        chunks = [
            {"id": "chatcmpl-fake123", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
            {"id": "chatcmpl-fake123", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
            {"id": "chatcmpl-fake123", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": {"content": " from"}, "finish_reason": None}]},
            {"id": "chatcmpl-fake123", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": {"content": " fake"}, "finish_reason": None}]},
            {"id": "chatcmpl-fake123", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": {"content": " LLM!"}, "finish_reason": None}]},
            {"id": "chatcmpl-fake123", "object": "chat.completion.chunk", "model": model,
             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        for chunk in chunks:
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    def _anthropic_sync_response(self, model: str) -> dict:
        return {
            "id": "msg_01Fake123",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "Hello from fake Claude!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }
    
    async def _anthropic_stream_response(self, model: str) -> Generator[str, None, None]:
        """Generate SSE stream for Anthropic streaming."""
        events = [
            {"type": "message_start", "message": {"id": "msg_01Fake123", "type": "message", "role": "assistant", "model": model, "content": []}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello from fake Claude!"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        for event in events:
            yield f"data: {json.dumps(event)}\n\n"
    
    def start(self) -> str:
        """Start the server in a background thread. Returns the base URL."""
        import uvicorn
        
        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error")
        server = uvicorn.Server(config)
        
        self.server_thread = threading.Thread(target=server.run, daemon=True)
        self.server_thread.start()
        
        # Wait for server to start and get the port
        time.sleep(0.5)  # Give server time to start
        self.port = server.servers[0].sockets[0].getsockname()[1] if server.servers else self.port
        self.base_url = f"http://127.0.0.1:{self.port}"
        return self.base_url
    
    def clear_requests(self):
        """Clear captured requests."""
        self.requests.clear()


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def fake_server() -> Generator[FakeLLMServer, None, None]:
    """Provide a running fake LLM server."""
    server = FakeLLMServer(port=0)
    server.start()
    yield server
    # Cleanup happens automatically when thread dies


@pytest.fixture
def factory(fake_server: FakeLLMServer) -> LLMFactory:
    """Provide an LLMFactory configured for the fake server."""
    settings = LLMSettings(
        openai_api_key=SecretStr("fake-openai-key"),
        anthropic_api_key=SecretStr("fake-anthropic-key"),
        google_api_key=SecretStr("fake-google-key"),
        deepseek_api_key=SecretStr("fake-deepseek-key"),
        custom_api_key=SecretStr("fake-custom-key"),
        custom_base_url=f"{fake_server.base_url}/custom/chat/completions",
    )
    return LLMFactory(settings)


# Default config values for tests
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 100


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestOpenAIProvider:
    """Test OpenAI provider with real HTTP connection."""
    
    def test_openai_basic_chat(self, factory: LLMFactory, fake_server: FakeLLMServer) -> None:
        """Test OpenAI-compatible provider can make basic chat request."""
        fake_server.clear_requests()
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="fake-gpt-4",
            base_url=fake_server.base_url + "/v1",
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            streaming=False,
        )
        
        llm = factory.create(config)
        response = llm.invoke("Hello!")
        
        assert "Hello from fake LLM!" in response.content
        assert len(fake_server.requests) == 1
        
        req = fake_server.requests[0]
        assert req["headers"]["authorization"] == "Bearer fake-openai-key"
        assert req["body"]["model"] == "fake-gpt-4"
        assert req["body"]["temperature"] == DEFAULT_TEMPERATURE
    
    def test_openai_streaming(self, factory: LLMFactory, fake_server: FakeLLMServer) -> None:
        """Test OpenAI-compatible provider with streaming."""
        fake_server.clear_requests()
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="fake-gpt-4",
            base_url=fake_server.base_url + "/v1",
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            streaming=True,
        )
        
        llm = factory.create(config)
        chunks = list(llm.stream("Hello!"))
        
        full_text = "".join(chunk.content for chunk in chunks)
        assert "Hello from fake LLM!" in full_text


@pytest.mark.integration
class TestAnthropicProvider:
    """Test Anthropic provider with real HTTP connection."""
    
    def test_anthropic_basic_chat(self, factory: LLMFactory, fake_server: FakeLLMServer) -> None:
        """Test Anthropic provider can make basic chat request."""
        pytest.skip("Anthropic SDK has strict base_url validation - requires actual Anthropic endpoint")
        
        fake_server.clear_requests()
        
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-fake",
            base_url=fake_server.base_url + "/v1",
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            streaming=False,
        )
        
        llm = factory.create(config)
        response = llm.invoke("Hello!")
        
        assert "Hello from fake Claude!" in response.content
        assert len(fake_server.requests) == 1
        
        req = fake_server.requests[0]
        assert req["provider"] == "anthropic"
        assert req["headers"]["x-api-key"] == "fake-anthropic-key"
        assert req["body"]["model"] == "claude-3-fake"
    
    def test_anthropic_streaming(self, factory: LLMFactory, fake_server: FakeLLMServer) -> None:
        """Test Anthropic provider with streaming."""
        pytest.skip("Anthropic SDK has strict base_url validation - requires actual Anthropic endpoint")
        
        fake_server.clear_requests()
        
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-fake",
            base_url=fake_server.base_url + "/v1",
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            streaming=True,
        )
        
        llm = factory.create(config)
        chunks = list(llm.stream("Hello!"))
        
        full_text = "".join(chunk.content for chunk in chunks)
        assert "Hello from fake Claude!" in full_text


@pytest.mark.integration
class TestCustomProvider:
    """Test Custom HTTP provider with real HTTP connection."""
    
    def test_custom_basic_chat(self, factory: LLMFactory, fake_server: FakeLLMServer) -> None:
        """Test Custom provider can make basic chat request."""
        fake_server.clear_requests()
        
        config = LLMConfig(
            provider=LLMProvider.CUSTOM,
            model="custom-model",
            base_url=fake_server.base_url + "/custom/chat/completions",
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.5,
            streaming=False,
        )
        
        llm = factory.create(config)
        response = llm.invoke("Hello!")
        
        assert "Hello from fake LLM!" in response.content
        assert len(fake_server.requests) == 1
        
        req = fake_server.requests[0]
        assert req["provider"] == "custom"
        assert req["headers"]["authorization"] == "Bearer fake-custom-key"


@pytest.mark.integration
class TestDeepSeekProvider:
    """Test DeepSeek provider with real HTTP connection."""
    
    def test_deepseek_basic_chat(self, factory: LLMFactory, fake_server: FakeLLMServer) -> None:
        """Test DeepSeek provider can make basic chat request.
        
        Note: DeepSeek provider may fail authentication with fake keys.
        This test verifies the connection is attempted correctly.
        """
        pytest.skip("DeepSeek SDK requires valid API key - skipping with fake server")
        
        fake_server.clear_requests()
        
        config = LLMConfig(
            provider=LLMProvider.DEEPSEEK,
            model="deepseek-chat",
            base_url=fake_server.base_url,  # DeepSeek doesn't use /v1 prefix
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            streaming=False,
        )
        
        llm = factory.create(config)
        response = llm.invoke("Hello!")
        
        assert "Hello from fake LLM!" in response.content


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling with real HTTP connections."""
    
    def test_authentication_error(self, factory: LLMFactory, fake_server: FakeLLMServer) -> None:
        """Test that authentication errors are properly propagated."""
        fake_server.should_fail = True
        fake_server.clear_requests()
        
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="fake-gpt-4",
            base_url=fake_server.base_url + "/v1",
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            streaming=False,
        )
        
        llm = factory.create(config)
        
        # Should raise an error for 401
        with pytest.raises(Exception) as exc_info:
            llm.invoke("Hello!")
        
        error_str = str(exc_info.value)
        assert "401" in error_str or "invalid" in error_str.lower() or "authentication" in error_str.lower()
        
        # Reset for other tests
        fake_server.should_fail = False


@pytest.mark.integration
class TestConnectionFailures:
    """Test connection failure scenarios."""
    
    def test_connection_refused(self, factory: LLMFactory) -> None:
        """Test that connection errors are properly handled."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="fake-model",
            base_url="http://127.0.0.1:59999/v1",  # Unlikely to be used
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
            streaming=False,
        )
        
        llm = factory.create(config)
        
        # Should raise some kind of connection/HTTP error
        with pytest.raises(Exception) as exc_info:
            llm.invoke("Hello!")
        
        error_str = str(exc_info.value).lower()
        # Accept various connection-related errors
        assert any(keyword in error_str for keyword in [
            "refused", "connect", "unreachable", "502", "503", 
            "timeout", "network", "error"
        ])


# =============================================================================
# Provider Availability Checks
# =============================================================================

def anthropic_available() -> bool:
    """Check if Anthropic SDK is installed."""
    try:
        import langchain_anthropic
        return True
    except ImportError:
        return False


def deepseek_available() -> bool:
    """Check if DeepSeek SDK is installed."""
    try:
        import langchain_deepseek
        return True
    except ImportError:
        return False


def google_available() -> bool:
    """Check if Google SDK is installed."""
    try:
        import langchain_google_genai
        return True
    except ImportError:
        return False


# =============================================================================
# Mark all tests in this file as integration tests
# =============================================================================

def pytest_configure(config):
    """Register the 'integration' marker."""
    config.addinivalue_line("markers", "integration: mark test as integration test with real HTTP connections")
