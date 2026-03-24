"""Tests for retry middleware."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from msagent.middlewares.retry import (
    CircuitBreakerRetryMiddleware,
    NonRetryableError,
    RetryConfig,
    RetryMiddleware,
    RetryableError,
    ToolRetryConfig,
    create_retry_middleware,
)


def create_mock_ai_message():
    """Create a proper mock AIMessage."""
    return AIMessage(content="test response")


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_values(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True

    def test_calculate_delay_without_jitter(self):
        config = RetryConfig(jitter=False)
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_calculate_delay_with_max_cap(self):
        config = RetryConfig(base_delay=10.0, max_delay=15.0, jitter=False)
        assert config.calculate_delay(0) == 10.0
        assert config.calculate_delay(1) == 15.0  # capped
        assert config.calculate_delay(2) == 15.0  # capped

    def test_calculate_delay_with_jitter(self):
        config = RetryConfig(jitter=True)
        delay = config.calculate_delay(0)
        assert 0.5 <= delay <= 1.5  # base_delay * [0.5, 1.5]


class TestRetryMiddleware:
    """Test RetryMiddleware functionality."""

    @pytest.fixture
    def middleware(self):
        return RetryMiddleware()

    @pytest.fixture
    def mock_state(self):
        return {"messages": []}

    @pytest.fixture
    def mock_runtime(self):
        runtime = MagicMock()
        runtime.context = MagicMock()
        return runtime

    @pytest.mark.asyncio
    async def test_successful_call_no_retry(self, middleware, mock_state, mock_runtime):
        """Test that successful calls don't trigger retry."""
        mock_handler = AsyncMock(return_value=create_mock_ai_message())
        mock_request = MagicMock()
        mock_request.state = mock_state
        mock_request.runtime = mock_runtime

        result = await middleware.awrap_model_call(mock_request, mock_handler)

        assert mock_handler.call_count == 1
        assert result is not None
        assert isinstance(result, AIMessage)

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, middleware, mock_state, mock_runtime):
        """Test that timeout errors trigger retry."""
        mock_handler = AsyncMock()
        mock_handler.side_effect = [
            TimeoutError("First timeout"),
            TimeoutError("Second timeout"),
            create_mock_ai_message(),
        ]
        mock_request = MagicMock()
        mock_request.state = mock_state
        mock_request.runtime = mock_runtime

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await middleware.awrap_model_call(mock_request, mock_handler)

        assert mock_handler.call_count == 3
        assert result is not None
        assert isinstance(result, AIMessage)

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, middleware, mock_state, mock_runtime):
        """Test that max retries raises last exception."""
        mock_handler = AsyncMock(side_effect=TimeoutError("Timeout"))
        mock_request = MagicMock()
        mock_request.state = mock_state
        mock_request.runtime = mock_runtime

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError, match="Timeout"):
                await middleware.awrap_model_call(mock_request, mock_handler)

        assert mock_handler.call_count == 4  # initial + 3 retries

    @pytest.mark.asyncio
    async def test_no_retry_for_non_retryable_errors(self, middleware, mock_state, mock_runtime):
        """Test that non-retryable errors don't trigger retry."""
        mock_handler = AsyncMock(side_effect=ValueError("Invalid value"))
        mock_request = MagicMock()
        mock_request.state = mock_state
        mock_request.runtime = mock_runtime

        with pytest.raises(ValueError, match="Invalid value"):
            await middleware.awrap_model_call(mock_request, mock_handler)

        assert mock_handler.call_count == 1

    @pytest.mark.asyncio
    async def test_non_retryable_error_wrapper(self, middleware, mock_state, mock_runtime):
        """Test NonRetryableError stops retry."""
        mock_handler = AsyncMock(side_effect=NonRetryableError("Stop"))
        mock_request = MagicMock()
        mock_request.state = mock_state
        mock_request.runtime = mock_runtime

        with pytest.raises(NonRetryableError):
            await middleware.awrap_model_call(mock_request, mock_handler)

        assert mock_handler.call_count == 1

    @pytest.mark.asyncio
    async def test_retryable_error_wrapper(self, middleware, mock_state, mock_runtime):
        """Test RetryableError always triggers retry."""
        mock_handler = AsyncMock()
        mock_handler.side_effect = [
            RetryableError("Retry this"),
            create_mock_ai_message(),
        ]
        mock_request = MagicMock()
        mock_request.state = mock_state
        mock_request.runtime = mock_runtime

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await middleware.awrap_model_call(mock_request, mock_handler)

        assert mock_handler.call_count == 2
        assert result is not None
        assert isinstance(result, AIMessage)


class TestToolRetry:
    """Test tool call retry functionality."""

    @pytest.fixture
    def middleware(self):
        return RetryMiddleware(
            enable_tool_retry=True,
            tool_config=ToolRetryConfig(max_retries=2, timeout=5.0),
        )

    @pytest.fixture
    def mock_request(self):
        request = MagicMock()
        request.tool_call = {"name": "test_tool", "id": "123"}
        request.tool = None
        return request

    @pytest.mark.asyncio
    async def test_tool_retry_disabled(self, mock_request):
        """Test that disabled tool retry doesn't retry."""
        middleware = RetryMiddleware(enable_tool_retry=False)
        mock_handler = AsyncMock(side_effect=TimeoutError("Timeout"))

        with pytest.raises(TimeoutError):
            await middleware.awrap_tool_call(mock_request, mock_handler)

        assert mock_handler.call_count == 1

    @pytest.mark.asyncio
    async def test_tool_timeout(self, middleware, mock_request):
        """Test tool call timeout triggers retry."""
        mock_handler = AsyncMock()
        mock_handler.side_effect = [
            asyncio.TimeoutError("Tool timeout"),
            ToolMessage(content="success", tool_call_id="123"),
        ]

        with patch("asyncio.wait_for", side_effect=lambda coro, timeout: coro):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await middleware.awrap_tool_call(mock_request, mock_handler)

        assert result is not None

    @pytest.mark.asyncio
    async def test_excluded_tools_not_retried(self, mock_request):
        """Test that excluded tools are not retried."""
        middleware = RetryMiddleware(
            enable_tool_retry=True,
            tool_config=ToolRetryConfig(
                max_retries=2, exclude_tools=["test_tool"]
            ),
        )
        mock_handler = AsyncMock(side_effect=TimeoutError("Timeout"))

        with pytest.raises(TimeoutError):
            await middleware.awrap_tool_call(mock_request, mock_handler)

        assert mock_handler.call_count == 1


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def middleware(self):
        return CircuitBreakerRetryMiddleware(
            llm_config=RetryConfig(max_retries=1, jitter=False),
            failure_threshold=3,
            recovery_timeout=1.0,
        )

    @pytest.fixture
    def mock_state(self):
        return {"messages": []}

    @pytest.fixture
    def mock_runtime(self):
        runtime = MagicMock()
        runtime.context = MagicMock()
        return runtime

    @pytest.mark.asyncio
    async def test_circuit_opens_after_threshold(self, middleware, mock_state, mock_runtime):
        """Test circuit opens after consecutive failures."""
        mock_handler = AsyncMock(side_effect=TimeoutError("Timeout"))

        # Trigger failures to open circuit
        for _ in range(3):
            with patch("asyncio.sleep"):
                mock_request = MagicMock()
                mock_request.state = mock_state
                mock_request.runtime = mock_runtime
                try:
                    await middleware.awrap_model_call(mock_request, mock_handler)
                except TimeoutError:
                    pass

        # Circuit should be open now
        assert middleware._circuit_open is True

        # Next call should fail immediately
        mock_handler.reset_mock()
        mock_request = MagicMock()
        mock_request.state = mock_state
        mock_request.runtime = mock_runtime
        with pytest.raises(NonRetryableError, match="Circuit breaker is open"):
            await middleware.awrap_model_call(mock_request, mock_handler)

        assert mock_handler.call_count == 0  # Handler not called

    @pytest.mark.asyncio
    async def test_circuit_closes_on_success(self, middleware, mock_state, mock_runtime):
        """Test circuit closes after successful call."""
        mock_handler = AsyncMock(return_value=create_mock_ai_message())

        # Set some failures first
        middleware._consecutive_failures = 2

        # Successful call should reset failure count
        with patch("asyncio.sleep"):
            mock_request = MagicMock()
            mock_request.state = mock_state
            mock_request.runtime = mock_runtime
            await middleware.awrap_model_call(mock_request, mock_handler)

        assert middleware._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_recovery(self, middleware, mock_state, mock_runtime):
        """Test circuit attempts recovery after timeout."""
        import time

        # Open the circuit
        middleware._circuit_open = True
        middleware._last_failure_time = time.time() - 2.0  # 2 seconds ago
        middleware.recovery_timeout = 1.0

        mock_handler = AsyncMock(return_value=create_mock_ai_message())

        # Should allow call after recovery timeout
        mock_request = MagicMock()
        mock_request.state = mock_state
        mock_request.runtime = mock_runtime
        result = await middleware.awrap_model_call(mock_request, mock_handler)

        assert result is not None
        assert middleware._circuit_open is False
        assert isinstance(result, AIMessage)


class TestFactoryFunction:
    """Test create_retry_middleware factory function."""

    def test_default_factory(self):
        middleware = create_retry_middleware()
        assert isinstance(middleware, RetryMiddleware)
        assert middleware.enable_llm_retry is True
        assert middleware.llm_config.max_retries == 3

    def test_factory_with_circuit_breaker(self):
        middleware = create_retry_middleware(enable_circuit_breaker=True)
        assert isinstance(middleware, CircuitBreakerRetryMiddleware)

    def test_factory_custom_values(self):
        middleware = create_retry_middleware(
            max_llm_retries=5,
            max_tool_retries=3,
            tool_timeout=60.0,
        )
        assert middleware.llm_config.max_retries == 5
        assert middleware.tool_config.max_retries == 3
        assert middleware.tool_config.timeout == 60.0


class TestRetryCallback:
    """Test retry callback functionality."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_retry(self):
        callback_calls = []

        def on_retry(attempt: int, exception: Exception, delay: float) -> None:
            callback_calls.append((attempt, exception, delay))

        config = RetryConfig(
            max_retries=2,
            jitter=False,
            on_retry=on_retry,
        )
        middleware = RetryMiddleware(llm_config=config)

        mock_state = {"messages": []}
        mock_runtime = MagicMock()
        mock_runtime.context = MagicMock()

        mock_handler = AsyncMock()
        mock_handler.side_effect = [
            TimeoutError("First"),
            TimeoutError("Second"),
            create_mock_ai_message(),
        ]

        with patch("asyncio.sleep"):
            mock_request = MagicMock()
            mock_request.state = mock_state
            mock_request.runtime = mock_runtime
            await middleware.awrap_model_call(mock_request, mock_handler)

        assert len(callback_calls) == 2
        assert callback_calls[0][0] == 1
        assert callback_calls[1][0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
