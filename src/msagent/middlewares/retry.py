"""Middleware for retrying LLM requests and tool calls with configurable strategies."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from msagent.agents import AgentState
from msagent.agents.context import AgentContext
from msagent.core.logging import get_logger

if TYPE_CHECKING:
    from langchain.tools.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime

logger = get_logger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Exception that should trigger a retry."""

    pass


class NonRetryableError(Exception):
    """Exception that should not trigger a retry."""

    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff (default: 2)
        jitter: Whether to add random jitter to delay (default: True)
        retryable_exceptions: Exception types that should trigger retry
        on_retry: Optional callback function called on each retry
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            TimeoutError,
            ConnectionError,
            asyncio.TimeoutError,
        )
    )
    on_retry: Callable[[int, Exception, float], None] | None = None

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given retry attempt."""
        delay = min(
            self.base_delay * (self.exponential_base**attempt),
            self.max_delay,
        )
        if self.jitter:
            delay *= random.uniform(0.5, 1.5)  # noqa: S311
        return delay


@dataclass
class ToolRetryConfig:
    """Configuration for tool call retry.

    Attributes:
        max_retries: Maximum number of retry attempts for tool calls
        timeout: Timeout for each tool call attempt (seconds)
        retryable_tools: List of tool names that should be retried (empty = all)
        exclude_tools: List of tool names that should never be retried
    """

    max_retries: int = 2
    timeout: float | None = None
    retryable_tools: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)


class RetryMiddleware(AgentMiddleware[AgentState, AgentContext]):
    """Middleware to add retry logic for LLM requests and tool calls.

    Examples:
        Basic usage with default config:
        >>> retry_middleware = RetryMiddleware()

        Custom LLM retry config:
        >>> llm_retry = RetryConfig(
        ...     max_retries=5,
        ...     base_delay=2.0,
        ...     retryable_exceptions=(TimeoutError, ConnectionError)
        ... )
        >>> tool_retry = ToolRetryConfig(max_retries=3, timeout=30.0)
        >>> retry_middleware = RetryMiddleware(
        ...     llm_config=llm_retry,
        ...     tool_config=tool_retry
        ... )

        With custom callback:
        >>> def on_retry(attempt, exception, delay):
        ...     print(f"Retry {attempt} after {delay}s due to {exception}")
        ...
        >>> config = RetryConfig(max_retries=3, on_retry=on_retry)
        >>> retry_middleware = RetryMiddleware(llm_config=config)
    """

    def __init__(
        self,
        llm_config: RetryConfig | None = None,
        tool_config: ToolRetryConfig | None = None,
        enable_llm_retry: bool = True,
        enable_tool_retry: bool = False,
    ):
        """Initialize retry middleware.

        Args:
            llm_config: Configuration for LLM request retries
            tool_config: Configuration for tool call retries
            enable_llm_retry: Whether to enable LLM retry (default: True)
            enable_tool_retry: Whether to enable tool retry (default: False)
        """
        super().__init__()
        self.llm_config = llm_config or RetryConfig()
        self.tool_config = tool_config or ToolRetryConfig()
        self.enable_llm_retry = enable_llm_retry
        self.enable_tool_retry = enable_tool_retry

    def _should_retry_exception(
        self, exc: Exception, config: RetryConfig
    ) -> bool:
        """Check if an exception should trigger a retry."""
        # Non-retryable errors take precedence
        if isinstance(exc, NonRetryableError):
            return False
        if isinstance(exc, RetryableError):
            return True
        return isinstance(exc, config.retryable_exceptions)

    async def _retry_with_backoff(
        self,
        operation: Callable[[], T],
        config: RetryConfig,
        operation_name: str,
    ) -> T:
        """Execute operation with retry logic.

        Args:
            operation: Async callable to execute
            config: Retry configuration
            operation_name: Name of operation for logging

        Returns:
            Result of the operation

        Raises:
            Last exception if all retries exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(config.max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                last_exception = e

                # Check if we should retry
                if attempt >= config.max_retries:
                    logger.warning(
                        f"{operation_name} failed after {attempt + 1} attempts: {e}"
                    )
                    raise

                if not self._should_retry_exception(e, config):
                    logger.debug(
                        f"{operation_name} failed with non-retryable error: {e}"
                    )
                    raise

                # Calculate delay and wait
                delay = config.calculate_delay(attempt)
                logger.info(
                    f"{operation_name} attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )

                # Call callback if provided
                if config.on_retry:
                    try:
                        config.on_retry(attempt + 1, e, delay)
                    except Exception:
                        pass  # Ignore callback errors

                await asyncio.sleep(delay)

        # This should never be reached, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected end of retry loop")

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Wrap LLM model call with retry logic.

        This intercepts the LLM invocation and applies retry logic for
        transient failures like timeouts or connection errors.
        
        Args:
            request: ModelRequest containing state and runtime
            handler: The handler function to call
            
        Returns:
            Model response (AIMessage or similar)
        """
        if not self.enable_llm_retry:
            return await handler(request)

        async def _call_model() -> Any:
            result = await handler(request)
            # Ensure we return an AIMessage or ExtendedModelResponse
            if isinstance(result, Command):
                # Extract message from command if needed
                messages = result.update.get("messages", []) if result.update else []
                if messages and isinstance(messages[-1], AIMessage):
                    return messages[-1]
                raise RetryableError("Unexpected command result from model")
            # Allow AIMessage or other valid response types
            return result

        return await self._retry_with_backoff(
            _call_model,
            self.llm_config,
            "LLM request",
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> ToolMessage | Command:
        """Wrap tool call with retry logic and timeout.

        This intercepts tool invocations and applies:
        - Timeout control
        - Retry for transient failures
        """
        if not self.enable_tool_retry:
            return await handler(request)

        tool_name = request.tool_call.get("name", "unknown")

        # Check if tool should be retried
        if self.tool_config.retryable_tools:
            if tool_name not in self.tool_config.retryable_tools:
                return await handler(request)

        if tool_name in self.tool_config.exclude_tools:
            return await handler(request)

        async def _call_tool() -> ToolMessage | Command:
            result = await handler(request)

            # Check if result indicates an error that should be retried
            if isinstance(result, ToolMessage):
                # Access status attribute if available (our custom ToolMessage)
                status = getattr(result, "status", None)
                if status == "error":
                    content = str(result.content)
                    # Retry on certain error patterns
                    if any(
                        pattern in content.lower()
                        for pattern in ["timeout", "connection", "temporarily"]
                    ):
                        raise RetryableError(f"Tool error: {content}")

            return result

        # Apply timeout if configured
        if self.tool_config.timeout:

            async def _call_with_timeout() -> ToolMessage | Command:
                try:
                    return await asyncio.wait_for(
                        _call_tool(),
                        timeout=self.tool_config.timeout,
                    )
                except asyncio.TimeoutError:
                    raise RetryableError(
                        f"Tool {tool_name} timed out after {self.tool_config.timeout}s"
                    )

            operation = _call_with_timeout
        else:
            operation = _call_tool

        # Create tool-specific retry config (fewer retries than LLM)
        tool_retry_config = RetryConfig(
            max_retries=self.tool_config.max_retries,
            base_delay=0.5,  # Shorter delays for tools
            max_delay=10.0,
            retryable_exceptions=(
                TimeoutError,
                asyncio.TimeoutError,
                ConnectionError,
                RetryableError,
            ),
        )

        return await self._retry_with_backoff(
            operation,
            tool_retry_config,
            f"Tool call '{tool_name}'",
        )

    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime[AgentContext],
    ) -> dict[str, Any] | None:
        """Hook called before model invocation.

        Can be used to log or modify state before the call.
        """
        logger.debug("Preparing LLM request with retry middleware enabled")
        return None

    async def aafter_model(
        self,
        state: AgentState,
        runtime: Runtime[AgentContext],
    ) -> dict[str, Any] | None:
        """Hook called after successful model invocation.

        Can be used to log successful completions.
        """
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage):
            logger.debug("LLM request completed successfully")
        return None


class CircuitBreakerRetryMiddleware(RetryMiddleware):
    """Extended retry middleware with circuit breaker pattern.

    Prevents cascading failures by temporarily disabling retries
    after consecutive failures exceed a threshold.

    Examples:
        >>> config = RetryConfig(max_retries=3)
        >>> cb_middleware = CircuitBreakerRetryMiddleware(
        ...     llm_config=config,
        ...     failure_threshold=5,
        ...     recovery_timeout=60.0
        ... )
    """

    def __init__(
        self,
        llm_config: RetryConfig | None = None,
        tool_config: ToolRetryConfig | None = None,
        enable_llm_retry: bool = True,
        enable_tool_retry: bool = False,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        """Initialize circuit breaker retry middleware.

        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Time to wait before trying again (seconds)
        """
        super().__init__(
            llm_config=llm_config,
            tool_config=tool_config,
            enable_llm_retry=enable_llm_retry,
            enable_tool_retry=enable_tool_retry,
        )
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._consecutive_failures = 0
        self._circuit_open = False
        self._last_failure_time: float | None = None

    def _check_circuit(self) -> bool:
        """Check if circuit breaker allows the request.

        Returns:
            True if request should proceed, False if circuit is open
        """
        if not self._circuit_open:
            return True

        # Check if recovery timeout has passed
        if self._last_failure_time:
            import time

            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info("Circuit breaker recovery timeout passed, closing circuit")
                self._circuit_open = False
                self._consecutive_failures = 0
                return True

        return False

    def _record_success(self) -> None:
        """Record a successful request."""
        if self._consecutive_failures > 0:
            self._consecutive_failures = 0
            logger.debug("Reset failure counter after successful request")

    def _record_failure(self) -> None:
        """Record a failed request."""
        import time

        self._consecutive_failures += 1
        self._last_failure_time = time.time()

        if self._consecutive_failures >= self.failure_threshold:
            self._circuit_open = True
            logger.warning(
                f"Circuit breaker opened after {self._consecutive_failures} "
                f"consecutive failures"
            )

    async def awrap_model_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        """Wrap model call with circuit breaker protection."""
        if not self._check_circuit():
            raise NonRetryableError(
                "Circuit breaker is open - too many consecutive failures. "
                f"Retry after {self.recovery_timeout}s"
            )

        try:
            result = await super().awrap_model_call(request, handler)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise


def create_retry_middleware(
    max_llm_retries: int = 3,
    max_tool_retries: int = 2,
    llm_timeout: float | None = None,
    tool_timeout: float | None = None,
    enable_circuit_breaker: bool = False,
) -> RetryMiddleware:
    """Factory function to create a retry middleware with common settings.

    Args:
        max_llm_retries: Maximum retries for LLM requests
        max_tool_retries: Maximum retries for tool calls
        llm_timeout: Timeout for LLM requests (None = no timeout)
        tool_timeout: Timeout for tool calls (None = no timeout)
        enable_circuit_breaker: Whether to enable circuit breaker pattern

    Returns:
        Configured RetryMiddleware instance
    """
    llm_config = RetryConfig(
        max_retries=max_llm_retries,
        base_delay=1.0,
        max_delay=30.0,
    )

    tool_config = ToolRetryConfig(
        max_retries=max_tool_retries,
        timeout=tool_timeout,
    )

    if enable_circuit_breaker:
        return CircuitBreakerRetryMiddleware(
            llm_config=llm_config,
            tool_config=tool_config,
            enable_llm_retry=True,
            enable_tool_retry=max_tool_retries > 0,
        )

    return RetryMiddleware(
        llm_config=llm_config,
        tool_config=tool_config,
        enable_llm_retry=True,
        enable_tool_retry=max_tool_retries > 0,
    )
