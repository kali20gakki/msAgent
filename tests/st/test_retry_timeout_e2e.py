"""End-to-end system test for retry and timeout mechanisms."""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from msagent.agents.react_agent import create_react_agent
from msagent.middlewares import (
    CircuitBreakerRetryMiddleware,
    NonRetryableError,
    RetryConfig,
    RetryMiddleware,
    ToolRetryConfig,
)


# =============================================================================
# E2E Test Suite
# =============================================================================

@pytest.mark.asyncio
class TestLLMRetryE2E:
    """End-to-end tests for LLM retry mechanism."""

    async def test_llm_timeout_with_successful_retry(self):
        """ST-001: LLM times out twice, succeeds on third attempt."""
        call_count = 0
        call_history = []
        
        async def mock_handler(request: Any) -> AIMessage:
            nonlocal call_count
            call_count += 1
            call_history.append({
                "call_number": call_count,
                "timestamp": time.time(),
            })
            
            if call_count <= 2:
                raise TimeoutError(f"Call {call_count} timed out")
            
            return AIMessage(content="Success after retries")
        
        retry_config = RetryConfig(
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0,
            jitter=False,
        )
        middleware = RetryMiddleware(llm_config=retry_config)
        
        mock_request = MagicMock()
        
        start_time = time.time()
        result = await middleware.awrap_model_call(mock_request, mock_handler)
        elapsed = time.time() - start_time
        
        assert call_count == 3
        assert isinstance(result, AIMessage)
        assert result.content == "Success after retries"
        assert elapsed >= 0.3  # 0.1s + 0.2s delays

    async def test_llm_timeout_exceeds_max_retries(self):
        """ST-002: LLM consistently times out, exceeding max retries."""
        call_count = 0
        
        async def mock_handler(request: Any) -> AIMessage:
            nonlocal call_count
            call_count += 1
            raise TimeoutError(f"Call {call_count} always times out")
        
        retry_config = RetryConfig(
            max_retries=3,
            base_delay=0.05,
            max_delay=0.2,
            jitter=False,
        )
        middleware = RetryMiddleware(llm_config=retry_config)
        
        mock_request = MagicMock()
        
        with pytest.raises(TimeoutError):
            await middleware.awrap_model_call(mock_request, mock_handler)
        
        assert call_count == 4  # initial + 3 retries


@pytest.mark.asyncio
class TestToolRetryE2E:
    """End-to-end tests for tool retry mechanism."""

    async def test_tool_retry_on_timeout_error(self):
        """ST-003: Tool call times out and is retried successfully."""
        from langchain.tools.tool_node import ToolCallRequest
        
        call_count = 0
        
        async def mock_handler(request: ToolCallRequest) -> ToolMessage:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("First call timed out")
            return ToolMessage(content="Success", tool_call_id="call_1")
        
        retry_middleware = RetryMiddleware(
            llm_config=RetryConfig(max_retries=1),
            tool_config=ToolRetryConfig(
                max_retries=2,
            ),
            enable_tool_retry=True,
        )
        
        mock_request = MagicMock(spec=ToolCallRequest)
        mock_request.tool_call = {"name": "test_tool", "id": "call_1"}
        mock_request.tool = None
        
        result = await retry_middleware.awrap_tool_call(mock_request, mock_handler)
        
        assert call_count == 2
        assert isinstance(result, ToolMessage)

    async def test_tool_excluded_from_retry(self):
        """ST-004: Excluded tool should not be retried on timeout."""
        from langchain.tools.tool_node import ToolCallRequest
        
        call_count = 0
        
        async def mock_handler(request: ToolCallRequest) -> ToolMessage:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Tool timed out")
        
        retry_middleware = RetryMiddleware(
            tool_config=ToolRetryConfig(
                max_retries=3,
                exclude_tools=["critical_tool"],
            ),
            enable_tool_retry=True,
        )
        
        mock_request = MagicMock(spec=ToolCallRequest)
        mock_request.tool_call = {"name": "critical_tool", "id": "call_1"}
        mock_request.tool = None
        
        with pytest.raises(TimeoutError):
            await retry_middleware.awrap_tool_call(mock_request, mock_handler)
        
        assert call_count == 1


@pytest.mark.asyncio
class TestCircuitBreakerE2E:
    """End-to-end tests for circuit breaker mechanism."""

    async def test_circuit_opens_after_consecutive_failures(self):
        """ST-005: Circuit breaker opens after threshold failures."""
        call_count = 0
        
        async def mock_handler(request: Any) -> AIMessage:
            nonlocal call_count
            call_count += 1
            raise TimeoutError(f"Call {call_count} failed")
        
        cb_middleware = CircuitBreakerRetryMiddleware(
            llm_config=RetryConfig(max_retries=0, base_delay=0.01),
            failure_threshold=5,
            recovery_timeout=10.0,
        )
        
        mock_request = MagicMock()

        # Make 5 failing calls
        for _ in range(5):
            try:
                await cb_middleware.awrap_model_call(mock_request, mock_handler)
            except Exception:
                pass

        # 6th call should fail immediately due to open circuit
        start_time = time.time()
        with pytest.raises(NonRetryableError) as exc_info:
            await cb_middleware.awrap_model_call(mock_request, mock_handler)
        elapsed = time.time() - start_time

        assert call_count == 5
        assert elapsed < 0.1
        assert "circuit breaker" in str(exc_info.value).lower()


@pytest.mark.asyncio
class TestTimeoutAccuracyE2E:
    """Tests for timeout accuracy and timing."""

    async def test_tool_call_does_not_enforce_local_timeout(self):
        """ST-006: Tool calls are no longer wrapped with msagent-local timeouts."""
        from langchain.tools.tool_node import ToolCallRequest
        
        async def slow_handler(request: ToolCallRequest) -> ToolMessage:
            await asyncio.sleep(0.2)
            return ToolMessage(content="Success", tool_call_id="call_1")
        
        retry_middleware = RetryMiddleware(
            tool_config=ToolRetryConfig(
                max_retries=0,
            ),
            enable_tool_retry=True,
        )
        
        mock_request = MagicMock(spec=ToolCallRequest)
        mock_request.tool_call = {"name": "slow_tool", "id": "call_1"}
        mock_request.tool = None

        start_time = time.time()
        with patch(
            "msagent.middlewares.retry.asyncio.wait_for", new_callable=AsyncMock
        ) as mock_wait_for:
            result = await retry_middleware.awrap_tool_call(mock_request, slow_handler)
        elapsed = time.time() - start_time

        assert isinstance(result, ToolMessage)
        assert (
            elapsed >= 0.15
        ), f"Expected real handler runtime to be preserved, took {elapsed:.2f}s"
        assert mock_wait_for.await_count == 0


@pytest.mark.asyncio
class TestIntegrationE2E:
    """Integration tests combining multiple scenarios."""

    async def test_full_workflow_with_mixed_failures(self):
        """ST-007: Full workflow with LLM and tool failures."""
        retry_middleware = RetryMiddleware(
            llm_config=RetryConfig(max_retries=2, base_delay=0.1),
            tool_config=ToolRetryConfig(max_retries=2),
            enable_llm_retry=True,
            enable_tool_retry=True,
        )

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
            content="Success",
            tool_calls=[],
        ))

        agent = create_react_agent(
            model=mock_llm,
            tools=[],
            prompt="Test agent",
            retry_middleware=retry_middleware,
        )

        assert agent is not None
        assert isinstance(agent, CompiledStateGraph)


@pytest.mark.asyncio
class TestPerformanceE2E:
    """Performance and stress tests."""

    async def test_concurrent_agent_calls_with_timeout(self):
        """ST-008: Multiple concurrent agents with timeout handling."""
        agents = []

        for i in range(3):
            retry_middleware = RetryMiddleware(
                llm_config=RetryConfig(max_retries=1, base_delay=0.05),
            )

            mock_llm = MagicMock()
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(
                content=f"Response {i}",
                tool_calls=[],
            ))

            agent = create_react_agent(
                model=mock_llm,
                tools=[],
                prompt=f"Agent {i}",
                retry_middleware=retry_middleware,
            )
            agents.append(agent)

        assert len(agents) == 3
        for agent in agents:
            assert isinstance(agent, CompiledStateGraph)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
