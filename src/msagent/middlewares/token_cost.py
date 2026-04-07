"""Token cost extraction utilities."""

from __future__ import annotations

from typing import Any


def extract_usage_counts(response: Any) -> tuple[int | None, int | None]:
    """Extract token usage counts from model response.
    
    Args:
        response: Model response object
        
    Returns:
        Tuple of (input_tokens, output_tokens)
    """
    input_tokens = None
    output_tokens = None
    
    # Try to extract from various response formats
    if hasattr(response, "usage_metadata"):
        usage = response.usage_metadata
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
        elif usage:
            input_tokens = getattr(usage, "input_tokens", None)
            output_tokens = getattr(usage, "output_tokens", None)

    if input_tokens is None and hasattr(response, "usage"):
        usage = response.usage
        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")
        elif usage:
            input_tokens = getattr(usage, "prompt_tokens", None)
            output_tokens = getattr(usage, "completion_tokens", None)
            
    return input_tokens, output_tokens
