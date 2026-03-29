from __future__ import annotations

from types import SimpleNamespace

from msagent.middlewares.token_cost import extract_usage_counts


def test_extract_usage_counts_prefers_usage_metadata() -> None:
    response = SimpleNamespace(
        usage_metadata={"input_tokens": 120, "output_tokens": 30},
        usage={"prompt_tokens": 1, "completion_tokens": 2},
    )

    assert extract_usage_counts(response) == (120, 30)


def test_extract_usage_counts_supports_object_style_usage() -> None:
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(input_tokens=9, output_tokens=4),
    )

    assert extract_usage_counts(response) == (9, 4)


def test_extract_usage_counts_falls_back_to_legacy_usage_fields() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=77, completion_tokens=11),
    )

    assert extract_usage_counts(response) == (77, 11)


def test_extract_usage_counts_returns_none_when_fields_are_missing() -> None:
    response = SimpleNamespace()

    assert extract_usage_counts(response) == (None, None)
