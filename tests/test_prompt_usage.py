from pathlib import Path

from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text

from msagent.cli.core.context import Context
from msagent.cli.ui.prompt import InteractivePrompt
from msagent.configs import ApprovalMode


def _build_prompt_context(**overrides) -> Context:
    data = {
        "agent": "general",
        "model": "default",
        "thread_id": "thread-1",
        "working_dir": Path.cwd(),
        "approval_mode": ApprovalMode.SEMI_ACTIVE,
        "recursion_limit": 80,
        "current_input_tokens": 6000,
        "current_output_tokens": 2000,
        "context_window": 64000,
    }
    data.update(overrides)
    return Context(**data)


def test_bottom_toolbar_shows_ctx_and_token_breakdown() -> None:
    prompt = InteractivePrompt.__new__(InteractivePrompt)
    prompt.context = _build_prompt_context()
    prompt._show_quit_message = False

    usage = fragment_list_to_text(to_formatted_text(prompt._get_bottom_toolbar()))

    assert "[ctx 8K/64K tokens (13%) | in 6K | out 2K]" in usage
    assert "$" not in usage


def test_bottom_toolbar_hides_usage_without_input_tokens() -> None:
    prompt = InteractivePrompt.__new__(InteractivePrompt)
    prompt.context = _build_prompt_context(current_input_tokens=None)
    prompt._show_quit_message = False

    usage = fragment_list_to_text(to_formatted_text(prompt._get_bottom_toolbar()))

    assert "ctx " not in usage
    assert " in " not in usage
    assert " out " not in usage


def test_placeholder_text_uses_msagent_prompt_and_hints() -> None:
    prompt = InteractivePrompt.__new__(InteractivePrompt)
    prompt.context = _build_prompt_context()

    text = prompt._build_placeholder_text()

    assert "/" in text
    assert "@" in text
    assert "ctx " not in text
    assert "general:default" not in text
