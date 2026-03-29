from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text

from msagent.cli.core.context import Context
from msagent.cli.ui.prompt import InteractivePrompt
from msagent.cli.ui.shared import build_agent_prompt
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


def test_placeholder_text_uses_current_agent_name() -> None:
    prompt = InteractivePrompt.__new__(InteractivePrompt)
    prompt.context = _build_prompt_context(agent="Hermes")

    text = prompt._build_placeholder_text()

    assert text == "尽管问Hermes，@ 引用文件，/ 使用命令"


def test_placeholder_text_stays_consistent_in_bash_mode() -> None:
    prompt = InteractivePrompt.__new__(InteractivePrompt)
    prompt.context = _build_prompt_context(agent="general", bash_mode=True)

    text = prompt._build_placeholder_text()

    assert text == "尽管问general，@ 引用文件，/ 使用命令"


def test_prompt_hotkeys_include_tool_output_toggle() -> None:
    prompt = InteractivePrompt.__new__(InteractivePrompt)
    prompt.context = _build_prompt_context()
    prompt.commands = ["/help", "/tool-output"]
    prompt.session = SimpleNamespace(prefilled_text=None)
    prompt.hotkeys = {}

    prompt._create_key_bindings()

    assert "Ctrl+O" in prompt.hotkeys
    assert prompt.hotkeys["Ctrl+O"] == "Expand/collapse latest tool output"


def test_build_agent_prompt_uses_current_agent_name() -> None:
    prompt_text = build_agent_prompt(_build_prompt_context(agent="Hermes"))

    assert prompt_text == "Hermes > "


@pytest.mark.asyncio
async def test_get_input_disables_prompt_toolkit_sigint_handling() -> None:
    prompt = InteractivePrompt.__new__(InteractivePrompt)
    prompt.context = _build_prompt_context()
    prompt.commands = ["/help"]
    prompt.session = SimpleNamespace(prefilled_text=None)

    captured: dict[str, object] = {}

    class FakePromptSession:
        async def prompt_async(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return "/help"

    prompt.prompt_session = FakePromptSession()

    content, is_command = await prompt.get_input()

    assert content == "/help"
    assert is_command is True
    assert captured["args"][0][0][1] == "general > "
    assert captured["kwargs"]["handle_sigint"] is False


def test_handle_external_sigint_shows_quit_hint_when_prompt_is_running() -> None:
    prompt = InteractivePrompt.__new__(InteractivePrompt)
    prompt._last_ctrl_c_time = None
    prompt._ctrl_c_timeout = 0.30
    prompt._show_quit_message = False
    scheduled = {}

    class FakeBuffer:
        text = ""

    class FakeApp:
        is_running = True
        current_buffer = FakeBuffer()

        class loop:
            @staticmethod
            def call_later(delay, callback):
                scheduled["delay"] = delay
                scheduled["callback"] = callback

        def invalidate(self):
            scheduled["invalidated"] = True

        def exit(self, exception=None):
            scheduled["exit_exception"] = exception

    prompt.prompt_session = SimpleNamespace(app=FakeApp())

    handled = prompt.handle_external_sigint()

    assert handled is True
    assert prompt._show_quit_message is True
    assert scheduled["delay"] == 0.30
    assert scheduled["invalidated"] is True
    assert "exit_exception" not in scheduled
