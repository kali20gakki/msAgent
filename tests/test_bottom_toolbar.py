from pathlib import Path

from prompt_toolkit.formatted_text import to_formatted_text
from prompt_toolkit.formatted_text.utils import fragment_list_to_text

from msagent.cli.core.context import Context
from msagent.cli.ui.shared import create_bottom_toolbar
from msagent.configs import ApprovalMode


def test_bottom_toolbar_shows_model_usage_and_mode() -> None:
    context = Context(
        agent="msagent",
        model="default",
        model_display="deepseek-chat (openai)",
        thread_id="thread-1",
        working_dir=Path.cwd(),
        approval_mode=ApprovalMode.SEMI_ACTIVE,
        recursion_limit=80,
        current_input_tokens=8000,
        current_output_tokens=141,
        context_window=128000,
    )

    toolbar = create_bottom_toolbar(context, context.working_dir)
    text = fragment_list_to_text(to_formatted_text(toolbar))

    assert "deepseek-chat (openai)" in text
    assert "[ctx 8K/128K tokens (6%) | in 8K | out 141]" in text
    assert "semi-active" in text
