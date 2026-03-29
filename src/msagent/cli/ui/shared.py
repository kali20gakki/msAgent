"""Shared UI functions for consistent styling across prompt sessions."""

import html
import os
from dataclasses import dataclass

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style

from msagent.cli.theme import theme
from msagent.configs import ApprovalMode
from msagent.core.settings import settings
from msagent.utils.cost import calculate_context_percentage, format_tokens
from msagent.utils.version import get_version


@dataclass
class SelectorState:
    """Shared selection state for prompt-toolkit list UIs."""

    index: int = 0
    scroll_offset: int = 0
    window_size: int | None = None

    def move_cyclic(self, delta: int, *, size: int) -> None:
        """Move selection with wrap-around semantics."""
        if size <= 0:
            self.index = 0
            return
        self.index = (self.index + delta) % size

    def move_linear(self, delta: int, *, size: int) -> None:
        """Move selection within bounds and keep it visible in the scroll window."""
        if size <= 0:
            self.index = 0
            self.scroll_offset = 0
            return

        next_index = max(0, min(self.index + delta, size - 1))
        self.index = next_index

        if self.window_size is None or self.window_size <= 0:
            return

        if self.index < self.scroll_offset:
            self.scroll_offset = self.index
        elif self.index >= self.scroll_offset + self.window_size:
            self.scroll_offset = self.index - self.window_size + 1


def get_prompt_color(context, *, bash_mode: bool = False) -> str:
    """Get prompt color based on approval mode and bash mode."""
    if bash_mode:
        return theme.danger_color
    mode_colors = {
        ApprovalMode.SEMI_ACTIVE: theme.approval_semi_active,
        ApprovalMode.ACTIVE: theme.approval_active,
        ApprovalMode.AGGRESSIVE: theme.approval_aggressive,
    }
    return mode_colors[context.approval_mode]


def build_agent_prompt(context) -> str:
    """Build the visible prompt prefix for the current agent."""
    base_prompt = settings.cli.prompt_style
    agent_name = str(getattr(context, "agent", "") or "").strip()
    if not agent_name:
        return base_prompt
    return f"{agent_name} {base_prompt}"


def create_prompt_style(context, *, bash_mode: bool = False) -> Style:
    """Create prompt style based on theme and approval mode."""
    prompt_color = get_prompt_color(context, bash_mode=bash_mode)

    return Style.from_dict(
        {
            # Prompt styling - dynamic based on approval mode
            "prompt": f"{prompt_color} nobold",
            "prompt.muted": f"{prompt_color} nobold",
            "prompt.arg": f"{theme.accent_color}",
            # Input styling
            "": f"{theme.primary_text}",
            "text": f"{theme.primary_text}",
            # Completion styling
            "completion-menu.completion": f"{theme.primary_text} bg:{theme.background_light}",
            "completion-menu.completion.current": f"{theme.background} bg:{theme.prompt_color}",
            "completion-menu.meta.completion": f"{theme.muted_text} bg:{theme.background_light}",
            "completion-menu.meta.completion.current": f"{theme.primary_text} bg:{theme.prompt_color}",
            # Thread completion styling
            "thread-completion": f"{theme.accent_color} bg:{theme.background_light}",
            # File/directory completion styling
            "file-completion": f"{theme.primary_text} bg:{theme.background_light}",
            "dir-completion": f"{theme.info_color} bg:{theme.background_light}",
            # Auto-suggestion styling
            "auto-suggestion": f"{theme.muted_text} italic",
            # Validation styling
            "validation-toolbar": f"{theme.error_color} bg:{theme.background_light}",
            # Selection styling
            "selected": f"bg:{theme.selection_color}",
            # Search styling
            "search": f"{theme.accent_color} bg:{theme.background_light}",
            "search.current": f"{theme.background} bg:{theme.warning_color}",
            # Placeholder styling
            "placeholder": f"{theme.muted_text} italic",
            # Muted text styling
            "muted": f"{theme.muted_text}",
            # Bottom toolbar styling - override default reverse
            "bottom-toolbar": f"noreverse {theme.muted_text}",
            "bottom-toolbar.text": f"noreverse {theme.muted_text}",
            "toolbar.model": f"noreverse {theme.accent_color}",
            # Toolbar mode styling - dynamic based on approval mode
            "toolbar.mode": f"noreverse {prompt_color}",
            # Toolbar bash mode styling - danger (pink)
            "toolbar.bash": f"noreverse {theme.danger_color}",
        }
    )


def build_usage_summary(context) -> str:
    """Build ctx and token usage summary for compact status displays."""
    input_tokens = context.current_input_tokens
    if input_tokens is None or input_tokens <= 0:
        return ""

    output_tokens = context.current_output_tokens or 0
    total_tokens = input_tokens + output_tokens

    usage_parts: list[str] = []
    if context.context_window is not None and context.context_window > 0:
        tokens_formatted = format_tokens(total_tokens)
        window_formatted = format_tokens(context.context_window)
        percentage = calculate_context_percentage(total_tokens, context.context_window)
        percentage_display = int(percentage + 0.5)
        usage_parts.append(
            f"ctx {tokens_formatted}/{window_formatted} tokens ({percentage_display}%)"
        )

    usage_parts.append(f"in {format_tokens(input_tokens)}")
    usage_parts.append(f"out {format_tokens(output_tokens)}")

    return " | ".join(usage_parts)


def _truncate_middle(text: str, max_length: int) -> str:
    """Truncate text in the middle to fit the available width."""
    if max_length <= 0:
        return ""
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]

    head = (max_length - 3) // 2
    tail = max_length - 3 - head
    return f"{text[:head]}...{text[-tail:]}"


def create_bottom_toolbar(
    context,
    working_dir: str,
    *,
    bash_mode: bool = False,
):
    """Create bottom toolbar with version, directory, model, usage, and mode info."""
    terminal_width = os.get_terminal_size().columns if os.isatty(1) else 80
    version = get_version()
    working_dir_str = str(working_dir)
    model_label = getattr(context, "model_display", None) or getattr(
        context, "model", ""
    )
    usage_summary = build_usage_summary(context)

    # Left side: version + directory
    left_prefix = f" v{version} | "
    mode_name = context.approval_mode.value

    right_parts: list[tuple[str, str]] = []
    right_plain_parts: list[str] = []

    if bash_mode:
        right_parts.append(("toolbar.bash", "bash-mode"))
        right_plain_parts.append("bash-mode")

    if model_label:
        right_parts.append(("toolbar.model", model_label))
        right_plain_parts.append(model_label)

    if usage_summary:
        usage_label = f"[{usage_summary}]"
        right_parts.append(("muted", usage_label))
        right_plain_parts.append(usage_label)

    right_parts.append(("toolbar.mode", mode_name))
    right_plain_parts.append(mode_name)

    right_content = " | ".join(right_plain_parts)
    working_dir_width = max(
        0, terminal_width - len(left_prefix) - len(right_content) - 1
    )
    display_working_dir = _truncate_middle(working_dir_str, working_dir_width)
    left_text = f"{left_prefix}{display_working_dir}"
    left_content = f"{html.escape(left_prefix)}{html.escape(display_working_dir)}"

    # Calculate padding
    padding = " " * max(0, terminal_width - len(left_text) - len(right_content) - 1)

    styled_right_parts = []
    for i, (style_name, value) in enumerate(right_parts):
        if i > 0:
            styled_right_parts.append("<muted> | </muted>")
        styled_right_parts.append(f"<{style_name}>{html.escape(value)}</{style_name}>")
    styled_right = "".join(styled_right_parts)

    return HTML(f"<muted>{left_content}{padding}</muted>{styled_right}<muted> </muted>")


def create_instruction(
    message: str,
    *,
    spacer: bool = True,
) -> list[Window]:
    """Create instruction window with optional spacer for interactive lists."""
    windows = [
        Window(
            height=1,
            content=FormattedTextControl(
                lambda: FormattedText([("class:muted", message)])
            ),
            dont_extend_height=True,
        )
    ]
    if spacer:
        windows.append(Window(height=1, char=" "))
    return windows


def create_selector_application(
    *,
    context,
    text_control: FormattedTextControl | None = None,
    key_bindings: KeyBindings,
    header_windows: list[Window] | None = None,
    body_windows: list[Window] | None = None,
    content_window: Window | None = None,
    full_screen: bool = False,
    mouse_support: bool = False,
) -> Application:
    """Create the shared prompt-toolkit shell used by list selectors."""
    if content_window is None:
        if text_control is None:
            raise ValueError(
                "text_control is required when content_window is not provided"
            )
        content_window = Window(content=text_control)

    layout_windows = [
        *(header_windows or []),
        content_window,
        *(body_windows or []),
        Window(
            height=1,
            content=FormattedTextControl(
                lambda: create_bottom_toolbar(
                    context,
                    context.working_dir,
                    bash_mode=context.bash_mode,
                )
            ),
        ),
    ]

    return Application(
        layout=Layout(HSplit(layout_windows)),
        key_bindings=key_bindings,
        full_screen=full_screen,
        style=create_prompt_style(context, bash_mode=context.bash_mode),
        erase_when_done=True,
        mouse_support=mouse_support,
    )
