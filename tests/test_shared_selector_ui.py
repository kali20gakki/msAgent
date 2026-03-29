from pathlib import Path
from types import SimpleNamespace

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from msagent.cli.ui.shared import (
    SelectorState,
    create_instruction,
    create_selector_application,
)
from msagent.configs import ApprovalMode


def _build_context() -> SimpleNamespace:
    return SimpleNamespace(
        approval_mode=ApprovalMode.ACTIVE,
        bash_mode=False,
        working_dir=Path.cwd(),
        model="demo-model",
        model_display="demo-model",
        current_input_tokens=None,
        current_output_tokens=None,
        context_window=None,
    )


def test_create_selector_application_wraps_content_with_shared_shell() -> None:
    context = _build_context()
    text_control = FormattedTextControl("selector body")
    key_bindings = KeyBindings()
    header_windows = create_instruction("Use arrows", spacer=False)
    body_windows = [Window(height=1, char=" ")]

    app = create_selector_application(
        context=context,
        text_control=text_control,
        key_bindings=key_bindings,
        header_windows=header_windows,
        body_windows=body_windows,
    )

    root = app.layout.container
    assert isinstance(root, HSplit)
    assert root.children[0] is header_windows[0]
    assert isinstance(root.children[1], Window)
    assert root.children[1].content is text_control
    assert root.children[2] is body_windows[0]
    assert isinstance(root.children[3], Window)
    assert app.key_bindings is key_bindings
    assert app.erase_when_done is True


def test_create_selector_application_accepts_custom_content_window() -> None:
    context = _build_context()
    key_bindings = KeyBindings()
    content_window = Window(content=FormattedTextControl("viewer body"), wrap_lines=False)

    app = create_selector_application(
        context=context,
        content_window=content_window,
        key_bindings=key_bindings,
        full_screen=True,
        mouse_support=True,
    )

    root = app.layout.container
    assert isinstance(root, HSplit)
    assert root.children[0] is content_window
    assert app.full_screen is True
    assert bool(app.mouse_support()) is True


def test_selector_state_moves_cyclically() -> None:
    state = SelectorState(index=0)

    state.move_cyclic(-1, size=3)
    assert state.index == 2

    state.move_cyclic(1, size=3)
    assert state.index == 0


def test_selector_state_moves_linearly_and_tracks_scroll_window() -> None:
    state = SelectorState(index=0, scroll_offset=0, window_size=3)

    state.move_linear(1, size=5)
    assert (state.index, state.scroll_offset) == (1, 0)

    state.move_linear(2, size=5)
    assert (state.index, state.scroll_offset) == (3, 1)

    state.move_linear(-2, size=5)
    assert (state.index, state.scroll_offset) == (1, 1)

    state.move_linear(-1, size=5)
    assert (state.index, state.scroll_offset) == (0, 0)
