from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from msagent.cli.completers.reference import ReferenceCompleter


async def _collect_texts(
    completer: ReferenceCompleter,
    text: str,
) -> list[str]:
    document = Document(text=text, cursor_position=len(text))
    complete_event = CompleteEvent(completion_requested=True)
    return [
        completion.text
        async for completion in completer.get_completions_async(document, complete_event)
    ]


@pytest.mark.asyncio
async def test_reference_completer_suggests_top_level_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# demo", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()

    completer = ReferenceCompleter(tmp_path, max_suggestions=10)

    completions = await _collect_texts(completer, "@sr")

    assert "src/" in completions
    assert "node_modules/" not in completions


@pytest.mark.asyncio
async def test_reference_completer_suggests_nested_paths_for_manual_input(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "msagent").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')", encoding="utf-8")

    completer = ReferenceCompleter(tmp_path, max_suggestions=10)

    completions = await _collect_texts(completer, "@src/m")

    assert "src/main.py" in completions
    assert "src/msagent/" in completions


@pytest.mark.asyncio
async def test_reference_completer_supports_absolute_root_prefix_completion() -> None:
    completer = ReferenceCompleter(Path.cwd(), max_suggestions=20)

    completions = await _collect_texts(completer, "@/us")

    # /usr/ exists on both macOS and Linux, /Users/ only on macOS
    assert "/usr/" in completions


@pytest.mark.asyncio
async def test_reference_completer_skips_completed_file(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# demo", encoding="utf-8")

    completer = ReferenceCompleter(tmp_path, max_suggestions=10)

    completions = await _collect_texts(completer, "@README.md")

    assert completions == []


@pytest.mark.asyncio
async def test_reference_completer_does_not_trigger_for_email_like_text(
    tmp_path: Path,
) -> None:
    completer = ReferenceCompleter(tmp_path, max_suggestions=10)

    completions = await _collect_texts(completer, "foo@bar")

    assert completions == []
