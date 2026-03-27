from pathlib import Path
from types import SimpleNamespace

import pytest

from msagent.tools.factory import ToolFactory
from msagent.tools.impl.file_system import glob, ls
from msagent.utils.render import create_tool_message


def test_tool_factory_exposes_deepagents_compatible_aliases() -> None:
    tool_names = {tool.name for tool in ToolFactory().get_impl_tools()}

    assert "ls" in tool_names
    assert "glob" in tool_names
    assert "grep" in tool_names


@pytest.mark.asyncio
async def test_ls_tool_lists_directory_contents(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "nested.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "top.txt").write_text("world", encoding="utf-8")

    runtime = SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path),
        tool_call_id="call-ls",
    )
    result = await ls.coroutine(runtime=runtime, dir_path=".", recursive=True)

    assert "./" in result.content
    assert "dir/" in result.content
    assert "nested.txt" in result.content
    assert "top.txt" in result.content


@pytest.mark.asyncio
async def test_glob_tool_matches_files(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "one.py").write_text("print('one')", encoding="utf-8")
    (tmp_path / "src" / "two.txt").write_text("two", encoding="utf-8")

    runtime = SimpleNamespace(
        context=SimpleNamespace(working_dir=tmp_path),
        tool_call_id="call-glob",
    )
    result = await glob.coroutine(
        pattern="src/**/*.py",
        runtime=runtime,
        dir_path=".",
    )

    assert "src/one.py" in result.content
    assert "two.txt" not in result.content


def test_create_tool_message_truncates_result_with_original_length_note() -> None:
    result = create_tool_message(
        result="x" * 240,
        tool_name="run_command",
        tool_call_id="call-1",
    )

    assert result.text == "x" * 240
    assert getattr(result, "short_content") == (
        ("x" * 200) + "... (truncated, original length: 240)"
    )
