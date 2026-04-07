from __future__ import annotations

from pathlib import Path

import pytest

from msagent.cli.resolvers.file import FileResolver


@pytest.mark.asyncio
async def test_get_files_falls_back_to_fd_when_git_lookup_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    async def fake_execute_bash_command(cmd, cwd, timeout):
        del cwd, timeout
        calls.append(cmd)
        if len(calls) == 1:
            return 1, "", "git failed"
        return 0, "src/main.py\nREADME.md\n", ""

    monkeypatch.setattr(
        "msagent.cli.resolvers.file.execute_bash_command",
        fake_execute_bash_command,
    )

    files = await FileResolver._get_files(tmp_path, limit=5, pattern="src")

    assert files == ["src/main.py", "README.md"]
    assert len(calls) == 2
    assert "git ls-files" in calls[0][2]
    assert "fd --type f" in calls[1][2]


@pytest.mark.asyncio
async def test_file_resolver_complete_sorts_dirs_before_files_and_applies_styles(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def fake_get_files(working_dir, limit=None, pattern=""):
        del working_dir, limit, pattern
        return ["src/main.py", "docs/guide.md"]

    async def fake_get_directories(working_dir, limit=None, pattern=""):
        del working_dir, limit, pattern
        return ["src", "docs"]

    monkeypatch.setattr(FileResolver, "_get_files", staticmethod(fake_get_files))
    monkeypatch.setattr(
        FileResolver,
        "_get_directories",
        staticmethod(fake_get_directories),
    )

    resolver = FileResolver()
    completions = await resolver.complete(
        fragment="s",
        ctx={"working_dir": tmp_path, "start_position": -2},
        limit=10,
    )

    assert [completion.text for completion in completions] == [
        "@:file:docs",
        "@:file:src",
        "@:file:docs/guide.md",
        "@:file:src/main.py",
    ]
    assert [completion.display_text for completion in completions[:2]] == [
        "@:file:docs/",
        "@:file:src/",
    ]
    assert [completion.style for completion in completions] == [
        "class:dir-completion",
        "class:dir-completion",
        "class:file-completion",
        "class:file-completion",
    ]
    assert all(completion.start_position == -2 for completion in completions)


@pytest.mark.asyncio
async def test_file_resolver_complete_returns_empty_when_lookup_raises(
    monkeypatch,
    tmp_path: Path,
) -> None:
    async def failing_get_files(working_dir, limit=None, pattern=""):
        del working_dir, limit, pattern
        raise RuntimeError("boom")

    monkeypatch.setattr(FileResolver, "_get_files", staticmethod(failing_get_files))

    resolver = FileResolver()
    completions = await resolver.complete(
        fragment="x",
        ctx={"working_dir": tmp_path},
        limit=5,
    )

    assert completions == []


def test_file_resolver_resolve_returns_absolute_path_for_valid_reference(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    file_path = tmp_path / "src" / "main.py"
    file_path.write_text("print('ok')", encoding="utf-8")

    resolver = FileResolver()
    resolved = Path(resolver.resolve("src/main.py", ctx={"working_dir": tmp_path}))

    assert resolved == file_path.resolve()


def test_file_resolver_resolve_falls_back_to_original_on_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def broken_resolve_path(*args, **kwargs):
        del args, kwargs
        raise ValueError("invalid")

    monkeypatch.setattr("msagent.cli.resolvers.file.resolve_path", broken_resolve_path)

    resolver = FileResolver()
    result = resolver.resolve("relative/path.txt", ctx={"working_dir": tmp_path})

    assert result == "relative/path.txt"


def test_file_resolver_detects_standalone_absolute_path(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("note", encoding="utf-8")

    resolver = FileResolver()

    assert resolver.is_standalone_reference(str(file_path)) is True
    assert resolver.is_standalone_reference("notes.txt") is False
