from __future__ import annotations

from msagent.agents.local_context import (
    build_local_environment_context,
    ensure_local_context_prompt,
)


def test_ensure_local_context_prompt_appends_placeholder_once() -> None:
    prompt = "You are a coding assistant."
    augmented = ensure_local_context_prompt(prompt)

    assert "{local_environment_context}" in augmented
    assert augmented.count("{local_environment_context}") == 1

    augmented_again = ensure_local_context_prompt(augmented)
    assert augmented_again == augmented


def test_build_local_environment_context_includes_project_signals(tmp_path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("# lock", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (tmp_path / "src").mkdir()

    context = build_local_environment_context(tmp_path)

    assert "## Local Runtime Snapshot" in context
    assert "## Project Signals" in context
    assert "Python" in context
    assert "uv" in context
    assert "## Top-Level Workspace Layout" in context
