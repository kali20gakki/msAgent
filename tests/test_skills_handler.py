#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from msagent.cli.dispatchers import commands as commands_module
from msagent.cli.dispatchers.commands import CommandDispatcher
from msagent.cli.handlers import skills as skills_module
from msagent.cli.handlers.skills import SkillsHandler
from msagent.skills.factory import DEFAULT_SKILL_CATEGORY, Skill


def _build_skill(
    *,
    name: str,
    description: str,
    category: str = DEFAULT_SKILL_CATEGORY,
) -> Skill:
    path = Path("/tmp") / category / name / "SKILL.md"
    return Skill(
        name=name,
        description=description,
        category=category,
        path=path,
    )


def _build_session(*, prefilled_text: str | None = None) -> SimpleNamespace:
    rendered_messages: list[object] = []
    return SimpleNamespace(
        prefilled_text=prefilled_text,
        renderer=SimpleNamespace(render_user_message=rendered_messages.append),
        message_dispatcher=SimpleNamespace(dispatch=AsyncMock()),
        context=SimpleNamespace(working_dir=Path.cwd(), bash_mode=False),
        rendered_messages=rendered_messages,
    )


def test_skills_handler_hides_default_category_for_legacy_skills() -> None:
    skill = _build_skill(
        name="op-mfu-calculator",
        description="Legacy flat skill",
    )

    formatted = SkillsHandler._format_skill_list([skill], 0, set(), 0, 10)
    rendered = "".join(text for _, text in formatted)

    assert "op-mfu-calculator" in rendered
    assert "default/op-mfu-calculator" not in rendered


def test_skills_handler_keeps_category_for_grouped_skills() -> None:
    skill = _build_skill(
        name="workspace-skill",
        description="Grouped skill",
        category="analysis",
    )

    formatted = SkillsHandler._format_skill_list([skill], 0, set(), 0, 10)
    rendered = "".join(text for _, text in formatted)

    assert "analysis/workspace-skill" in rendered


def test_skills_handler_lists_inline_description_preview() -> None:
    skill = _build_skill(
        name="workspace-skill",
        description="Inspect the workspace and summarize the important context.",
        category="analysis",
    )

    formatted = SkillsHandler._format_skill_list([skill], 0, set(), 0, 10)
    rendered = "".join(text for _, text in formatted)

    assert "Inspect the workspace and summarize the important context." in rendered


def test_resolve_skill_requires_qualified_name_when_ambiguous() -> None:
    skills = [
        _build_skill(name="workspace-skill", description="One", category="analysis"),
        _build_skill(name="workspace-skill", description="Two", category="ops"),
    ]

    with pytest.raises(ValueError) as exc_info:
        SkillsHandler._resolve_skill(skills, "workspace-skill")

    assert "analysis/workspace-skill" in str(exc_info.value)
    assert "ops/workspace-skill" in str(exc_info.value)


@pytest.mark.asyncio
async def test_handle_with_skill_name_prefills_next_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session()
    handler = SkillsHandler(session)
    skill = _build_skill(
        name="workspace-skill",
        description="Grouped skill",
        category="analysis",
    )

    monkeypatch.setattr(skills_module.console, "print_success", lambda *_args: None)
    monkeypatch.setattr(skills_module.console, "print", lambda *_args, **_kwargs: None)

    await handler.handle([skill], args=["analysis/workspace-skill"])

    assert session.prefilled_text is not None
    assert session.prefilled_text == "/workspace-skill "
    session.message_dispatcher.dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_with_skill_name_and_task_dispatches_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session()
    handler = SkillsHandler(session)
    skill = _build_skill(
        name="workspace-skill",
        description="Grouped skill",
        category="analysis",
    )

    monkeypatch.setattr(skills_module.console, "print_success", lambda *_args: None)
    monkeypatch.setattr(skills_module.console, "print", lambda *_args, **_kwargs: None)

    await handler.handle(
        [skill],
        args=["analysis/workspace-skill", "summarize", "the", "workspace"],
    )

    session.message_dispatcher.dispatch.assert_awaited_once()
    dispatched_prompt = session.message_dispatcher.dispatch.await_args.args[0]
    assert "Use the skill `analysis/workspace-skill`" in dispatched_prompt
    assert dispatched_prompt.endswith("Task:\nsummarize the workspace")
    assert len(session.rendered_messages) == 1
    rendered_message = session.rendered_messages[0]
    assert getattr(rendered_message, "short_content") == "/workspace-skill summarize the workspace"


def test_queue_skill_uses_category_shortcut_when_name_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session()
    handler = SkillsHandler(session)
    skills = [
        _build_skill(name="workspace-skill", description="One", category="analysis"),
        _build_skill(name="workspace-skill", description="Two", category="ops"),
    ]

    monkeypatch.setattr(skills_module.console, "print_success", lambda *_args: None)
    monkeypatch.setattr(skills_module.console, "print", lambda *_args, **_kwargs: None)

    handler._queue_skill_for_next_prompt(skills[0], skills)

    assert session.prefilled_text == "/analysis:workspace-skill "


@pytest.mark.asyncio
async def test_command_dispatcher_routes_skill_shortcut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SimpleNamespace(
        context=SimpleNamespace(
            working_dir=Path.cwd(),
            bash_mode=False,
            thread_id="thread-1",
            current_input_tokens=None,
            current_output_tokens=None,
        ),
        prompt=SimpleNamespace(hotkeys={}),
        renderer=SimpleNamespace(render_help=lambda *_args, **_kwargs: None),
        update_context=lambda **_kwargs: None,
        running=True,
    )
    dispatcher = CommandDispatcher(session)
    skill = _build_skill(
        name="workspace-skill",
        description="Grouped skill",
        category="analysis",
    )
    shortcut_mock = AsyncMock(return_value=True)

    monkeypatch.setattr(commands_module.initializer, "cached_agent_skills", [skill])
    monkeypatch.setattr(dispatcher.skills_handler, "handle_shortcut", shortcut_mock)
    monkeypatch.setattr(commands_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(commands_module.console, "print", lambda *_args, **_kwargs: None)

    await dispatcher.dispatch("/workspace-skill summarize the workspace")

    shortcut_mock.assert_awaited_once_with(
        [skill],
        "workspace-skill",
        ["summarize", "the", "workspace"],
        raw_input="/workspace-skill summarize the workspace",
    )


def test_skills_handler_build_description_preview_shortens_long_text() -> None:
    long_desc = "A very long description that goes on and on about what the skill does " * 5
    preview = SkillsHandler._build_description_preview(long_desc, width=60)
    assert len(preview) <= 63
    assert preview.endswith("...") or len(" ".join(long_desc.split())) <= 60


def test_skills_handler_build_description_preview_returns_no_description_for_empty() -> None:
    preview = SkillsHandler._build_description_preview("", width=80)
    assert preview == "No description"


def test_skills_handler_wrap_description_preserves_paragraph_breaks() -> None:
    desc = "First paragraph.\n\nSecond paragraph with more text."
    lines = SkillsHandler._wrap_description(desc, width=80)
    assert "First paragraph." in lines
    assert "" in lines
    assert "Second paragraph with more text." in lines


def test_skills_handler_wrap_description_returns_no_description_for_empty() -> None:
    lines = SkillsHandler._wrap_description("", width=80)
    assert lines == ["No description"]


def test_skills_handler_build_skill_task_prompt_includes_get_skill_call() -> None:
    skill = _build_skill(name="workspace-skill", description="A skill", category="analysis")
    prompt = SkillsHandler._build_skill_task_prompt(skill, task="analyze the code")
    assert "get_skill(name=\"workspace-skill\"" in prompt
    assert "category=\"analysis\"" in prompt
    assert "analyze the code" in prompt


def test_skills_handler_build_skill_task_prompt_without_task() -> None:
    skill = _build_skill(name="workspace-skill", description="A skill", category="analysis")
    prompt = SkillsHandler._build_skill_task_prompt(skill, task=None)
    assert "get_skill(name=\"workspace-skill\"" in prompt
    assert "Task:\n" in prompt


def test_skills_handler_sort_skills_by_display_name_casefold() -> None:
    skills = [
        _build_skill(name="Z-skill", description="Z"),
        _build_skill(name="a-skill", description="a"),
        _build_skill(name="M-skill", description="M"),
    ]
    sorted_skills = SkillsHandler._sort_skills(skills)
    assert sorted_skills[0].name == "a-skill"
    assert sorted_skills[1].name == "M-skill"
    assert sorted_skills[2].name == "Z-skill"


def test_skills_handler_normalize_skill_ref_is_case_insensitive() -> None:
    assert SkillsHandler._normalize_skill_ref("Workspace-Skill") == "workspace-skill"
    assert SkillsHandler._normalize_skill_ref("  ANALYSIS/OP  ") == "analysis/op"


def test_skills_handler_resolve_skill_raises_on_empty_ref() -> None:
    skills = [_build_skill(name="skill", description="desc")]
    with pytest.raises(ValueError, match="Skill name is required"):
        SkillsHandler._resolve_skill(skills, "")


def test_skills_handler_resolve_skill_raises_on_not_found() -> None:
    skills = [_build_skill(name="existing", description="desc")]
    with pytest.raises(ValueError, match="not found"):
        SkillsHandler._resolve_skill(skills, "nonexistent")


def test_skills_handler_try_resolve_skill_returns_none_for_not_found() -> None:
    skills = [_build_skill(name="existing", description="desc")]
    result = SkillsHandler._try_resolve_skill(skills, "nonexistent")
    assert result is None


def test_skills_handler_try_resolve_skill_raises_for_ambiguous() -> None:
    skills = [
        _build_skill(name="dup", description="One", category="a"),
        _build_skill(name="dup", description="Two", category="b"),
    ]
    with pytest.raises(ValueError, match="Multiple skills"):
        SkillsHandler._try_resolve_skill(skills, "dup")


def test_skills_handler_build_shortcut_name_uses_category_when_ambiguous() -> None:
    skills = [
        _build_skill(name="dup", description="One", category="a"),
        _build_skill(name="dup", description="Two", category="b"),
    ]
    name = SkillsHandler._build_shortcut_name(skills[0], skills)
    assert name == "a:dup"


def test_skills_handler_build_shortcut_name_uses_bare_name_when_unique() -> None:
    skills = [_build_skill(name="unique", description="desc")]
    name = SkillsHandler._build_shortcut_name(skills[0], skills)
    assert name == "unique"


def test_skills_handler_build_shortcut_input_appends_task() -> None:
    skill = _build_skill(name="skill", description="desc")
    result = SkillsHandler._build_shortcut_input(skill, [skill], task="my task")
    assert result == "/skill my task"


def test_skills_handler_build_shortcut_input_adds_trailing_space_without_task() -> None:
    skill = _build_skill(name="skill", description="desc")
    result = SkillsHandler._build_shortcut_input(skill, [skill], task=None)
    assert result == "/skill "


@pytest.mark.asyncio
async def test_skills_handler_reports_no_skills_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(skills_module.console, "print_error", errors.append)
    monkeypatch.setattr(skills_module.console, "print", lambda *_args, **_kwargs: None)

    session = _build_session()
    handler = SkillsHandler(session)
    await handler.handle([])
    assert "No skills available" in errors


@pytest.mark.asyncio
async def test_skills_handle_shortcut_returns_false_for_unknown_skill() -> None:
    session = _build_session()
    handler = SkillsHandler(session)
    skills = [_build_skill(name="existing", description="desc")]
    result = await handler.handle_shortcut(skills, "nonexistent")
    assert result is False


@pytest.mark.asyncio
async def test_skills_handle_shortcut_queues_skill_without_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session()
    handler = SkillsHandler(session)
    skill = _build_skill(name="skill", description="desc")
    monkeypatch.setattr(skills_module.console, "print_success", lambda *_args: None)
    monkeypatch.setattr(skills_module.console, "print", lambda *_args, **_kwargs: None)

    result = await handler.handle_shortcut([skill], "skill")
    assert result is True
    assert session.prefilled_text is not None


@pytest.mark.asyncio
async def test_skills_handle_shortcut_runs_skill_with_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session()
    handler = SkillsHandler(session)
    skill = _build_skill(name="skill", description="desc")
    monkeypatch.setattr(skills_module.console, "print_success", lambda *_args: None)
    monkeypatch.setattr(skills_module.console, "print", lambda *_args, **_kwargs: None)

    result = await handler.handle_shortcut([skill], "skill", args=["do", "something"])
    assert result is True
    session.message_dispatcher.dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_skills_handler_handles_exception_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors: list[str] = []
    monkeypatch.setattr(skills_module.console, "print_error", errors.append)
    monkeypatch.setattr(skills_module.console, "print", lambda *_args, **_kwargs: None)

    session = _build_session()
    handler = SkillsHandler(session)

    async def fake_get_skill_selection(_skills):
        raise RuntimeError("UI failure")

    monkeypatch.setattr(handler, "_get_skill_selection", fake_get_skill_selection)

    skill = _build_skill(name="skill", description="desc")
    await handler.handle([skill])

    assert any("Error displaying skills" in e for e in errors)
