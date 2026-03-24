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
