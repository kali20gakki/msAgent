from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from msagent.cli.dispatchers.commands import CommandDispatcher
from msagent.cli.handlers import add_skill as add_skill_module
from msagent.cli.handlers.add_skill import AddSkillHandler
from msagent.skills.installer import SkillInstallError, SkillInstaller


def _write_skill(skill_dir: Path, *, name: str, description: str = "test skill") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nbody\n",
        encoding="utf-8",
    )


def _build_session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        context=SimpleNamespace(
            agent="Hermes",
            working_dir=tmp_path,
            bash_mode=False,
            thread_id="thread-1",
            current_input_tokens=None,
            current_output_tokens=None,
        ),
        prompt=SimpleNamespace(hotkeys={}),
        renderer=SimpleNamespace(
            render_help=lambda *_args, **_kwargs: None,
            render_hotkeys=lambda *_args, **_kwargs: None,
        ),
        update_context=lambda **_kwargs: None,
        running=True,
        needs_reload=False,
    )


@pytest.mark.asyncio
async def test_add_skill_handler_installs_skill_and_updates_agent_patterns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir = tmp_path / "external-skill"
    _write_skill(skill_dir, name="custom_skill")

    session = _build_session(tmp_path)
    handler = AddSkillHandler(session)
    messages: list[str] = []

    monkeypatch.setattr(add_skill_module.console, "print_success", messages.append)
    monkeypatch.setattr(add_skill_module.console, "print_error", messages.append)
    monkeypatch.setattr(add_skill_module.console, "print", lambda *_args, **_kwargs: None)

    await handler.handle([str(skill_dir)])

    installed_skill = tmp_path / ".msagent" / "skills" / "custom_skill" / "SKILL.md"
    assert installed_skill.exists()

    hermes_config = yaml.safe_load((tmp_path / ".msagent" / "agents" / "Hermes.yml").read_text(encoding="utf-8"))
    assert "default:custom_skill" in hermes_config["skills"]["patterns"]
    assert session.needs_reload is True
    assert session.running is False
    assert any("Installed skill 'custom_skill'" in message for message in messages)


@pytest.mark.asyncio
async def test_skill_installer_rejects_shadowed_skill(tmp_path: Path) -> None:
    workspace_skill_dir = tmp_path / "skills" / "workspace-skill"
    external_skill_dir = tmp_path / "incoming" / "external-skill"

    _write_skill(workspace_skill_dir, name="shadowed_skill")
    _write_skill(external_skill_dir, name="shadowed_skill")

    installer = SkillInstaller(tmp_path)

    with pytest.raises(SkillInstallError) as exc_info:
        await installer.install(str(external_skill_dir))

    assert "higher-priority directory" in str(exc_info.value)


@pytest.mark.asyncio
async def test_skill_installer_requires_description(tmp_path: Path) -> None:
    skill_dir = tmp_path / "invalid-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: invalid_skill\ndescription: ''\n---\nbody\n",
        encoding="utf-8",
    )

    installer = SkillInstaller(tmp_path)

    with pytest.raises(SkillInstallError) as exc_info:
        await installer.install(str(skill_dir))

    assert "non-empty 'description'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_skill_installer_uses_skill_name_for_target_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "incoming" / "mismatched-dir-name"
    _write_skill(skill_dir, name="custom_skill_name")

    installer = SkillInstaller(tmp_path)
    result = await installer.install(str(skill_dir))

    assert result.target_root == tmp_path / ".msagent" / "skills" / "custom_skill_name"
    assert (result.target_root / "SKILL.md").exists()
    assert result.warnings


@pytest.mark.asyncio
async def test_skill_installer_surfaces_invalid_frontmatter_errors(tmp_path: Path) -> None:
    skill_dir = tmp_path / "broken-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: broken-skill\ndescription: [oops\n---\nbody\n",
        encoding="utf-8",
    )

    installer = SkillInstaller(tmp_path)

    with pytest.raises(SkillInstallError) as exc_info:
        await installer.install(str(skill_dir))

    assert "Invalid frontmatter" in str(exc_info.value)


@pytest.mark.asyncio
async def test_command_dispatcher_routes_add_skill_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _build_session(tmp_path)
    session.message_dispatcher = SimpleNamespace(dispatch=AsyncMock())
    dispatcher = CommandDispatcher(session)
    handle_mock = AsyncMock()

    monkeypatch.setattr(dispatcher.add_skill_handler, "handle", handle_mock)
    monkeypatch.setattr(add_skill_module.console, "print_error", lambda *_args: None)
    monkeypatch.setattr(add_skill_module.console, "print", lambda *_args, **_kwargs: None)

    await dispatcher.dispatch('/add-skill "/tmp/custom skill"')

    handle_mock.assert_awaited_once_with(["/tmp/custom skill"])
    assert "/add-skill" in dispatcher.commands
