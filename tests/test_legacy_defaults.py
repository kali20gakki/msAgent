from pathlib import Path

import pytest

from msagent.cli.bootstrap.initializer import Initializer
from msagent.core.constants import CONFIG_SKILLS_DIR
from msagent.skills.factory import DEFAULT_SKILL_CATEGORY, SkillFactory


def test_legacy_system_prompt_is_preserved() -> None:
    prompt_path = Path("resources/configs/default/prompts/agents/Hermes.md")
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "Ascend NPU Profiling 性能分析助手" in prompt
    assert "msprof-mcp" in prompt
    assert "Todo 使用约束" in prompt
    assert "纯 subagent 内部短任务" in prompt
    assert "Subagent 使用约束" in prompt
    assert "执行与验证约束" in prompt
    assert "失败与调试约束" in prompt


def test_default_general_purpose_subagent_prompt_discourages_todo_management() -> None:
    prompt_path = Path("resources/configs/default/prompts/subagents/general-purpose.md")
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "Do not manage the user-facing todo list by default" in prompt
    assert "Do not spawn additional subagents unless explicitly instructed" in prompt
    assert "Stay within the delegated scope" in prompt


def test_default_explorer_subagent_prompt_has_scope_constraints() -> None:
    prompt_path = Path("resources/configs/default/prompts/subagents/explorer.md")
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "Stay within the delegated search scope" in prompt
    assert "Do not create or manage user-facing todos" in prompt
    assert "Do not spawn additional subagents unless explicitly instructed" in prompt


@pytest.mark.asyncio
async def test_skill_factory_loads_workspace_and_config_skills(tmp_path: Path) -> None:
    workspace_skills = tmp_path / "skills" / "analysis" / "workspace-skill"
    config_skills = tmp_path / ".msagent" / "skills" / "analysis" / "config-skill"
    workspace_skills.mkdir(parents=True)
    config_skills.mkdir(parents=True)

    skill_text = """---
name: {name}
description: test skill
---
content
"""
    (workspace_skills / "SKILL.md").write_text(
        skill_text.format(name="workspace-skill"), encoding="utf-8"
    )
    (config_skills / "SKILL.md").write_text(
        skill_text.format(name="config-skill"), encoding="utf-8"
    )

    skills = await SkillFactory().load_skills(
        [tmp_path / "skills", tmp_path / ".msagent" / "skills"]
    )

    assert "analysis" in skills
    assert "workspace-skill" in skills["analysis"]
    assert "config-skill" in skills["analysis"]


@pytest.mark.asyncio
async def test_skill_factory_loads_legacy_flat_skills(tmp_path: Path) -> None:
    legacy_skill_dir = tmp_path / "skills" / "op-mfu-calculator"
    legacy_skill_dir.mkdir(parents=True)
    (legacy_skill_dir / "SKILL.md").write_text(
        """---
name: op-mfu-calculator
description: test legacy skill
---
content
""",
        encoding="utf-8",
    )

    skills = await SkillFactory().load_skills(tmp_path / "skills")

    assert DEFAULT_SKILL_CATEGORY in skills
    assert "op-mfu-calculator" in skills[DEFAULT_SKILL_CATEGORY]
    assert (
        skills[DEFAULT_SKILL_CATEGORY]["op-mfu-calculator"].display_name
        == "op-mfu-calculator"
    )


def test_initializer_resolves_default_skill_search_order(tmp_path: Path) -> None:
    init = Initializer()
    default_skills = tmp_path / "resources" / "skills"

    init.skill_factory.get_default_skills_dir = lambda: default_skills

    skill_dirs = init._resolve_skills_dirs(tmp_path)

    assert skill_dirs == [
        tmp_path / "skills",
        default_skills,
        tmp_path / CONFIG_SKILLS_DIR,
    ]


def test_skill_factory_default_skills_dir_uses_resources_package() -> None:
    default_skills_dir = SkillFactory.get_default_skills_dir()

    assert default_skills_dir.name == "skills"
    assert "resources" in default_skills_dir.as_posix()
