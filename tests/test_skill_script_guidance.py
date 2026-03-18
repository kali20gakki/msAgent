from pathlib import Path

from msagent.agents.factory import AgentFactory
from msagent.skills.factory import DEFAULT_SKILL_CATEGORY, Skill


def test_skill_discovers_scripts(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "analysis" / "demo-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\n", encoding="utf-8"
    )
    (scripts_dir / "run_demo.py").write_text("print('ok')\n", encoding="utf-8")

    skill = Skill(
        name="demo-skill",
        description="demo",
        category=DEFAULT_SKILL_CATEGORY,
        path=skill_dir / "SKILL.md",
    )

    assert skill.get_script_relative_paths() == ["scripts/run_demo.py"]


def test_build_skills_text_contains_script_workflow(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "analysis" / "demo-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\n", encoding="utf-8"
    )
    (scripts_dir / "run_demo.py").write_text("print('ok')\n", encoding="utf-8")

    skill = Skill(
        name="demo-skill",
        description="demo description",
        category="analysis",
        path=skill_dir / "SKILL.md",
    )

    text = AgentFactory._build_skills_text([skill], use_catalog=False)

    assert "Always call `get_skill(name, category)`" in text
    assert "prefer running those scripts" in text
    assert "`scripts/run_demo.py`" in text
