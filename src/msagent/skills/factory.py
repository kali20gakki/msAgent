"""Skills factory for loading SKILL.md metadata from configured directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from msagent.core.constants import DEFAULT_CONFIG_DIR

DEFAULT_SKILL_CATEGORY = "default"


@dataclass(frozen=True, slots=True)
class Skill:
    """Serializable view of a skill used by CLI and catalog tools."""

    name: str
    description: str
    category: str
    path: Path

    @property
    def root_dir(self) -> Path:
        return self.path.parent

    @property
    def display_name(self) -> str:
        if self.category == DEFAULT_SKILL_CATEGORY:
            return self.name
        return f"{self.category}/{self.name}"

    def get_script_relative_paths(self, limit: int = 8) -> list[str]:
        scripts_dir = self.root_dir / "scripts"
        if not scripts_dir.exists():
            return []
        return [
            str(path.relative_to(self.root_dir)).replace("\\", "/")
            for path in sorted(scripts_dir.rglob("*"))
            if path.is_file()
        ][:limit]


class SkillFactory:
    """Loads skills from one or more directories."""

    def __init__(self) -> None:
        self._module_map: dict[str, str] = {}

    async def load_skills(
        self,
        skills_dir: Path | list[Path],
    ) -> dict[str, dict[str, Skill]]:
        directories = [skills_dir] if isinstance(skills_dir, Path) else list(skills_dir)

        loaded: dict[str, dict[str, Skill]] = {}
        self._module_map.clear()

        for directory in directories:
            if not directory.exists():
                continue

            for skill_file in sorted(directory.rglob("SKILL.md")):
                skill = self._load_skill_file(skill_file, base_dir=directory)
                if skill is None:
                    continue

                loaded.setdefault(skill.category, {})
                # First one wins to avoid unstable duplicate ordering.
                loaded[skill.category].setdefault(skill.name, skill)
                self._module_map[f"{skill.category}:{skill.name}"] = skill.category

        return loaded

    def get_module_map(self) -> dict[str, str]:
        return dict(self._module_map)

    @staticmethod
    def get_default_skills_dir() -> Path:
        return DEFAULT_CONFIG_DIR / "skills"

    @staticmethod
    def _load_skill_file(skill_file: Path, *, base_dir: Path) -> Skill | None:
        try:
            content = skill_file.read_text(encoding="utf-8")
        except Exception:
            return None

        frontmatter: dict[str, object] = {}
        body = content.lstrip()
        if body.startswith("---"):
            parts = body.split("---", 2)
            if len(parts) >= 3:
                parsed = yaml.safe_load(parts[1]) or {}
                if isinstance(parsed, dict):
                    frontmatter = parsed

        name = str(frontmatter.get("name") or skill_file.parent.name).strip()
        description = str(frontmatter.get("description") or "").strip()
        if not name:
            return None

        parent = skill_file.parent.parent
        category = (
            parent.name
            if parent != base_dir and parent.exists()
            else DEFAULT_SKILL_CATEGORY
        )

        return Skill(
            name=name,
            description=description,
            category=category,
            path=skill_file,
        )
