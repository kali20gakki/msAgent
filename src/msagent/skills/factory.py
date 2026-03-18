import asyncio
import re
from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_SKILL_CATEGORY = "default"


class Skill(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    category: str
    path: Path
    allowed_tools: list[str] | None = Field(default=None)

    def read_content(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except Exception:
            return ""

    @property
    def root_dir(self) -> Path:
        return self.path.parent

    @property
    def scripts_dir(self) -> Path:
        return self.root_dir / "scripts"

    def get_script_paths(self, limit: int = 20) -> list[Path]:
        try:
            if not self.scripts_dir.exists() or not self.scripts_dir.is_dir():
                return []
            files = [
                p
                for p in self.scripts_dir.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts
            ]
            files.sort()
            return files[:limit]
        except Exception:
            return []

    def get_script_relative_paths(self, limit: int = 20) -> list[str]:
        rel_paths: list[str] = []
        for p in self.get_script_paths(limit=limit):
            try:
                rel = p.relative_to(self.root_dir)
            except Exception:
                rel = p
            rel_paths.append(rel.as_posix())
        return rel_paths

    @property
    def display_name(self) -> str:
        if self.category == DEFAULT_SKILL_CATEGORY:
            return self.name
        return f"{self.category}/{self.name}"

    @classmethod
    async def from_file(cls, skill_md: Path, category: str) -> "Skill | None":
        try:
            content = await asyncio.to_thread(skill_md.read_text, encoding="utf-8")
            frontmatter_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
            if not frontmatter_match:
                return None

            frontmatter = yaml.safe_load(frontmatter_match.group(1))
            if not frontmatter or "name" not in frontmatter:
                return None

            return cls(
                name=frontmatter["name"],
                description=frontmatter.get("description", ""),
                category=category,
                path=skill_md,
                allowed_tools=frontmatter.get("allowed_tools"),
            )
        except Exception:
            return None


class SkillFactory:
    def __init__(self):
        self._skills: dict[str, dict[str, Skill]] = {}
        self._module_map: dict[str, str] = {}

    @staticmethod
    def get_default_skills_dir() -> Path:
        return Path(str(files("resources") / "skills"))

    @staticmethod
    def get_repo_skills_dir() -> Path:
        return SkillFactory.get_default_skills_dir()

    @staticmethod
    def get_packaged_skills_dir() -> Path:
        return SkillFactory.get_default_skills_dir()

    async def load_skills(
        self, skills_dir: Path | list[Path] | tuple[Path, ...]
    ) -> dict[str, dict[str, Skill]]:
        skill_dirs = (
            list(skills_dir)
            if isinstance(skills_dir, (list, tuple))
            else [skills_dir]
        )
        skill_dirs = [path for path in skill_dirs if await asyncio.to_thread(path.exists)]

        if not skill_dirs:
            self._skills = {}
            self._module_map = {}
            return {}

        skills: dict[str, dict[str, Skill]] = {}
        module_map: dict[str, str] = {}

        for root_dir in skill_dirs:
            entries = await asyncio.to_thread(lambda: list(root_dir.iterdir()))
            for entry in entries:
                if not entry.is_dir():
                    continue

                legacy_skill_md = entry / "SKILL.md"
                if await asyncio.to_thread(legacy_skill_md.exists):
                    await self._register_skill(
                        skills,
                        module_map,
                        legacy_skill_md,
                        DEFAULT_SKILL_CATEGORY,
                    )
                    continue

                category_name = entry.name
                category_entries = await asyncio.to_thread(lambda: list(entry.iterdir()))
                for skill_dir in category_entries:
                    if not skill_dir.is_dir():
                        continue

                    skill_md = skill_dir / "SKILL.md"
                    if not await asyncio.to_thread(skill_md.exists):
                        continue

                    await self._register_skill(
                        skills,
                        module_map,
                        skill_md,
                        category_name,
                    )

        self._skills = skills
        self._module_map = module_map
        return skills

    @staticmethod
    async def _register_skill(
        skills: dict[str, dict[str, Skill]],
        module_map: dict[str, str],
        skill_md: Path,
        category_name: str,
    ) -> None:
        skills.setdefault(category_name, {})
        skill = await Skill.from_file(skill_md, category_name)
        if not skill or skill.name in skills[category_name]:
            return

        skills[category_name][skill.name] = skill
        composite_key = f"{category_name}:{skill.name}"
        module_map[composite_key] = category_name

    def get_module_map(self) -> dict[str, str]:
        return self._module_map

    def get_all_skills(self) -> dict[str, dict[str, Skill]]:
        return self._skills

    def get_skill(self, category: str, name: str) -> Skill | None:
        return self._skills.get(category, {}).get(name)
