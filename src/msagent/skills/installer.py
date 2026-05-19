#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Install and validate user-provided skills."""

from __future__ import annotations

import asyncio
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from msagent.cli.bootstrap.initializer import initializer
from msagent.core.constants import CONFIG_SKILLS_DIR
from msagent.skills.factory import DEFAULT_SKILL_CATEGORY, SkillFactory

SKILL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
COPY_IGNORE_PATTERNS = shutil.ignore_patterns(".git", "__pycache__", ".DS_Store")


class SkillInstallError(ValueError):
    """Raised when a skill cannot be installed safely."""


@dataclass(frozen=True, slots=True)
class SkillInstallResult:
    """Summary of a successfully installed skill."""

    name: str
    description: str
    category: str
    pattern: str
    source_root: Path
    target_root: Path
    warnings: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        if self.category == DEFAULT_SKILL_CATEGORY:
            return self.name
        return f"{self.category}/{self.name}"


class SkillInstaller:
    """Validate and install custom skills into the local config directory."""

    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir
        self.skill_factory = SkillFactory()

    async def install(self, raw_path: str) -> SkillInstallResult:
        source_root, skill_file = self._resolve_skill_root(raw_path)
        content = await asyncio.to_thread(skill_file.read_text, encoding="utf-8")
        try:
            frontmatter = SkillFactory.parse_frontmatter(content)
        except ValueError as exc:
            raise SkillInstallError(str(exc)) from exc

        name = str(frontmatter.get("name") or "").strip()
        description = str(frontmatter.get("description") or "").strip()
        warnings: list[str] = []

        if not name:
            raise SkillInstallError("SKILL.md frontmatter must include a non-empty 'name'")
        if not description:
            raise SkillInstallError("SKILL.md frontmatter must include a non-empty 'description'")
        if not SKILL_NAME_PATTERN.fullmatch(name):
            raise SkillInstallError("Skill name may only contain letters, numbers, '-' and '_'")
        if source_root.name != name:
            warnings.append(
                f"Directory name '{source_root.name}' does not match skill name '{name}'. "
                "The skill will still be installed."
            )

        category = DEFAULT_SKILL_CATEGORY
        pattern = f"{category}:{name}"
        target_dir_name = self._build_target_dir_name(name)
        target_root = self.working_dir / CONFIG_SKILLS_DIR / target_dir_name
        target_skill_file = target_root / "SKILL.md"

        if target_root.exists():
            raise SkillInstallError(f"Target path already exists: {target_root.as_posix()}")

        await self._ensure_not_shadowed(pattern)

        target_root.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copytree, source_root, target_root, ignore=COPY_IGNORE_PATTERNS)

        try:
            installed_skills = await self.skill_factory.load_skills(target_root.parent)
            installed = installed_skills.get(category, {}).get(name)
            if installed is None or installed.path.resolve() != target_skill_file.resolve():
                raise SkillInstallError("Installed skill could not be loaded from the destination directory")
        except Exception:
            await asyncio.to_thread(shutil.rmtree, target_root, ignore_errors=True)
            raise

        return SkillInstallResult(
            name=name,
            description=description,
            category=category,
            pattern=pattern,
            source_root=source_root,
            target_root=target_root,
            warnings=warnings,
        )

    def _resolve_skill_root(self, raw_path: str) -> tuple[Path, Path]:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = (self.working_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if not candidate.exists():
            raise SkillInstallError(f"Path does not exist: {candidate.as_posix()}")

        if candidate.is_file():
            if candidate.name != "SKILL.md":
                raise SkillInstallError("File path must point to SKILL.md")
            return candidate.parent, candidate

        direct_skill = candidate / "SKILL.md"
        if direct_skill.is_file():
            return candidate, direct_skill

        nested_skill_files = sorted(candidate.rglob("SKILL.md"))
        if not nested_skill_files:
            raise SkillInstallError("No SKILL.md found under the provided path")
        if len(nested_skill_files) > 1:
            raise SkillInstallError("Multiple SKILL.md files found; please provide a single skill path")
        return nested_skill_files[0].parent, nested_skill_files[0]

    async def _ensure_not_shadowed(self, pattern: str) -> None:
        skills_dirs = initializer.resolve_skills_dirs(self.working_dir)
        config_dir = self.working_dir / CONFIG_SKILLS_DIR

        shadowing_dirs: list[Path] = []
        for skills_dir in skills_dirs:
            if skills_dir.resolve() == config_dir.resolve():
                break
            shadowing_dirs.append(skills_dir)

        if not shadowing_dirs:
            return

        existing = await self.skill_factory.load_skills(shadowing_dirs)
        category, name = pattern.split(":", 1)
        shadowed_skill = existing.get(category, {}).get(name)
        if shadowed_skill is not None:
            raise SkillInstallError(
                f"Skill '{pattern}' is already provided by a higher-priority directory: "
                f"{shadowed_skill.root_dir.as_posix()}"
            )

    @staticmethod
    def _build_target_dir_name(name: str) -> str:
        if not SKILL_NAME_PATTERN.fullmatch(name):
            raise SkillInstallError("Skill name may only contain letters, numbers, '-' and '_'")
        return name
