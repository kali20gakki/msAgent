"""Catalog tools for browsing and reading loaded skills."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field

from msagent.skills.factory import DEFAULT_SKILL_CATEGORY, SkillFactory


def _context_value(context: Any, key: str) -> Any:
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def _runtime_skill_catalog(runtime: Any) -> list[Any]:
    context = getattr(runtime, "context", None)
    if context is None:
        return []
    return list(_context_value(context, "skill_catalog") or [])


def _initializer_skill_catalog() -> list[Any]:
    try:
        from msagent.cli.bootstrap.initializer import initializer
    except Exception:
        return []
    return list(getattr(initializer, "cached_agent_skills", []) or [])


async def _fallback_skill_catalog() -> list[Any]:
    cached = _initializer_skill_catalog()
    if cached:
        return cached

    skill_factory = SkillFactory()
    default_dir = skill_factory.get_default_skills_dir()
    working_dir = Path.cwd()
    candidates = [
        working_dir / "skills",
        default_dir,
        working_dir / ".msagent" / "skills",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    loaded = await skill_factory.load_skills(unique)
    return [
        skill
        for category in loaded.values()
        for skill in category.values()
    ]


class FetchSkillsInput(BaseModel):
    pattern: str = Field(default=".*", description="Regex used to filter skills")


@tool("fetch_skills", args_schema=FetchSkillsInput)
async def fetch_skills(*, pattern: str = ".*", runtime: Any = None) -> str:
    """List loaded skills in JSON format."""
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ToolException(f"Invalid regex pattern: {exc}") from exc

    catalog = _runtime_skill_catalog(runtime)
    if not catalog:
        catalog = await _fallback_skill_catalog()

    payload: list[dict[str, str]] = []
    for skill in catalog:
        category = getattr(skill, "category", DEFAULT_SKILL_CATEGORY)
        name = getattr(skill, "name", "")
        description = getattr(skill, "description", "")
        display_name = (
            name if category == DEFAULT_SKILL_CATEGORY else f"{category}/{name}"
        )

        if compiled.search(f"{display_name}\n{description}"):
            payload.append(
                {
                    "display_name": display_name,
                    "category": category,
                    "name": name,
                    "description": description,
                }
            )

    return json.dumps(payload, ensure_ascii=False)


class GetSkillInput(BaseModel):
    name: str = Field(description="Skill name")
    category: str | None = Field(default=None, description="Optional skill category")


@tool("get_skill", args_schema=GetSkillInput)
async def get_skill(*, name: str, category: str | None = None, runtime: Any = None) -> str:
    """Return SKILL.md content for a selected skill."""
    catalog = _runtime_skill_catalog(runtime)
    if not catalog:
        catalog = await _fallback_skill_catalog()

    skills = [
        skill
        for skill in catalog
        if getattr(skill, "name", None) == name
    ]
    if not skills:
        raise ToolException(f"Skill '{name}' not found")

    if category is not None:
        skills = [skill for skill in skills if getattr(skill, "category", None) == category]
        if not skills:
            raise ToolException(
                f"Skill '{name}' not found in category '{category}'"
            )
    elif len(skills) > 1:
        categories = ", ".join(sorted({getattr(skill, "category", "") for skill in skills}))
        raise ToolException(f"Multiple skills named '{name}' found. Specify category: {categories}")

    selected = skills[0]
    path = getattr(selected, "path", None)
    if path is None:
        raise ToolException(f"Skill '{name}' has no valid path")

    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        raise ToolException(f"Failed to read skill '{name}': {exc}") from exc

