import json
import re

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import ToolException

from msagent.agents.context import AgentContext


def _build_script_hints(skill_name: str, script_paths: list[str]) -> str:
    if not script_paths:
        return ""

    lines = []
    lines.append("")
    lines.append("## Script Execution Hints (Auto-generated)")
    lines.append(
        f"For skill `{skill_name}`, prefer executing scripts below before writing custom logic:"
    )
    for rel_path in script_paths:
        lines.append(f"- `{rel_path}`")
    lines.append("")
    lines.append("Command templates:")
    lines.append("- Python: `python3 <absolute_script_path> [args]`")
    lines.append("- Shell: `bash <absolute_script_path> [args]`")
    lines.append("- PowerShell: `pwsh -File <absolute_script_path> [args]`")
    lines.append(
        "If script output conflicts with assumptions, trust script output and adjust your plan."
    )
    return "\n".join(lines)


@tool
async def fetch_skills(
    runtime: ToolRuntime[AgentContext], pattern: str | None = None
) -> str:
    """Discover and search for available skills in the catalog.

    When users ask you to perform tasks, check if any of the available skills can help
    complete the task more effectively. Skills provide specialized capabilities and domain knowledge.

    Skills are specialized knowledge packages that provide workflows, domain expertise,
    and tool integrations. Use this to find skills relevant to your current task.

    WITHOUT pattern: Returns ALL available skills (use for browsing/exploring)
    WITH pattern: Returns ONLY matching skills (use when you know what you're looking for)

    The pattern searches skill names AND descriptions using case-insensitive regex.

    Args:
        pattern: Optional regex pattern to filter skills. Common patterns:
            - Simple keyword: "python", "code", "review"
            - Multiple keywords: "test|debug"
            - Category: "general|coding"

    Returns:
        JSON array of skill objects with: category, name, description.
        Returns empty array if no matches found.

    When to use:
        - Starting a task: fetch_skills("keyword") to find relevant skills
        - Exploring capabilities: fetch_skills() to see all available skills
        - Looking for domain expertise: fetch_skills("python") to find Python-related skills

    Example workflow:
    1. fetch_skills("code") - find code-related skills
    2. get_skill("python-best-practices", category="coding") - read the full skill content
    3. Apply the skill's guidance to your task

    Examples:
        fetch_skills() - list all available skills
        fetch_skills("python") - find Python-related skills
        fetch_skills("code|review") - find skills for coding or reviewing
    """
    skills = runtime.context.skill_catalog

    if not skills:
        return json.dumps([])

    if pattern is None:
        result = [
            {
                "display_name": skill.display_name,
                "category": skill.category,
                "name": skill.name,
                "description": skill.description,
            }
            for skill in skills
        ]
        return json.dumps(result, indent=2)

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ToolException(f"Invalid regex pattern: {e}") from e

    matches = []
    for skill in skills:
        if (
            regex.search(skill.name)
            or regex.search(skill.category)
            or regex.search(skill.description)
        ):
            matches.append(
                {
                    "display_name": skill.display_name,
                    "category": skill.category,
                    "name": skill.name,
                    "description": skill.description,
                }
            )

    return json.dumps(matches, indent=2)


fetch_skills.metadata = {"approval_config": {"always_approve": True}}


@tool
async def get_skill(
    name: str,
    runtime: ToolRuntime[AgentContext],
    category: str | None = None,
) -> str:
    """Read the full content of a specific skill.

    Use this after fetch_skills() when you've identified a skill you want to use.
    Returns the complete SKILL.md content including all instructions, workflows,
    and guidance.

    Args:
        name: Name of the skill (from fetch_skills() output)
        category: Optional category of the skill. Required only when multiple skills
            share the same name.

    Returns:
        Complete SKILL.md content with instructions and guidance
    """
    skills = runtime.context.skill_catalog

    if category:
        skill = next(
            (s for s in skills if s.category == category and s.name == name),
            None,
        )
    else:
        matches = [skill for skill in skills if skill.name == name]
        if len(matches) > 1:
            categories = ", ".join(sorted({skill.category for skill in matches}))
            raise ToolException(
                f"Multiple skills named '{name}' found. Specify category: {categories}"
            )
        skill = matches[0] if matches else None

    if not skill:
        skill_name = f"{category}/{name}" if category else name
        raise ToolException(f"Skill '{skill_name}' not found")

    content = skill.read_content()
    if not content:
        raise ToolException(f"Failed to read skill '{skill.display_name}'")

    hints = _build_script_hints(skill.display_name, skill.get_script_relative_paths())
    if hints:
        return f"{content}\n{hints}"
    return content


get_skill.metadata = {"approval_config": {"always_approve": True}}


SKILL_CATALOG_TOOLS = [fetch_skills, get_skill]
