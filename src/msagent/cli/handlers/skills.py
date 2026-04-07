from __future__ import annotations

import shutil
import textwrap
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import FormattedTextControl

from msagent.cli.theme import console, theme
from msagent.cli.ui.shared import (
    SelectorState,
    create_instruction,
    create_selector_application,
)
from msagent.core.logging import get_logger
from msagent.core.settings import settings

if TYPE_CHECKING:
    from msagent.skills.factory import Skill

logger = get_logger(__name__)


class SkillsHandler:
    def __init__(self, session) -> None:
        self.session = session

    async def handle(
        self, skills: list[Skill], args: list[str] | None = None
    ) -> None:
        try:
            if not skills:
                console.print_error("No skills available")
                console.print("")
                return

            sorted_skills = self._sort_skills(skills)

            if args:
                await self._handle_command_args(sorted_skills, args)
                return

            selected_skill = await self._get_skill_selection(sorted_skills)
            if selected_skill:
                self._queue_skill_for_next_prompt(selected_skill, sorted_skills)

        except Exception as e:
            console.print_error(f"Error displaying skills: {e}")
            console.print("")
            logger.debug("Skill display error", exc_info=True)

    async def _handle_command_args(self, skills: list[Skill], args: list[str]) -> None:
        skill = self._resolve_skill(skills, args[0])
        task = " ".join(args[1:]).strip()

        if task:
            await self._run_skill(skill, skills, task)
            return

        self._queue_skill_for_next_prompt(skill, skills)

    async def handle_shortcut(
        self,
        skills: list[Skill],
        skill_ref: str,
        args: list[str] | None = None,
        *,
        raw_input: str | None = None,
    ) -> bool:
        skill = self._try_resolve_skill(skills, skill_ref)
        if skill is None:
            return False

        task = " ".join(args or []).strip()
        if task:
            await self._run_skill(skill, skills, task, raw_input=raw_input)
            return True

        self._queue_skill_for_next_prompt(skill, skills)
        return True

    async def _get_skill_selection(self, skills: list[Skill]) -> Skill | None:
        state = SelectorState(window_size=10)
        expanded_indices: set[int] = set()

        text_control = FormattedTextControl(
            text=lambda: self._format_skill_list(
                skills,
                state.index,
                expanded_indices,
                state.scroll_offset,
                state.window_size or 10,
            ),
            focusable=True,
            show_cursor=False,
        )

        kb = KeyBindings()

        @kb.add(Keys.Up)
        def _(_event):
            state.move_linear(-1, size=len(skills))

        @kb.add(Keys.Down)
        def _(_event):
            state.move_linear(1, size=len(skills))

        @kb.add(Keys.Tab)
        def _(_event):
            if state.index in expanded_indices:
                expanded_indices.remove(state.index)
            else:
                expanded_indices.add(state.index)

        selected = [False]

        @kb.add(Keys.Enter)
        def _(event):
            selected[0] = True
            event.app.exit()

        @kb.add(Keys.ControlC)
        def _(event):
            event.app.exit()

        context = self.session.context
        app = create_selector_application(
            context=context,
            text_control=text_control,
            key_bindings=kb,
            header_windows=create_instruction(
                "Enter: select skill | Tab: expand/collapse"
            ),
        )

        try:
            await app.run_async()
            if selected[0]:
                return skills[state.index]
        except (KeyboardInterrupt, EOFError):
            pass

        return None

    @staticmethod
    def _format_skill_list(
        skills: list[Skill],
        selected_index: int,
        expanded_indices: set[int],
        scroll_offset: int,
        window_size: int,
    ) -> FormattedText:
        prompt_symbol = settings.cli.prompt_style.strip()
        lines: list[tuple[str, str]] = []

        visible_skills = skills[scroll_offset : scroll_offset + window_size]
        wrap_width = max(20, shutil.get_terminal_size((80, 24)).columns - 6)

        for idx, skill in enumerate(visible_skills):
            absolute_index = scroll_offset + idx
            name = skill.display_name
            description = skill.description or "No description"
            preview = SkillsHandler._build_description_preview(description, wrap_width)

            if absolute_index == selected_index:
                lines.append((theme.selection_color, f"{prompt_symbol} {name}"))
            else:
                lines.append(("", f"  {name}"))

            lines.append(("", "\n"))
            lines.append(("dim", f"    {preview}"))

            if absolute_index in expanded_indices:
                lines.append(("", "\n"))
                lines.append(("muted", f"    Path: {skill.root_dir.as_posix()}"))
                for wrapped_line in SkillsHandler._wrap_description(
                    description, wrap_width
                ):
                    lines.append(("", "\n"))
                    lines.append(("dim", f"    {wrapped_line}"))

            if idx < len(visible_skills) - 1:
                lines.append(("", "\n\n"))

        return FormattedText(lines)

    @staticmethod
    def _sort_skills(skills: list[Skill]) -> list[Skill]:
        return sorted(skills, key=lambda skill: skill.display_name.casefold())

    @staticmethod
    def _normalize_skill_ref(skill_ref: str) -> str:
        return skill_ref.strip().casefold()

    @classmethod
    def _resolve_skill(cls, skills: list[Skill], skill_ref: str) -> Skill:
        normalized_ref = cls._normalize_skill_ref(skill_ref)
        if not normalized_ref:
            raise ValueError("Skill name is required")

        display_matches = [
            skill
            for skill in skills
            if cls._normalize_skill_ref(skill.display_name) == normalized_ref
        ]
        if len(display_matches) == 1:
            return display_matches[0]

        category_matches = [
            skill
            for skill in skills
            if cls._normalize_skill_ref(f"{skill.category}:{skill.name}") == normalized_ref
            or cls._normalize_skill_ref(f"{skill.category}/{skill.name}") == normalized_ref
        ]
        if len(category_matches) == 1:
            return category_matches[0]

        name_matches = [
            skill
            for skill in skills
            if cls._normalize_skill_ref(skill.name) == normalized_ref
        ]
        if len(name_matches) == 1:
            return name_matches[0]
        if len(name_matches) > 1:
            skill_options = ", ".join(
                sorted(skill.display_name for skill in name_matches)
            )
            raise ValueError(
                f"Multiple skills named '{skill_ref}' found. Use one of: "
                f"{skill_options}"
            )

        raise ValueError(f"Skill '{skill_ref}' not found")

    @classmethod
    def _try_resolve_skill(cls, skills: list[Skill], skill_ref: str) -> Skill | None:
        try:
            return cls._resolve_skill(skills, skill_ref)
        except ValueError as exc:
            if str(exc) == f"Skill '{skill_ref}' not found":
                return None
            raise

    @staticmethod
    def _build_description_preview(description: str, width: int) -> str:
        flattened = " ".join(
            line.strip() for line in description.splitlines() if line.strip()
        )
        if not flattened:
            return "No description"
        return textwrap.shorten(flattened, width=width, placeholder="...")

    @staticmethod
    def _wrap_description(description: str, width: int) -> list[str]:
        wrapped_lines: list[str] = []
        for raw_line in description.splitlines() or [""]:
            stripped = raw_line.strip()
            if not stripped:
                if wrapped_lines and wrapped_lines[-1] != "":
                    wrapped_lines.append("")
                continue
            wrapped_lines.extend(
                textwrap.wrap(
                    stripped,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
                or [stripped]
            )
        return wrapped_lines or ["No description"]

    @staticmethod
    def _build_skill_task_prompt(skill: Skill, task: str | None = None) -> str:
        task_suffix = task.strip() if task else ""
        return (
            f"Use the skill `{skill.display_name}` for the next task.\n\n"
            f"Before you apply it, call `get_skill(name=\"{skill.name}\", "
            f"category=\"{skill.category}\")` to read the full instructions and "
            "follow that skill closely.\n\n"
            f"Task:\n{task_suffix}"
        )

    @classmethod
    def _build_shortcut_name(cls, skill: Skill, skills: list[Skill]) -> str:
        same_name_count = sum(
            1
            for other in skills
            if cls._normalize_skill_ref(other.name) == cls._normalize_skill_ref(skill.name)
        )
        if same_name_count == 1:
            return skill.name
        return f"{skill.category}:{skill.name}"

    @classmethod
    def _build_shortcut_input(
        cls,
        skill: Skill,
        skills: list[Skill],
        task: str | None = None,
    ) -> str:
        shortcut = f"/{cls._build_shortcut_name(skill, skills)}"
        if task and task.strip():
            return f"{shortcut} {task.strip()}"
        return f"{shortcut} "

    def _queue_skill_for_next_prompt(self, skill: Skill, skills: list[Skill]) -> None:
        existing_task = (self.session.prefilled_text or "").strip()
        self.session.prefilled_text = self._build_shortcut_input(
            skill,
            skills,
            task=existing_task or None,
        )
        console.print_success(
            f"Selected skill '{skill.display_name}'. It will be used in the next "
            "prompt."
        )
        console.print("")

    async def _run_skill(
        self,
        skill: Skill,
        skills: list[Skill],
        task: str,
        *,
        raw_input: str | None = None,
    ) -> None:
        prompt = self._build_skill_task_prompt(skill, task)
        short_content = raw_input or self._build_shortcut_input(skill, skills, task).strip()
        self.session.renderer.render_user_message(
            HumanMessage(content=prompt, short_content=short_content)
        )
        await self.session.message_dispatcher.dispatch(prompt)
