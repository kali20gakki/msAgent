"""Custom skill installation command handler."""

from __future__ import annotations

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console
from msagent.core.logging import get_logger
from msagent.skills.installer import SkillInstallError, SkillInstaller

logger = get_logger(__name__)


class AddSkillHandler:
    """Install a custom skill into the local config directory."""

    def __init__(self, session) -> None:
        self.session = session

    async def handle(self, args: list[str]) -> None:
        if not args:
            console.print_error("Usage: /add-skill <path-to-skill-or-SKILL.md>")
            console.print("")
            return

        try:
            current_agent = await initializer.load_agent_config(
                self.session.context.agent,
                self.session.context.working_dir,
            )
            installer = SkillInstaller(self.session.context.working_dir)
            result = await installer.install(args[0])
            await initializer.add_agent_skill_pattern(
                current_agent.name,
                result.pattern,
                self.session.context.working_dir,
            )

            for warning in result.warnings:
                console.print(f"[yellow]Warning:[/yellow] {warning}")

            console.print_success(
                f"Installed skill '{result.display_name}' to {result.target_root.as_posix()} "
                f"and enabled it for agent '{current_agent.name}'."
            )
            console.print("")

            self.session.needs_reload = True
            self.session.running = False
        except SkillInstallError as exc:
            console.print_error(str(exc))
            console.print("")
        except Exception as exc:
            console.print_error(f"Error installing skill: {exc}")
            console.print("")
            logger.debug("Add skill error", exc_info=True)
