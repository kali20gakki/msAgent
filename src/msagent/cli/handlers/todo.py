"""Todo handler for displaying current task list."""

from langchain_core.runnables import RunnableConfig

from msagent.cli.bootstrap.initializer import initializer
from msagent.cli.theme import console
from msagent.core.logging import get_logger
from msagent.tools.internal.todo import render_todos_panel

logger = get_logger(__name__)


class TodoHandler:
    """Handles todo list display."""

    def __init__(self, session) -> None:
        """Initialize with reference to CLI session."""
        self.session = session

    async def handle(self, max_items: int = 10) -> None:
        """Show current todo list with box border.
        
        参考 langchain-code 的设计，使用 Panel + box.ROUNDED 创建带圆角的边框效果，
        已完成的任务使用删除线(strike)划掉。
        """
        try:
            async with initializer.get_checkpointer(
                self.session.context.agent, self.session.context.working_dir
            ) as checkpointer:
                config = RunnableConfig(
                    configurable={"thread_id": self.session.context.thread_id}
                )
                latest_checkpoint = await checkpointer.aget_tuple(config)

                if not latest_checkpoint or not latest_checkpoint.checkpoint:
                    console.print_error("No todos currently")
                    console.print("")
                    return

                channel_values = latest_checkpoint.checkpoint.get("channel_values", {})
                todos = channel_values.get("todos")

                if not todos:
                    console.print_error("No todos currently")
                    console.print("")
                    return

                # 使用 Panel 显示带边框的 todo 列表
                panel = render_todos_panel(
                    todos,
                    max_items=max_items,
                    max_completed=max_items,
                    show_completed_indicator=False,
                )
                console.print("  ", end="")
                console.console.print(panel)
                console.print("")

        except Exception as e:
            console.print_error(f"Error displaying todos: {e}")
            console.print("")
            logger.debug("Todo display error", exc_info=True)
