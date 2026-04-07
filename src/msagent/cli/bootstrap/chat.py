from pathlib import Path

from msagent.cli.bootstrap.timer import enable_timer
from msagent.cli.core.context import Context
from msagent.cli.core.session import Session
from msagent.cli.theme import console


async def handle_chat_command(args) -> int:
    """Handle the chat command."""
    try:
        if args.timer:
            enable_timer()

        context = await Context.create(
            agent=args.agent,
            model=args.model,
            working_dir=Path(args.working_dir),
            approval_mode=args.approval_mode,
            stream_output=getattr(args, "stream", True),
        )

        session = Session(context)

        # One-shot mode
        if args.message:
            return await session.send(args.message)

        # Interactive mode
        first_start = True
        while True:
            await session.start(
                show_welcome=first_start,
            )
            first_start = False

            if session.needs_reload:
                session = Session(context)
                continue
            else:
                break

        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        console.print_error(f"Error starting chat session: {e}")
        console.print("")
        return 1
