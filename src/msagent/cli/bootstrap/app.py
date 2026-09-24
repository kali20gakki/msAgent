# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Main CLI application entry point."""

import asyncio
import sys
from pathlib import Path

from msagent.cli.bootstrap.legacy import (
    DEFAULT_SESSION_COMMAND,
    create_legacy_parser,
    create_session_parser,
    dispatch_legacy_command,
    normalize_argv,
    render_config_help,
    render_root_help,
    render_version_info,
)
from msagent.cli.theme import console
from msagent.core.logging import configure_logging, get_logger
from msagent.core.paths import AppPaths
from msagent.core.storage_layout import StorageLayoutError, validate_and_initialize_storage_layout


def create_parser():
    """Create the compatibility CLI parser."""
    return create_legacy_parser()


async def main() -> int:
    """Main CLI entry point."""
    raw_argv = sys.argv[1:]
    if raw_argv in (["--help"], ["-h"]):
        render_root_help()
        return 0
    if raw_argv in (["config", "--help"], ["config", "-h"]):
        render_config_help()
        return 0
    if raw_argv in (["--version"], ["-V"]):
        render_version_info()
        return 0

    argv = normalize_argv(raw_argv)
    if argv and argv[0] == DEFAULT_SESSION_COMMAND:
        parser = create_session_parser()
        args = parser.parse_args(argv[1:])
    else:
        parser = create_parser()
        args = parser.parse_args(argv)

    working_dir = Path(getattr(args, "working_dir", Path.cwd()))
    app_paths = AppPaths.resolve()
    try:
        validate_and_initialize_storage_layout(app_paths)
    except StorageLayoutError as exc:
        console.print_error(str(exc))
        return 1

    configure_logging(
        show_logs=getattr(args, "verbose", False),
        working_dir=working_dir,
        log_dir=app_paths.logs_dir,
    )
    logger = get_logger(__name__)

    try:
        return await dispatch_legacy_command(args)
    except Exception as e:
        console.print_error(f"Unexpected error: {e}")
        console.print("")
        logger.exception("CLI error")
        return 1


def cli():
    """Synchronous CLI entry point for setuptools."""
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    cli()
