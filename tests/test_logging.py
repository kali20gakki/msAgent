from __future__ import annotations

import logging
import warnings
from pathlib import Path

from msagent.core.logging import configure_logging, get_logger


def test_configure_logging_hides_tracebacks_without_verbose(
    tmp_path: Path,
    capsys,
) -> None:
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    original_level = root_logger.level

    try:
        configure_logging(show_logs=False, working_dir=tmp_path)

        assert any(
            isinstance(handler, logging.NullHandler)
            for handler in root_logger.handlers
        )

        logger = get_logger("msagent.tests.logging")
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logger.exception("hidden traceback")

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
    finally:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        for handler in original_handlers:
            root_logger.addHandler(handler)
        root_logger.setLevel(original_level)


def test_configure_logging_suppresses_pydantic_serializer_warning(tmp_path: Path) -> None:
    original_filters = warnings.filters[:]
    try:
        configure_logging(show_logs=False, working_dir=tmp_path)

        assert any(
            action == "ignore"
            and category is UserWarning
            and getattr(message, "pattern", "") == r"Pydantic serializer warnings:"
            and "pydantic" in getattr(module, "pattern", "")
            for action, message, category, module, _ in warnings.filters
        )
    finally:
        warnings.filters[:] = original_filters
