import logging
import sys
import warnings
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, cast

from msagent.core.constants import CONFIG_LOG_DIR
from msagent.core.settings import settings

# File format includes full context (for log file)
FILE_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

_ERROR_LOGGERS = (
    "langchain_google_genai._function_utils",
    "langchain_anthropic",
    "langchain_openai",
)

_WARNING_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "langgraph.checkpoint",
    "aiosqlite",
    "markdown_it",
)


def _install_unraisablehook_filter() -> None:
    original_hook = sys.unraisablehook

    def _filtered_hook(unraisable: object) -> None:
        exc_value = getattr(unraisable, "exc_value", None)
        if isinstance(exc_value, ValueError) and "I/O operation on closed file" in str(exc_value):
            return
        cast(Any, original_hook)(unraisable)

    if getattr(sys.unraisablehook, "__name__", "") != "_filtered_hook":
        sys.unraisablehook = _filtered_hook


def configure_logging(show_logs: bool = False, working_dir: Path | None = None) -> None:
    """Configure application logging.

    Args:
        show_logs: Enable file logging and show log location hint.
        working_dir: Working directory for log file.
    """
    if working_dir is None:
        working_dir = Path.cwd()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Keep a no-op handler installed so Python's logging.lastResort
    # does not dump ERROR tracebacks to stderr when verbose logging is off.
    root_logger.addHandler(logging.NullHandler())

    for logger_name in _ERROR_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    for logger_name in _WARNING_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    # Suppress langchain_aws warnings
    warnings.filterwarnings("ignore", module="langchain_aws.chat_models.bedrock")

    # Suppress LangSmith UUID v7 deprecation warning
    warnings.filterwarnings(
        "ignore",
        message="LangSmith now uses UUID v7",
        category=UserWarning,
        module="pydantic.v1.main",
    )

    # Suppress GPT-2 tokenizer fallback warning for Ollama models
    warnings.filterwarnings(
        "ignore",
        message="Using fallback GPT-2 tokenizer",
        category=UserWarning,
    )

    # Suppress benign pydantic serializer noise when LangGraph/deepagents
    # passes runtime context objects through internal serialization paths.
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings:",
        category=UserWarning,
        module=r"pydantic(\.v1)?\.main",
    )

    if show_logs:
        # Enable file logging
        log_dir = working_dir / CONFIG_LOG_DIR
        log_file_path = log_dir / "app.log"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = TimedRotatingFileHandler(
                log_file_path,
                when="midnight",
                interval=1,
                backupCount=7,  # Keep 7 days of logs
                encoding="utf-8",
            )
            # Rotated files will be named app.log.YYYY-MM-DD
            file_handler.suffix = "%Y-%m-%d"
            file_handler.setFormatter(logging.Formatter(FILE_LOG_FORMAT))
            root_logger.addHandler(file_handler)

            # Print log file location hint (using plain print, not logging)
            abs_log_path = log_file_path.resolve()
            print(f"📝 Logs written to: {abs_log_path}", flush=True)
        except OSError:
            pass  # Sandbox may block file creation

    # Suppress urllib3 "I/O operation on closed file" during botocore GC cleanup.
    _install_unraisablehook_filter()


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        logging.Logger: Logger
    """
    return logging.getLogger(name)
