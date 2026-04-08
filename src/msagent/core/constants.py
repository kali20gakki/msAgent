import importlib.metadata
import platform
from pathlib import Path

UNKNOWN = "unknown"


def _detect_package_version() -> str:
    for package_name in ("mindstudio-agent", "msagent"):
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            continue

    project_root = Path(__file__).resolve().parents[3]
    pyproject_path = project_root / "pyproject.toml"

    try:
        import tomllib

        with pyproject_path.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:
        return UNKNOWN


APP_NAME = "msagent"
APP_VERSION = _detect_package_version()
CONFIG_DIR_NAME = f".{APP_NAME}"
CONFIG_MCP_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/config.mcp.json")
CONFIG_MCP_FILE = "config.mcp.json"  # Short name for compatibility
CONFIG_APPROVAL_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/config.approval.json")
CONFIG_LANGGRAPH_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/langgraph.json")
CONFIG_LLMS_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/config.llms.yml")
CONFIG_CHECKPOINTERS_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/config.checkpointers.yml")
CONFIG_AGENTS_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/config.agents.yml")
CONFIG_SUBAGENTS_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/config.subagents.yml")
CONFIG_CHECKPOINTS_URL_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/config.checkpoints.db")
CONFIG_HISTORY_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/.history")
CONFIG_MEMORY_FILE_NAME = Path(f"{CONFIG_DIR_NAME}/memory.md")
CONFIG_MEMORY_FILE = "memory.md"  # Short name for compatibility

CONFIG_LLMS_DIR = Path(f"{CONFIG_DIR_NAME}/llms")
CONFIG_CHECKPOINTERS_DIR = Path(f"{CONFIG_DIR_NAME}/checkpointers")
CONFIG_AGENTS_DIR = Path(f"{CONFIG_DIR_NAME}/agents")
CONFIG_SUBAGENTS_DIR = Path(f"{CONFIG_DIR_NAME}/subagents")
CONFIG_SKILLS_DIR = Path(f"{CONFIG_DIR_NAME}/skills")
CONFIG_SANDBOXES_DIR = Path(f"{CONFIG_DIR_NAME}/sandboxes")
CONFIG_MCP_CACHE_DIR = Path(f"{CONFIG_DIR_NAME}/cache/mcp")
CONFIG_SANDBOX_CACHE_DIR = Path(f"{CONFIG_DIR_NAME}/cache/sandboxes")
CONFIG_LOG_DIR = Path(f"{CONFIG_DIR_NAME}/logs")
CONFIG_MCP_OAUTH_DIR = Path(f"{CONFIG_DIR_NAME}/oauth/mcp")
CONFIG_CONVERSATION_HISTORY_DIR = Path(f"{CONFIG_DIR_NAME}/conversation_history")

DEFAULT_CONFIG_DIR_NAME = "resources.configs.default"
DEFAULT_CONFIG_DIR = (
    Path(__file__).parent.parent.parent / "resources" / "configs" / "default"
)

TOOL_CATEGORY_IMPL = "impl"
TOOL_CATEGORY_MCP = "mcp"
TOOL_CATEGORY_INTERNAL = "internal"

# Keep config schema versions aligned with the packaged msagent version so
# default resources and runtime migrations follow a single source of truth.
CONFIG_VERSION_TOKEN = "__APP_VERSION__"
AGENT_CONFIG_VERSION = APP_VERSION
LLM_CONFIG_VERSION = APP_VERSION
CHECKPOINTER_CONFIG_VERSION = APP_VERSION
SANDBOX_CONFIG_VERSION = APP_VERSION
MCP_CACHE_VERSION = "1.0.0"

PLATFORM = platform.system()
OS_VERSION = platform.version()
