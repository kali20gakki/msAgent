"""Helpers for running the official deep-agents-ui frontend."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from msagent.core.constants import APP_NAME

DEEP_AGENTS_UI_REPO = "https://github.com/langchain-ai/deep-agents-ui"
DEEP_AGENTS_UI_DIRNAME = "deep-agents-ui"
DEFAULT_UI_PORT = 3000
DEFAULT_CONFIG_MARKER = "msagent-default-config"
BRANDING_MARKER = "msagent-branding"
ENV_UI_DEPLOYMENT_URL = "NEXT_PUBLIC_MSAGENT_DEPLOYMENT_URL"
ENV_UI_ASSISTANT_ID = "NEXT_PUBLIC_MSAGENT_ASSISTANT_ID"
UI_PAGE_PATH = Path("src") / "app" / "page.tsx"


def cache_root() -> Path:
    """Return the persistent cache root used for web assets."""
    xdg_cache = os.getenv("XDG_CACHE_HOME", "").strip()
    if xdg_cache:
        return Path(xdg_cache) / APP_NAME
    return Path.home() / ".cache" / APP_NAME


def ui_checkout_dir() -> Path:
    """Return the directory used to cache the official UI checkout."""
    return cache_root() / "web" / DEEP_AGENTS_UI_DIRNAME


def _resolve_tool_command(command: str) -> str:
    """Resolve a required external command to an explicit executable path."""
    resolved = shutil.which(command)
    if resolved:
        return resolved
    raise RuntimeError(
        f"{command} is required to run the official deep-agents-ui frontend"
    )


def build_ui_dev_command(*, host: str, port: int) -> list[str]:
    """Build the command used to launch deep-agents-ui in dev mode."""
    return [
        _resolve_tool_command("npx"),
        "next",
        "dev",
        "--turbopack",
        "--hostname",
        host,
        "--port",
        str(port),
    ]


def ensure_node_available() -> None:
    """Ensure Node.js package manager commands are available for the UI."""
    _resolve_tool_command("npm")
    _resolve_tool_command("npx")


def ensure_ui_checkout() -> Path:
    """Clone the official deep-agents-ui repo into the local cache if missing."""
    ensure_node_available()

    checkout_dir = ui_checkout_dir()
    package_json = checkout_dir / "package.json"
    if package_json.exists():
        return checkout_dir

    if checkout_dir.exists():
        shutil.rmtree(checkout_dir)
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", DEEP_AGENTS_UI_REPO, str(checkout_dir)],
        check=True,
    )
    return checkout_dir


def ensure_ui_dependencies(checkout_dir: Path) -> None:
    """Install UI dependencies on first use."""
    node_modules = checkout_dir / "node_modules"
    if node_modules.exists():
        return

    env = dict(os.environ)
    env.setdefault("NEXT_TELEMETRY_DISABLED", "1")
    subprocess.run(
        [
            _resolve_tool_command("npm"),
            "install",
            "--no-fund",
            "--no-audit",
            "--legacy-peer-deps",
        ],
        cwd=str(checkout_dir),
        env=env,
        check=True,
    )


def ensure_ui_customizations(checkout_dir: Path) -> None:
    """Apply all msAgent-specific customizations to the cached UI checkout."""
    _update_ui_page(checkout_dir, _apply_ui_customizations)


def ensure_ui_default_config_support(checkout_dir: Path) -> None:
    """Patch the cached official UI so msAgent can prefill deployment defaults."""
    _update_ui_page(checkout_dir, _apply_default_config_support)


def ensure_ui_branding(checkout_dir: Path) -> None:
    """Patch visible UI copy so the cached official UI is branded as msAgent."""
    _update_ui_page(checkout_dir, _apply_ui_branding)


def _apply_ui_customizations(content: str) -> str:
    """Apply branding and default-config patches in a single pass."""
    return _apply_ui_branding(_apply_default_config_support(content))


def _apply_default_config_support(content: str) -> str:
    """Return page content with msAgent default-config support injected."""
    if DEFAULT_CONFIG_MARKER in content:
        return content

    anchor = '  const [assistantId, setAssistantId] = useQueryState("assistantId");\n'
    defaults_block = """  const [assistantId, setAssistantId] = useQueryState("assistantId");
  // msagent-default-config
  const defaultConfig: StandaloneConfig | null =
    process.env.NEXT_PUBLIC_MSAGENT_DEPLOYMENT_URL &&
    process.env.NEXT_PUBLIC_MSAGENT_ASSISTANT_ID
      ? {
          deploymentUrl: process.env.NEXT_PUBLIC_MSAGENT_DEPLOYMENT_URL,
          assistantId: process.env.NEXT_PUBLIC_MSAGENT_ASSISTANT_ID,
          langsmithApiKey:
            process.env.NEXT_PUBLIC_LANGSMITH_API_KEY || undefined,
        }
      : null;
"""
    if anchor not in content:
        raise RuntimeError("Failed to patch deep-agents-ui: assistantId anchor not found")
    content = content.replace(anchor, defaults_block, 1)

    old_effect = """  useEffect(() => {
    const savedConfig = getConfig();
    if (savedConfig) {
      setConfig(savedConfig);
      if (!assistantId) {
        setAssistantId(savedConfig.assistantId);
      }
    } else {
      setConfigDialogOpen(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
"""
    new_effect = """  useEffect(() => {
    const savedConfig = getConfig();
    if (savedConfig) {
      setConfig(savedConfig);
      if (!assistantId) {
        setAssistantId(savedConfig.assistantId);
      }
    } else if (defaultConfig) {
      saveConfig(defaultConfig);
      setConfig(defaultConfig);
      if (!assistantId) {
        setAssistantId(defaultConfig.assistantId);
      }
    } else {
      setConfigDialogOpen(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
"""
    if old_effect not in content:
        raise RuntimeError("Failed to patch deep-agents-ui: initial config effect not found")
    return content.replace(old_effect, new_effect, 1)


def _apply_ui_branding(content: str) -> str:
    """Return page content with visible strings branded for msAgent."""
    replacements = {
        "Deep Agent UI": "msAgent",
        "Welcome to Standalone Chat": "Welcome to msAgent",
        "Configure your deployment to get started": "Connect to msAgent and start chatting",
    }
    updated = content
    for original, replacement in replacements.items():
        updated = updated.replace(original, replacement)

    if BRANDING_MARKER not in updated:
        anchor = 'function HomePageContent() {\n'
        branded_anchor = f'// {BRANDING_MARKER}\nfunction HomePageContent() {{\n'
        if anchor in updated:
            updated = updated.replace(anchor, branded_anchor, 1)

    return updated


def _update_ui_page(checkout_dir: Path, transform: Callable[[str], str]) -> None:
    """Rewrite the cached UI page only when a customization changes it."""
    page_path = checkout_dir / UI_PAGE_PATH
    content = page_path.read_text(encoding="utf-8")
    updated = transform(content)
    if updated != content:
        page_path.write_text(updated, encoding="utf-8")


def clear_stale_dev_lock(checkout_dir: Path) -> None:
    """Remove a stale Next.js dev lock left behind by an earlier crash."""
    lock_file = checkout_dir / ".next" / "dev" / "lock"
    lock_file.unlink(missing_ok=True)


def build_ui_environment(
    *,
    deployment_url: str,
    assistant_id: str,
) -> dict[str, str]:
    """Build the environment used to launch deep-agents-ui."""
    env = dict(os.environ)
    env.setdefault("NEXT_TELEMETRY_DISABLED", "1")
    env[ENV_UI_DEPLOYMENT_URL] = deployment_url
    env[ENV_UI_ASSISTANT_ID] = assistant_id
    return env
