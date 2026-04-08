"""Helpers for running the official deep-agents-ui frontend."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from importlib.resources import as_file, files
from pathlib import Path

from msagent.core.constants import APP_NAME

DEEP_AGENTS_UI_REPO = "https://github.com/langchain-ai/deep-agents-ui"
DEEP_AGENTS_UI_DIRNAME = "deep-agents-ui"
BUNDLED_UI_ARCHIVE_PACKAGE = "msagent.web.vendor"
BUNDLED_UI_ARCHIVE_NAME = "deep-agents-ui.tar.gz"
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
    """Materialize a local UI checkout, preferring the bundled archive over git clone."""
    ensure_node_available()

    checkout_dir = ui_checkout_dir()
    package_json = checkout_dir / "package.json"
    if package_json.exists():
        return checkout_dir

    if extract_bundled_ui_checkout(checkout_dir):
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


def extract_bundled_ui_checkout(checkout_dir: Path) -> bool:
    """Seed the cached checkout from the bundled UI archive when available."""
    archive_resource = _bundled_ui_archive_resource()
    if archive_resource is None:
        return False

    if checkout_dir.exists():
        shutil.rmtree(checkout_dir)
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)

    with as_file(archive_resource) as archive_path:
        _extract_ui_archive(archive_path, checkout_dir)
    return True


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


def _bundled_ui_archive_resource():
    try:
        archive = files(BUNDLED_UI_ARCHIVE_PACKAGE).joinpath(BUNDLED_UI_ARCHIVE_NAME)
    except ModuleNotFoundError:
        return None
    return archive if archive.is_file() else None


def _extract_ui_archive(archive_path: Path, checkout_dir: Path) -> None:
    with tempfile.TemporaryDirectory(
        prefix="msagent-ui-",
        dir=str(checkout_dir.parent),
    ) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract_tar(archive, temp_dir)
        source_dir = _find_extracted_ui_root(temp_dir)
        shutil.move(str(source_dir), str(checkout_dir))


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    destination_root = destination.resolve()
    for member in archive.getmembers():
        member_path = (destination / member.name).resolve()
        if not member_path.is_relative_to(destination_root):
            raise RuntimeError("Bundled deep-agents-ui archive contains an invalid path")
    archive.extractall(destination)


def _find_extracted_ui_root(extract_dir: Path) -> Path:
    direct_candidates = [extract_dir]
    direct_candidates.extend(path for path in extract_dir.iterdir() if path.is_dir())
    for candidate in direct_candidates:
        if (candidate / "package.json").exists():
            return candidate

    package_json_candidates = list(extract_dir.rglob("package.json"))
    if len(package_json_candidates) == 1:
        return package_json_candidates[0].parent

    raise RuntimeError("Bundled deep-agents-ui archive did not contain a package.json file")


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
