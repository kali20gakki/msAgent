from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface
except ModuleNotFoundError:  # pragma: no cover - build dependency is not installed in tests
    class BuildHookInterface:  # type: ignore[no-redef]
        def __init__(
            self,
            *args,
            root: str = ".",
            directory: str = ".",
            target_name: str = "",
            config: dict[str, object] | None = None,
            **kwargs,
        ) -> None:
            del args, kwargs
            self.root = root
            self.directory = directory
            self.target_name = target_name
            self.config = config or {}


DEFAULT_UI_ARCHIVE_NAME = "deep-agents-ui.tar.gz"
DEFAULT_UI_ARCHIVE_URL_TEMPLATE = "https://codeload.github.com/langchain-ai/deep-agents-ui/tar.gz/{ref}"
ENV_BUNDLE_WEB_UI = "MSAGENT_BUNDLE_WEB_UI"
ENV_UI_ARCHIVE_URL = "MSAGENT_UI_ARCHIVE_URL"
ENV_UI_REF = "MSAGENT_UI_REF"


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


class CustomBuildHook(BuildHookInterface):
    def clean(self, versions: list[str]) -> None:
        del versions
        generated_dir = Path(self.directory) / "msagent-build"
        shutil.rmtree(generated_dir, ignore_errors=True)

    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        del version
        if not _is_truthy(os.getenv(ENV_BUNDLE_WEB_UI, "1")):
            return

        archive_path = self._ensure_ui_archive()
        force_include = build_data.setdefault("force_include", {})
        if not isinstance(force_include, dict):
            raise TypeError("build_data.force_include must be a dict[str, str]")

        force_include[str(archive_path)] = self._distribution_path()

    def _distribution_path(self) -> str:
        if self.target_name == "sdist":
            return f"src/msagent/web/vendor/{DEFAULT_UI_ARCHIVE_NAME}"
        return f"msagent/web/vendor/{DEFAULT_UI_ARCHIVE_NAME}"

    def _ensure_ui_archive(self) -> Path:
        source_archive = Path(self.root) / "src" / "msagent" / "web" / "vendor" / DEFAULT_UI_ARCHIVE_NAME
        if source_archive.is_file():
            return source_archive

        generated_dir = Path(self.directory) / "msagent-build" / "web-ui"
        generated_dir.mkdir(parents=True, exist_ok=True)
        generated_archive = generated_dir / DEFAULT_UI_ARCHIVE_NAME
        if generated_archive.is_file():
            return generated_archive

        archive_url = os.getenv(ENV_UI_ARCHIVE_URL, "").strip()
        if not archive_url:
            archive_ref = os.getenv(ENV_UI_REF, "main").strip() or "main"
            archive_url = DEFAULT_UI_ARCHIVE_URL_TEMPLATE.format(ref=archive_ref)

        request = urllib.request.Request(
            archive_url,
            headers={"User-Agent": "msagent-wheel-build"},
        )
        try:
            with urllib.request.urlopen(request) as response, generated_archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        except urllib.error.URLError as exc:  # pragma: no cover - depends on network
            raise RuntimeError(
                f"Failed to download bundled deep-agents-ui archive from {archive_url}"
            ) from exc

        return generated_archive
