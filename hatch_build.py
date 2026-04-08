from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

try:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface  # type: ignore[import-not-found]
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
DEFAULT_UI_STANDALONE_ARCHIVE_NAME = "deep-agents-ui-standalone.tar.gz"
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

        force_include = build_data.setdefault("force_include", {})
        if not isinstance(force_include, dict):
            raise TypeError("build_data.force_include must be a dict[str, str]")

        source_archive = self._ensure_ui_archive()
        standalone_archive = self._ensure_ui_standalone_archive(source_archive)
        force_include[str(source_archive)] = self._distribution_path(DEFAULT_UI_ARCHIVE_NAME)
        force_include[str(standalone_archive)] = self._distribution_path(
            DEFAULT_UI_STANDALONE_ARCHIVE_NAME
        )

    def _distribution_path(self, archive_name: str) -> str:
        if self.target_name == "sdist":
            return f"src/msagent/web/vendor/{archive_name}"
        return f"msagent/web/vendor/{archive_name}"

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

    def _ensure_ui_standalone_archive(self, source_archive: Path) -> Path:
        source_standalone_archive = (
            Path(self.root) / "src" / "msagent" / "web" / "vendor" / DEFAULT_UI_STANDALONE_ARCHIVE_NAME
        )
        if source_standalone_archive.is_file():
            return source_standalone_archive

        generated_dir = Path(self.directory) / "msagent-build" / "web-ui"
        generated_dir.mkdir(parents=True, exist_ok=True)
        generated_archive = generated_dir / DEFAULT_UI_STANDALONE_ARCHIVE_NAME
        if generated_archive.is_file():
            return generated_archive

        npm_command = shutil.which("npm")
        node_command = shutil.which("node")
        if not npm_command or not node_command:
            raise RuntimeError(
                "Building the bundled standalone web UI requires local `node` and `npm`."
            )

        ui_module = self._load_ui_module()
        env = dict(os.environ)
        env.setdefault("NEXT_TELEMETRY_DISABLED", "1")

        with tempfile.TemporaryDirectory(prefix="msagent-ui-build-") as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            source_dir = temp_dir / "source"
            self._extract_archive(source_archive, source_dir)
            ui_module.ensure_ui_customizations(source_dir)
            self._ensure_next_standalone_config(source_dir)
            self._run(
                [
                    npm_command,
                    "install",
                    "--no-fund",
                    "--no-audit",
                    "--legacy-peer-deps",
                ],
                cwd=source_dir,
                env=env,
            )
            self._run([npm_command, "run", "build"], cwd=source_dir, env=env)

            bundle_dir = temp_dir / "bundle" / "deep-agents-ui-standalone"
            self._assemble_standalone_bundle(source_dir, bundle_dir)
            with tarfile.open(generated_archive, "w:gz") as archive:
                archive.add(bundle_dir, arcname=bundle_dir.name)

        return generated_archive

    def _load_ui_module(self):
        src_dir = Path(self.root) / "src"
        sys.path.insert(0, str(src_dir))
        try:
            return importlib.import_module("msagent.web.ui")
        finally:
            sys.path.remove(str(src_dir))

    def _run(self, command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
        subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            check=True,
        )

    def _ensure_next_standalone_config(self, source_dir: Path) -> None:
        config_candidates = [
            source_dir / "next.config.ts",
            source_dir / "next.config.mjs",
            source_dir / "next.config.js",
        ]
        for config_path in config_candidates:
            if not config_path.exists():
                continue

            content = config_path.read_text(encoding="utf-8")
            if re.search(r'output\s*:\s*["\']standalone["\']', content):
                return

            updated = re.sub(
                r"(const\s+nextConfig(?:\s*:\s*NextConfig)?\s*=\s*\{)",
                r'\1\n  output: "standalone",',
                content,
                count=1,
            )
            if updated == content:
                raise RuntimeError(f"Failed to enable standalone output in {config_path}")

            config_path.write_text(updated, encoding="utf-8")
            return

        (source_dir / "next.config.mjs").write_text(
            'const nextConfig = {\n  output: "standalone",\n};\n\nexport default nextConfig;\n',
            encoding="utf-8",
        )

    def _assemble_standalone_bundle(self, source_dir: Path, bundle_dir: Path) -> None:
        standalone_root = source_dir / ".next" / "standalone"
        static_root = source_dir / ".next" / "static"
        if not (standalone_root / "server.js").is_file():
            raise RuntimeError("Next standalone build did not produce server.js")
        if not static_root.is_dir():
            raise RuntimeError("Next standalone build did not produce .next/static")

        bundle_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(standalone_root, bundle_dir, dirs_exist_ok=True)
        shutil.copytree(
            static_root,
            bundle_dir / ".next" / "static",
            dirs_exist_ok=True,
        )
        public_dir = source_dir / "public"
        if public_dir.is_dir():
            shutil.copytree(public_dir, bundle_dir / "public", dirs_exist_ok=True)

    def _extract_archive(self, archive_path: Path, checkout_dir: Path) -> None:
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="msagent-ui-extract-",
            dir=str(checkout_dir.parent),
        ) as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            with tarfile.open(archive_path, "r:gz") as archive:
                self._safe_extract_tar(archive, temp_dir)
            source_dir = self._find_extracted_root(temp_dir, marker_path=Path("package.json"))
            shutil.move(str(source_dir), str(checkout_dir))

    def _safe_extract_tar(self, archive: tarfile.TarFile, destination: Path) -> None:
        destination_root = destination.resolve()
        for member in archive.getmembers():
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination_root):
                raise RuntimeError("Bundled deep-agents-ui archive contains an invalid path")
        archive.extractall(destination)

    def _find_extracted_root(self, extract_dir: Path, *, marker_path: Path) -> Path:
        direct_candidates = [extract_dir]
        direct_candidates.extend(path for path in extract_dir.iterdir() if path.is_dir())
        for candidate in direct_candidates:
            if (candidate / marker_path).exists():
                return candidate

        marker_candidates = list(extract_dir.rglob(str(marker_path)))
        if len(marker_candidates) == 1:
            return marker_candidates[0].parent

        raise RuntimeError(
            f"Bundled deep-agents-ui archive did not contain {marker_path.as_posix()}"
        )
