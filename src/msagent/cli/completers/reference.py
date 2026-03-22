"""Reference completer for @ references."""

import os
import re
import time
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path

from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    FuzzyCompleter,
    WordCompleter,
)
from prompt_toolkit.document import Document

from msagent.cli.resolvers import RefType
from msagent.utils.image import SUPPORTED_IMAGE_EXTENSIONS
from msagent.utils.path import resolve_path


def parse_reference(ref: str) -> tuple[RefType | None, str]:
    """Parse typed reference into type and value.

    Args:
        ref: Reference string (e.g., "@:file:path" or ":file:path" or "path")

    Returns:
        Tuple of (RefType, value) or (None, original) if no type prefix
    """
    content = ref.lstrip("@")

    if content.startswith(":"):
        parts = content[1:].split(":", 1)
        if len(parts) == 2 and parts[1]:
            type_str, value = parts
            try:
                return RefType(type_str), value
            except ValueError:
                pass

    return None, content


class ReferenceCompleter(Completer):
    """Completer for @ references."""

    _FRAGMENT_PATTERN = re.compile(r"[^\s@]+")
    _TRIGGER_GUARDS = frozenset((".", "-", "_", "`", "'", '"', ":", "@", "#", "~"))
    _IGNORED_NAME_GROUPS: dict[str, tuple[str, ...]] = {
        "vcs_metadata": (".DS_Store", ".bzr", ".git", ".hg", ".svn"),
        "tooling_caches": (
            ".build",
            ".cache",
            ".coverage",
            ".fleet",
            ".gradle",
            ".idea",
            ".ipynb_checkpoints",
            ".pnpm-store",
            ".pytest_cache",
            ".pub-cache",
            ".ruff_cache",
            ".swiftpm",
            ".tox",
            ".venv",
            ".vs",
            ".vscode",
            ".yarn",
            ".yarn-cache",
        ),
        "js_frontend": (
            ".next",
            ".nuxt",
            ".parcel-cache",
            ".svelte-kit",
            ".turbo",
            ".vercel",
            "node_modules",
        ),
        "python_packaging": (
            "__pycache__",
            "build",
            "coverage",
            "dist",
            "htmlcov",
            "pip-wheel-metadata",
            "venv",
        ),
        "java_jvm": (".mvn", "out", "target"),
        "dotnet_native": ("bin", "cmake-build-debug", "cmake-build-release", "obj"),
        "bazel_buck": ("bazel-bin", "bazel-out", "bazel-testlogs", "buck-out"),
        "misc_artifacts": (
            ".dart_tool",
            ".serverless",
            ".stack-work",
            ".terraform",
            ".terragrunt-cache",
            "DerivedData",
            "Pods",
            "deps",
            "tmp",
            "vendor",
        ),
    }
    _IGNORED_NAMES = frozenset(
        name for group in _IGNORED_NAME_GROUPS.values() for name in group
    )
    _IGNORED_PATTERN_PARTS: tuple[str, ...] = (
        r".*_cache$",
        r".*-cache$",
        r".*\.egg-info$",
        r".*\.dist-info$",
        r".*\.py[co]$",
        r".*\.class$",
        r".*\.sw[po]$",
        r".*~$",
        r".*\.(?:tmp|bak)$",
    )
    _IGNORED_PATTERNS = re.compile(
        "|".join(f"(?:{part})" for part in _IGNORED_PATTERN_PARTS),
        re.IGNORECASE,
    )

    def __init__(
        self,
        working_dir: Path,
        max_suggestions: int = 10,
        refresh_interval: float = 2.0,
    ):
        """Initialize reference completer."""
        self.max_suggestions = max_suggestions
        self.working_dir = working_dir
        self._refresh_interval = refresh_interval
        self._cache_limit = max(max_suggestions * 200, 1000)
        self._cache_time = 0.0
        self._cached_paths: list[str] = []
        self._top_cache_time = 0.0
        self._top_cached_paths: list[str] = []
        self._fragment_hint: str | None = None
        self._type_hint: RefType | None = None

        self._word_completer = WordCompleter(
            self._get_paths,
            WORD=False,
            pattern=self._FRAGMENT_PATTERN,
        )
        self._fuzzy = FuzzyCompleter(
            self._word_completer,
            WORD=False,
            pattern=r"^[^\s@]*",
        )

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """Sync stub - completions are async only."""
        return iter([])

    async def get_completions_async(
        self, document: Document, complete_event: CompleteEvent
    ) -> AsyncGenerator[Completion]:
        """Get completions asynchronously."""
        fragment = self._extract_fragment(document.text_before_cursor)
        if fragment is None:
            return

        type_filter, ref_fragment = parse_reference(fragment)
        if self._is_completed_path(ref_fragment, type_filter):
            return

        mention_doc = Document(text=ref_fragment, cursor_position=len(ref_fragment))
        self._fragment_hint = ref_fragment
        self._type_hint = type_filter

        try:
            candidates = list(self._fuzzy.get_completions(mention_doc, complete_event))
            fragment_lower = ref_fragment.lower()
            candidates.sort(key=lambda candidate: self._rank_completion(candidate, fragment_lower))

            seen: set[tuple[str, int]] = set()
            yielded = 0
            for candidate in candidates:
                key = (candidate.text, candidate.start_position)
                if key in seen:
                    continue
                seen.add(key)
                yield candidate
                yielded += 1
                if yielded >= self.max_suggestions:
                    break
        finally:
            self._fragment_hint = None
            self._type_hint = None

    @staticmethod
    def _extract_fragment(text: str) -> str | None:
        """Extract the active @ reference fragment before the cursor."""
        index = text.rfind("@")
        if index == -1:
            return None

        if index > 0:
            prev = text[index - 1]
            if prev.isalnum() or prev in ReferenceCompleter._TRIGGER_GUARDS:
                return None

        fragment = text[index + 1 :]
        if not fragment:
            return ""
        if any(ch.isspace() for ch in fragment):
            return None
        return fragment

    @classmethod
    def _is_ignored(cls, name: str) -> bool:
        if not name:
            return True
        if name in cls._IGNORED_NAMES:
            return True
        return bool(cls._IGNORED_PATTERNS.fullmatch(name))

    def _get_paths(self) -> list[str]:
        fragment = self._fragment_hint or ""
        type_filter = self._type_hint

        if candidates := self._get_path_fragment_candidates(fragment, type_filter):
            return candidates

        if "/" not in fragment and len(fragment) < 3:
            return self._filter_candidates(self._get_top_level_paths(), type_filter)

        return self._filter_candidates(self._get_deep_paths(), type_filter)

    def _get_top_level_paths(self) -> list[str]:
        now = time.monotonic()
        if now - self._top_cache_time <= self._refresh_interval:
            return self._top_cached_paths

        entries: list[str] = []
        try:
            for entry in sorted(
                self.working_dir.iterdir(),
                key=lambda path: (not path.is_dir(), path.name.lower()),
            ):
                name = entry.name
                if self._is_ignored(name):
                    continue
                entries.append(f"{name}/" if entry.is_dir() else name)
                if len(entries) >= self._cache_limit:
                    break
        except OSError:
            return self._top_cached_paths

        self._top_cached_paths = entries
        self._top_cache_time = now
        return self._top_cached_paths

    def _get_deep_paths(self) -> list[str]:
        now = time.monotonic()
        if now - self._cache_time <= self._refresh_interval:
            return self._cached_paths

        paths: list[str] = []
        try:
            for current_root, dirs, files in os.walk(self.working_dir):
                relative_root = Path(current_root).relative_to(self.working_dir)
                dirs[:] = sorted(d for d in dirs if not self._is_ignored(d))

                if relative_root.parts and any(
                    self._is_ignored(part) for part in relative_root.parts
                ):
                    dirs[:] = []
                    continue

                if relative_root.parts:
                    paths.append(relative_root.as_posix() + "/")
                    if len(paths) >= self._cache_limit:
                        break

                for file_name in sorted(files):
                    if self._is_ignored(file_name):
                        continue

                    relative = (relative_root / file_name).as_posix()
                    if not relative:
                        continue

                    paths.append(relative)
                    if len(paths) >= self._cache_limit:
                        break

                if len(paths) >= self._cache_limit:
                    break
        except OSError:
            return self._cached_paths

        self._cached_paths = paths
        self._cache_time = now
        return self._cached_paths

    def _get_path_fragment_candidates(
        self,
        fragment: str,
        type_filter: RefType | None,
    ) -> list[str]:
        split_fragment = self._split_path_fragment(fragment)
        if split_fragment is None:
            return []

        directory_fragment, name_prefix = split_fragment
        resolved_directory = self._resolve_directory(directory_fragment)
        if resolved_directory is None or not resolved_directory.is_dir():
            return []

        prefix_lower = name_prefix.lower()
        candidates: list[str] = []
        try:
            for entry in sorted(
                resolved_directory.iterdir(),
                key=lambda path: (not path.is_dir(), path.name.lower()),
            ):
                name = entry.name
                if self._is_ignored(name):
                    continue
                if prefix_lower and not name.lower().startswith(prefix_lower):
                    continue
                if not self._path_matches_type(entry, type_filter):
                    continue

                candidate = self._join_path_fragment(directory_fragment, name)
                if entry.is_dir():
                    candidate += "/"
                candidates.append(candidate)

                if len(candidates) >= self._cache_limit:
                    break
        except OSError:
            return []

        return candidates

    @staticmethod
    def _split_path_fragment(fragment: str) -> tuple[str, str] | None:
        if not fragment:
            return None

        if fragment.endswith("/"):
            base = fragment.rstrip("/") or fragment
            return base, ""

        path_fragment = Path(fragment)
        directory_fragment = str(path_fragment.parent)
        name_prefix = path_fragment.name

        if directory_fragment == ".":
            if fragment.startswith("./"):
                directory_fragment = "."
            elif fragment.startswith("."):
                directory_fragment = ""
            elif fragment.startswith("~"):
                directory_fragment = "~"
            elif fragment.startswith("/"):
                directory_fragment = "/"
            else:
                return None

        return directory_fragment, name_prefix

    def _resolve_directory(self, directory_fragment: str) -> Path | None:
        try:
            if directory_fragment in ("", "."):
                return self.working_dir
            if directory_fragment == "/":
                return Path("/")
            if directory_fragment == "~":
                return Path.home()
            return resolve_path(str(self.working_dir), directory_fragment)
        except Exception:
            return None

    @staticmethod
    def _join_path_fragment(directory_fragment: str, name: str) -> str:
        if directory_fragment == "":
            return name
        if directory_fragment == ".":
            return f"./{name}"
        if directory_fragment == "/":
            return f"/{name}"
        if directory_fragment == "~":
            return f"~/{name}"
        return f"{directory_fragment.rstrip('/')}/{name}"

    def _filter_candidates(
        self,
        paths: list[str],
        type_filter: RefType | None,
    ) -> list[str]:
        if type_filter is None or type_filter == RefType.FILE:
            return paths

        filtered: list[str] = []
        for path in paths:
            if path.endswith("/"):
                filtered.append(path)
                continue
            if Path(path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                filtered.append(path)
        return filtered

    def _path_matches_type(self, path: Path, type_filter: RefType | None) -> bool:
        if path.is_dir():
            return True
        if type_filter in (None, RefType.FILE):
            return True
        return path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

    def _is_completed_path(
        self,
        fragment: str,
        type_filter: RefType | None,
    ) -> bool:
        candidate = fragment.rstrip("/")
        if not candidate:
            return False

        try:
            resolved = resolve_path(str(self.working_dir), candidate)
        except Exception:
            return False

        if not resolved.is_file():
            return False

        if type_filter == RefType.IMAGE:
            return resolved.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        return True

    @staticmethod
    def _rank_completion(candidate: Completion, fragment_lower: str) -> tuple[int, int, str]:
        path = candidate.text
        base_name = path.rstrip("/").split("/")[-1].lower()

        if not fragment_lower:
            category = 0
        elif base_name.startswith(fragment_lower):
            category = 0
        elif fragment_lower in base_name:
            category = 1
        elif path.lower().startswith(fragment_lower):
            category = 2
        elif fragment_lower in path.lower():
            category = 3
        else:
            category = 4

        return (category, int(not path.endswith("/")), path.lower())
