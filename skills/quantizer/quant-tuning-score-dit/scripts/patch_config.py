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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch the upstream AISBench VBench template in place.

The AISBench ``eval_vbench_standard.py`` template hard-codes three path-style
variables at module scope:

* ``DATA_PATH``          — root directory of generated videos
* ``VBENCH_CACHE_DIR``   — VBench small-model cache
* ``full_json_dir``      — VBench_kmeans_info*.json

This module copies the upstream template and rewrites those lines so the
scorer can call ``ais_bench <patched_config> --mode eval`` without editing
the upstream ``benchmark/ais_bench/`` tree.

Failures raise ``TemplateError`` (mapped by ``score.py`` to stable error codes
``TEMPLATE_MISMATCH`` / ``TEMPLATE_SYNTAX_ERROR`` / ``FULL_JSON_*``).
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


class TemplateError(RuntimeError):
    """Raised when the upstream AISBench template is incompatible."""


def map_template_err(msg: str) -> str:
    """Map a ``TemplateError`` message to a stable error code (shared with score.py).

    Specific codes take priority; ``TEMPLATE_MISMATCH`` is the catch-all.
    Order matters: the most specific substring check wins.
    """
    for needle, code in (
        ("TEMPLATE_SYNTAX_ERROR", "TEMPLATE_SYNTAX_ERROR"),
        ("FULL_JSON_REQUIRED", "FULL_JSON_REQUIRED"),
        ("FULL_JSON_NOT_FOUND", "FULL_JSON_NOT_FOUND"),
    ):
        if needle in msg:
            return code
    return "TEMPLATE_MISMATCH"


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------


def _resolve_template_path() -> Path:
    """Locate ``eval_vbench_standard.py`` without importing ``ais_bench``."""
    filename = "eval_vbench_standard.py"
    candidates = []
    try:
        import ais_bench  # type: ignore[import-not-found]
        candidates.append(Path(ais_bench.__file__).resolve().parent / "configs" / "vbench_examples" / filename)
    except Exception:
        pass
    candidates += [
        Path("benchmark") / "ais_bench" / "configs" / "vbench_examples" / filename,
        Path("ais_bench") / "configs" / "vbench_examples" / filename,
        Path("/opt/ais_bench") / "configs" / "vbench_examples" / filename,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise TemplateError(
        f"Could not locate AISBench VBench template {filename!r}. "
        f"Tried: {[str(c) for c in candidates]}. "
        f"Install ais_bench or run from a checkout containing "
        f"benchmark/ais_bench/configs/vbench_examples/."
    )


# ---------------------------------------------------------------------------
# Substitutions
# ---------------------------------------------------------------------------


_DATA_PATH_PATTERN = re.compile(r'^DATA_PATH\s*=\s*"[^"]*"\s*$', re.MULTILINE)
_CACHE_DIR_PATTERN = re.compile(r'^VBENCH_CACHE_DIR\s*=\s*"[^"]*"\s*$', re.MULTILINE)
# full_json_dir shows up in two shapes in upstream:
#   1. literal:    full_json_dir = "..."      (rare; newer template)
#   2. comment:    # full_json_dir: ...        (older template, value commented out)
_FULL_JSON_LITERAL = re.compile(r'full_json_dir\s*=\s*"[^"]*"')
_FULL_JSON_COMMENT = re.compile(r'^[ \t]*#[ \t]*full_json_dir:[^\n]*$', re.MULTILINE)


def _check_path(value: str, label: str, *, must_be_dir: bool = False) -> None:
    """Light sanity check before embedding the value into Python source."""
    if not value or not value.strip():
        raise TemplateError(f"{label} is empty")
    if any(ch in value for ch in ('"', "\\", "\n", "\r")):
        raise TemplateError(f"{label} contains disallowed characters (quote / backslash / newline)")
    if must_be_dir and not Path(value).expanduser().is_dir():
        raise TemplateError(f"{label} does not point to an existing directory: {value!r}")


def _replace_assignment(text: str, pattern: re.Pattern[str], new_value: str, label: str) -> str:
    """Replace the first matching assignment line; raise if not found exactly once."""
    new_text, count = pattern.subn(f'{label} = "{new_value}"', text, count=1)
    if count != 1:
        raise TemplateError(
            f"TEMPLATE_MISMATCH: failed to rewrite assignment {label!r} "
            f"(expected exactly one match for {pattern.pattern!r}). "
            f"Upstream AISBench template format may have drifted."
        )
    return new_text


def _replace_full_json(text: str, new_value: str) -> str:
    """Replace or uncomment the ``full_json_dir`` value.

    Upstream ``eval_vbench_standard.py`` historically ships with the value
    commented out; newer versions may set it to a literal empty string. We
    handle both shapes and raise ``TEMPLATE_MISMATCH`` if neither matches.
    """
    new_text, count = _FULL_JSON_LITERAL.subn(f'full_json_dir="{new_value}"', text, count=1)
    if count == 1:
        return new_text

    new_text, count = _FULL_JSON_COMMENT.subn(f'    full_json_dir="{new_value}",', text, count=1)
    if count == 1:
        return new_text

    raise TemplateError(
        "TEMPLATE_MISMATCH: failed to set full_json_dir (no literal assignment "
        "and no documentation comment to uncomment). "
        "Upstream AISBench template format has drifted."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def patch_config(
    *,
    data_path: str,
    vbench_cache_dir: str,
    full_json_dir: str,
    out_path: Path,
) -> Path:
    """Copy the upstream template and substitute the three path variables."""
    if not full_json_dir:
        raise TemplateError("FULL_JSON_REQUIRED: --full-json-dir must be supplied.")
    if not Path(full_json_dir).expanduser().is_file():
        raise TemplateError(f"FULL_JSON_NOT_FOUND: {full_json_dir!r} is not an existing file")
    _check_path(data_path, "data_path", must_be_dir=True)
    _check_path(vbench_cache_dir, "vbench_cache_dir", must_be_dir=True)

    text = _resolve_template_path().read_text(encoding="utf-8")
    text = _replace_assignment(text, _DATA_PATH_PATTERN, data_path, "DATA_PATH")
    text = _replace_assignment(text, _CACHE_DIR_PATTERN, vbench_cache_dir, "VBENCH_CACHE_DIR")
    text = _replace_full_json(text, full_json_dir)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise TemplateError(f"TEMPLATE_SYNTAX_ERROR: patched template does not parse: {exc}") from exc

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Debug CLI (not used by score.py; kept for poking at the patch in isolation)
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Patch AISBench VBench template.")
    p.add_argument("--data-path", required=True)
    p.add_argument("--vbench-cache-dir", required=True)
    p.add_argument("--full-json-dir", required=True)
    p.add_argument("--out", required=True)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    try:
        written = patch_config(
            data_path=args.data_path,
            vbench_cache_dir=args.vbench_cache_dir,
            full_json_dir=args.full_json_dir,
            out_path=Path(args.out),
        )
    except TemplateError as exc:
        import json
        msg = str(exc)
        sys.stderr.write(json.dumps({"ok": False, "error": {"code": map_template_err(msg), "message": msg}}) + "\n")
        return 2
    print(str(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())