#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cleanup accuracy cache (replaces MCP accuracy_cleanup)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Bootstrap: add shared library to sys.path before any cross-skill imports
# ---------------------------------------------------------------------------
_common_dir = Path(__file__).resolve().parents[2] / "msmodelslim-tools-common" / "scripts"
if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

from script_utils import emit_result, ensure_msmodelslim


def accuracy_cleanup(save_path: str, keep_last: int) -> Dict[str, Any]:
    from msmodelslim.utils.security import yaml_safe_dump, yaml_safe_load

    try:
        if keep_last < 0:
            return {"ok": False, "error": "keep_last must be >= 0"}
        accuracy_path = Path(save_path) / "history" / "accuracy.yaml"
        if not accuracy_path.exists():
            return {"ok": True, "message": "accuracy file not found, nothing to cleanup", "removed": 0}
        raw = yaml_safe_load(str(accuracy_path)) or {}
        if not isinstance(raw, dict):
            return {"ok": False, "error": "accuracy.yaml content is invalid"}
        items = list(raw.items())
        removed = max(0, len(items) - keep_last)
        kept = dict(items[-keep_last:]) if keep_last > 0 else {}
        yaml_safe_dump(kept, str(accuracy_path))
        return {"ok": True, "message": "accuracy cleanup finished", "removed": removed, "remaining": len(kept)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup accuracy cache, keeping last N entries.")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--keep-last", type=int, required=True)
    args = parser.parse_args()

    ensure_msmodelslim()
    return emit_result(accuracy_cleanup(args.save_path, args.keep_last))


if __name__ == "__main__":
    sys.exit(main())
