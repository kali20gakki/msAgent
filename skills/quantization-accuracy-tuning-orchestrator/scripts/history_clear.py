#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clear tuning history (replaces MCP history_clear)."""

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


def history_clear(save_path: str) -> Dict[str, Any]:
    from msmodelslim.infra.yaml_practice_history_manager import YamlTuningHistoryManager

    try:
        history_path = str(Path(save_path) / "history")
        history = YamlTuningHistoryManager().load_history(history_path)
        history.clear_records()
        return {"ok": True, "message": "history cleared"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear tuning history under save_path/history.")
    parser.add_argument("--save-path", required=True)
    args = parser.parse_args()

    ensure_msmodelslim()
    return emit_result(history_clear(args.save_path))


if __name__ == "__main__":
    sys.exit(main())
