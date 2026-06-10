#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Append tuning history record (replaces MCP history_append)."""

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

from script_utils import emit_result, ensure_msmodelslim, parse_optional_json


def _load_practice(practice_path: str):
    from msmodelslim.core.practice import PracticeConfig
    from msmodelslim.utils.security import yaml_safe_load

    content = yaml_safe_load(practice_path)
    return PracticeConfig.model_validate(content)


def history_append(save_path: str, practice_path: str, evaluate_result: Dict[str, Any]) -> Dict[str, Any]:
    from msmodelslim.core.tune_strategy import EvaluateResult
    from msmodelslim.infra.yaml_practice_history_manager import YamlTuningHistoryManager

    try:
        history_path = str(Path(save_path) / "history")
        history = YamlTuningHistoryManager().load_history(history_path)
        practice_obj = _load_practice(practice_path)
        evaluate_obj = EvaluateResult.model_validate(evaluate_result)
        history.append_history(practice_obj, evaluate_obj)
        return {"ok": True, "message": "history appended"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Append one tuning history record.")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--practice-path", required=True)
    parser.add_argument("--evaluate-result", required=True, help="JSON object of EvaluateResult")
    args = parser.parse_args()

    ensure_msmodelslim()
    evaluate_result = parse_optional_json(args.evaluate_result, default={})
    if not isinstance(evaluate_result, dict):
        return emit_result({"ok": False, "error": "evaluate-result must be a JSON object"})
    return emit_result(history_append(args.save_path, args.practice_path, evaluate_result))


if __name__ == "__main__":
    sys.exit(main())
