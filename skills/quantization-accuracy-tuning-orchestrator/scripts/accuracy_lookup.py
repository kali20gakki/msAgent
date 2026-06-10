#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lookup accuracy cache (replaces MCP accuracy_lookup)."""

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


def _load_practice(practice_path: str):
    from msmodelslim.core.practice import PracticeConfig
    from msmodelslim.utils.security import yaml_safe_load

    content = yaml_safe_load(practice_path)
    return PracticeConfig.model_validate(content)


def _load_evaluate_config(evaluate_config_path: str):
    from msmodelslim.infra.service_oriented_evaluate_service import ServiceOrientedEvaluateServiceConfig
    from msmodelslim.utils.security import yaml_safe_load

    config_dict = yaml_safe_load(evaluate_config_path)
    return ServiceOrientedEvaluateServiceConfig.model_validate(config_dict)


def accuracy_lookup(save_path: str, evaluate_config_path: str, practice_path: str) -> Dict[str, Any]:
    from msmodelslim.infra.yaml_practice_accuracy_manager import YamlTuningAccuracyManager

    try:
        history_path = str(Path(save_path) / "history")
        accuracy = YamlTuningAccuracyManager().load_accuracy(history_path)
        evaluate_config = _load_evaluate_config(evaluate_config_path)
        practice_obj = _load_practice(practice_path)
        evaluate_result = accuracy.get_accuracy(practice_obj, evaluate_config)
        if evaluate_result is None:
            return {"ok": True, "cache_hit": False}
        return {
            "ok": True,
            "cache_hit": True,
            "evaluate_result": evaluate_result.model_dump(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Lookup accuracy cache for practice + evaluate config.")
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--evaluate-config-path", required=True)
    parser.add_argument("--practice-path", required=True)
    args = parser.parse_args()

    ensure_msmodelslim()
    return emit_result(
        accuracy_lookup(args.save_path, args.evaluate_config_path, args.practice_path)
    )


if __name__ == "__main__":
    sys.exit(main())
