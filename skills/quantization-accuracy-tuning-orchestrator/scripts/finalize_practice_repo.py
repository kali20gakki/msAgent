#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finalize practice repo (replaces MCP practice_repo_finalize)."""

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
from shared import get_lab_practice_dir  # noqa: E402


def practice_repo_finalize(
    model_type: str,
    model_path: str,
    final_practice_path: str,
    trust_remote_code: bool = False,
) -> Dict[str, Any]:
    from msmodelslim.app.auto_tuning.model_info_interface import ModelInfoInterface
    from msmodelslim.core.practice import PracticeConfig
    from msmodelslim.infra.yaml_practice_manager import YamlPracticeManager
    from msmodelslim.model import PluginModelFactory
    from msmodelslim.utils.config import msmodelslim_config
    from msmodelslim.utils.security import yaml_safe_load

    try:
        custom_practice_dir = msmodelslim_config.env_vars.custom_practice_repo
        custom_practice_path = Path(custom_practice_dir) if custom_practice_dir else None

        practice_manager = YamlPracticeManager(
            official_config_dir=get_lab_practice_dir(),
            custom_config_dir=custom_practice_path,
        )
        model_adapter = PluginModelFactory().create(model_type, Path(model_path), trust_remote_code)
        practice_dict = yaml_safe_load(final_practice_path)
        practice_obj = PracticeConfig.model_validate(practice_dict)

        if practice_manager.is_saving_supported() and isinstance(model_adapter, ModelInfoInterface):
            practice_manager.save_practice(
                model_pedigree=model_adapter.get_model_pedigree(),
                practice=practice_obj,
            )
        return {"ok": True, "message": "practice repo finalized"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Save final Practice YAML to practice repository.")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--final-practice-path", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    ensure_msmodelslim()
    return emit_result(
        practice_repo_finalize(
            model_type=args.model_type,
            model_path=args.model_path,
            final_practice_path=args.final_practice_path,
            trust_remote_code=args.trust_remote_code,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
