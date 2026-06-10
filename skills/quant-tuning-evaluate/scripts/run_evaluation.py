#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run model evaluation from Evaluation YAML (replaces MCP evaluation_run)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Bootstrap: add shared library to sys.path before any cross-skill imports
# ---------------------------------------------------------------------------
_common_dir = Path(__file__).resolve().parents[2] / "msmodelslim-tools-common" / "scripts"
if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

from script_utils import emit_result, ensure_msmodelslim, parse_int_list
from shared import to_device_type  # noqa: E402


def run_evaluation(
    quant_model_path: str,
    evaluate_id: str,
    evaluate_config_path: str,
    save_path: str,
    device: str = "npu",
    device_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    from msmodelslim.app.auto_tuning.evaluation_service_infra import EvaluateContext
    from msmodelslim.infra.service_oriented_evaluate_service import (
        ServiceOrientedEvaluateService,
        ServiceOrientedEvaluateServiceConfig,
    )
    from msmodelslim.utils.security import yaml_safe_load

    try:
        evaluate_config_dict = yaml_safe_load(evaluate_config_path)
        evaluate_config = ServiceOrientedEvaluateServiceConfig.model_validate(evaluate_config_dict)
        evaluate_service = ServiceOrientedEvaluateService()
        evaluate_result = evaluate_service.evaluate(
            context=EvaluateContext(
                evaluate_id=evaluate_id,
                device=to_device_type(device),
                device_indices=device_indices,
                working_dir=Path(save_path),
            ),
            evaluate_config=evaluate_config,
            model_path=Path(quant_model_path),
        )
        return {"ok": True, "evaluate_result": evaluate_result.model_dump()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evaluation from Evaluation YAML.")
    parser.add_argument("--quant-model-path", required=True)
    parser.add_argument("--evaluate-id", required=True)
    parser.add_argument("--evaluate-config-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--device-indices",
        default=None,
        help="Comma-separated indices or JSON array, e.g. 0,1 or [0,1]",
    )
    args = parser.parse_args()

    ensure_msmodelslim()
    result = run_evaluation(
        quant_model_path=args.quant_model_path,
        evaluate_id=args.evaluate_id,
        evaluate_config_path=args.evaluate_config_path,
        save_path=args.save_path,
        device=args.device,
        device_indices=parse_int_list(args.device_indices),
    )
    return emit_result(result)


if __name__ == "__main__":
    sys.exit(main())
