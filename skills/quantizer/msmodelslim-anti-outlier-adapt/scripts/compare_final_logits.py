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
"""Compare final logits before and after an anti-outlier transform.

This script consumes saved logits only. It does not load, quantize, or mutate a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np


DEFAULT_ATOL = 1e-5
DEFAULT_RTOL = 1e-4
DEFAULT_MIN_COSINE = 0.99999
DEFAULT_TOP_K = 5
DEFAULT_MIN_TOP_K_OVERLAP = 0.8
DEFAULT_MAX_JS_DIVERGENCE = 1e-3
DEFAULT_TOP1_MARGIN_TOLERANCE = 1e-4
RUN_SCHEMA = "msagent.anti_outlier_single_run/v1"
PATCH_VALIDATION_SCHEMA = "msagent.anti_outlier_logits_capture_patch_validation/v1"
PATCH_SCHEMA = "msagent.anti_outlier_logits_capture_patch/v1"
LOGITS_SCOPES = {"full_sequence", "last_token"}
ALGORITHM_INTERFACE_NAMES = {
    "quarot": "QuaRotInterface",
    "flex_smooth_quant": "FlexSmoothQuantInterface",
    "flex_awq_ssz": "FlexSmoothQuantInterface",
    "iter_smooth": "IterSmoothInterface",
}


def _load_logits(path: Path) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "logits" in loaded.files:
                array = loaded["logits"]
            elif len(loaded.files) == 1:
                array = loaded[loaded.files[0]]
            else:
                raise ValueError(
                    f"{path} contains multiple arrays; provide an NPZ with a 'logits' key"
                )
        finally:
            loaded.close()
    else:
        array = loaded
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise ValueError(f"{path} does not contain numeric logits")
    return np.asarray(array, dtype=np.float64)


def _error_quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    quantile_levels = (0.0, 0.5, 0.9, 0.95, 0.99, 1.0)
    return {str(level): float(np.quantile(values, level)) for level in quantile_levels}


def _resolve_artifact(record_path: Path, artifact: Any) -> Path:
    if not isinstance(artifact, str) or not artifact:
        raise ValueError("run record contains a missing artifact path")
    path = Path(artifact)
    if not path.is_absolute():
        path = record_path.parent / path
    return path.expanduser().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_run_record(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run record {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("run record must contain a JSON object")
    return loaded


def validate_run_provenance(
    record: Mapping[str, Any],
    record_path: Path,
    algorithm: str,
    fp_logits_path: Path,
    anti_outlier_logits_path: Path,
) -> dict[str, Any]:
    """Validate that both logits belong to one successful patched run."""
    errors: list[str] = []
    if record.get("schema") != RUN_SCHEMA:
        errors.append("run record schema is unsupported")
    if record.get("algorithm") != algorithm:
        errors.append("algorithm does not match the run record")
    if record.get("quantization_run") is not False:
        errors.append("run record is not marked quantization_run=false")
    if record.get("status") not in ("PASS", "COMPARISON_FAILED"):
        errors.append("run record does not show a completed processor run")
    if record.get("processor_status") != "PASS":
        errors.append("run record does not show a successful processor execution")

    processor_config = record.get("processor_config")
    if not isinstance(processor_config, Mapping):
        errors.append("run record has no processor configuration")
    elif processor_config.get("type") != algorithm:
        errors.append("processor config type does not match the algorithm")

    practice = record.get("practice")
    spec = practice.get("spec") if isinstance(practice, Mapping) else None
    process = spec.get("process") if isinstance(spec, Mapping) else None
    save = spec.get("save") if isinstance(spec, Mapping) else None
    if not isinstance(process, list) or len(process) != 1:
        errors.append("practice must contain exactly one processor")
    elif not isinstance(process[0], Mapping) or process[0].get("type") != algorithm:
        errors.append("practice processor does not match the algorithm")
    elif isinstance(processor_config, Mapping) and dict(process[0]) != dict(
        processor_config
    ):
        errors.append("practice processor does not match the recorded processor config")
    if save != []:
        errors.append("practice save must be empty")

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        errors.append("run record has no artifacts")
        artifacts = {}
    try:
        recorded_before = _resolve_artifact(record_path, artifacts.get("before_logits"))
        recorded_after = _resolve_artifact(record_path, artifacts.get("after_logits"))
    except ValueError as exc:
        errors.append(str(exc))
        recorded_before = recorded_after = Path()
    if recorded_before != fp_logits_path.expanduser().resolve():
        errors.append("before logits path does not match the run record")
    if recorded_after != anti_outlier_logits_path.expanduser().resolve():
        errors.append("after logits path does not match the run record")
    artifact_hashes = record.get("artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        errors.append("run record has no logits artifact hashes")
    else:
        for name, path in (
            ("before_logits", fp_logits_path.expanduser().resolve()),
            ("after_logits", anti_outlier_logits_path.expanduser().resolve()),
        ):
            expected_hash = artifact_hashes.get(name)
            if not path.is_file():
                errors.append(f"{name} artifact does not exist")
            elif (
                not isinstance(expected_hash, str)
                or _sha256_file(path) != expected_hash
            ):
                errors.append(f"{name} artifact SHA256 does not match")

    checkpoint_id = record.get("checkpoint_identity")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        errors.append("run record has no checkpoint identity")
    input_summary = record.get("input_summary")
    if input_summary is None:
        errors.append("run record has no input summary")

    interface_validation = record.get("interface_validation")
    if not isinstance(interface_validation, Mapping):
        errors.append("run record has no interface validation provenance")
    else:
        try:
            interface_path = _resolve_artifact(
                record_path, interface_validation.get("path")
            )
        except ValueError as exc:
            errors.append(str(exc))
            interface_path = Path()
        interface_hash = interface_validation.get("sha256")
        if not interface_path.is_file():
            errors.append("interface validation artifact does not exist")
        elif (
            not isinstance(interface_hash, str)
            or _sha256_file(interface_path) != interface_hash
        ):
            errors.append("interface validation SHA256 does not match")
        try:
            interface_document = _load_run_record(interface_path)
        except ValueError as exc:
            errors.append(str(exc))
            interface_document = None
        if interface_document is not None:
            expected_interface = ALGORITHM_INTERFACE_NAMES.get(algorithm)
            if interface_document.get("interface") != expected_interface:
                errors.append("interface validation does not match the algorithm")
            if interface_document.get("status") != "PASS":
                errors.append("interface validation did not pass")
            if interface_document.get("checkpoint_identity") != checkpoint_id:
                errors.append("interface validation checkpoint identity does not match")

    patch = record.get("patch")
    validation: dict[str, Any] | None = None
    if not isinstance(patch, Mapping):
        errors.append("run record has no patch provenance")
    else:
        patch_path = _resolve_artifact(record_path, patch.get("path"))
        patch_sha256 = patch.get("sha256")
        validation_path = _resolve_artifact(
            record_path, patch.get("validation_artifact")
        )
        if not patch_path.is_file():
            errors.append("recorded logits patch does not exist")
        elif (
            not isinstance(patch_sha256, str)
            or _sha256_file(patch_path) != patch_sha256
        ):
            errors.append("recorded logits patch SHA256 does not match")
        try:
            validation = _load_run_record(validation_path)
        except ValueError as exc:
            errors.append(str(exc))
        if validation is not None:
            if validation.get("schema") != PATCH_VALIDATION_SCHEMA:
                errors.append("patch validation schema is unsupported")
            if validation.get("status") != "PASS":
                errors.append("patch validation did not pass")
            if validation.get("patch_path") != str(patch_path):
                errors.append("patch validation path does not match run provenance")
            if validation.get("patch_sha256") != patch_sha256:
                errors.append("patch validation SHA256 does not match run provenance")
            if validation.get("checkpoint_identity") != checkpoint_id:
                errors.append("patch validation checkpoint identity does not match")
            if validation.get("input_summary") != input_summary:
                errors.append("patch validation input summary does not match")
            validation_hash = patch.get("validation_sha256")
            if (
                not isinstance(validation_hash, str)
                or _sha256_file(validation_path) != validation_hash
            ):
                errors.append("patch validation SHA256 does not match run provenance")
            validation_checks = validation.get("checks")
            required_checks = (
                "outputs_finite",
                "shape_matches",
                "dtype_matches",
                "repeatable",
            )
            if not isinstance(validation_checks, Mapping) or any(
                validation_checks.get(name) is not True for name in required_checks
            ):
                errors.append("patch validation behavior checks did not all pass")
            metadata = patch.get("metadata")
            validation_metadata = validation.get("metadata")
            if metadata != validation_metadata:
                errors.append("patch metadata does not match its validation artifact")
            required_metadata = (
                "schema",
                "model_type",
                "adapter_class",
                "checkpoint_identity",
                "logits_scope",
            )
            if not isinstance(metadata, Mapping) or any(
                not isinstance(metadata.get(name), str) or not metadata.get(name)
                for name in required_metadata
            ):
                errors.append("patch metadata is incomplete")
            else:
                if metadata.get("schema") != PATCH_SCHEMA:
                    errors.append("patch metadata schema is unsupported")
                if metadata.get("logits_scope") not in LOGITS_SCOPES:
                    errors.append("patch metadata logits scope is unsupported")
                if metadata.get("checkpoint_identity") != checkpoint_id:
                    errors.append(
                        "patch checkpoint identity does not match run provenance"
                    )
                if record.get("model_type") != metadata.get("model_type"):
                    errors.append("patch model type does not match run provenance")
                if record.get("adapter_class") != metadata.get("adapter_class"):
                    errors.append("patch adapter class does not match run provenance")

    if errors:
        raise ValueError("invalid anti-outlier run provenance: " + "; ".join(errors))

    return {
        "run_record": str(record_path.expanduser().resolve()),
        "checkpoint_identity": checkpoint_id,
        "input_summary": input_summary,
        "interface_validation": {
            "path": str(_resolve_artifact(record_path, interface_validation["path"])),
            "sha256": interface_validation["sha256"],
        },
        "patch": {
            "path": str(_resolve_artifact(record_path, patch["path"])),
            "sha256": patch["sha256"],
            "validation_artifact": str(
                _resolve_artifact(record_path, patch["validation_artifact"])
            ),
            "logits_scope": patch["metadata"]["logits_scope"],
        },
    }


def _select_distribution_logits(logits: np.ndarray, logits_scope: str) -> np.ndarray:
    if logits_scope == "last_token":
        selected = logits
    elif logits_scope == "full_sequence":
        if logits.ndim < 2:
            raise ValueError(
                "full_sequence logits must have a sequence and vocabulary dimension"
            )
        selected = logits[..., -1, :]
    else:
        raise ValueError(f"unsupported logits scope: {logits_scope}")
    if selected.ndim < 1 or selected.shape[-1] == 0:
        raise ValueError("selected logits have no vocabulary dimension")
    return selected.reshape(-1, selected.shape[-1])


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def _kl_divergence(probability: np.ndarray, reference: np.ndarray) -> np.ndarray:
    terms = np.zeros_like(probability)
    nonzero = probability > 0
    terms[nonzero] = probability[nonzero] * np.log(
        probability[nonzero] / reference[nonzero]
    )
    return np.sum(terms, axis=-1)


def _top_k_samples(
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    baseline_top_k: np.ndarray,
    candidate_top_k: np.ndarray,
) -> list[dict[str, Any]]:
    samples = []
    for row in range(min(8, baseline_probability.shape[0])):
        samples.append(
            {
                "row": row,
                "baseline": [
                    {
                        "token_id": int(token_id),
                        "probability": float(baseline_probability[row, token_id]),
                    }
                    for token_id in baseline_top_k[row]
                ],
                "candidate": [
                    {
                        "token_id": int(token_id),
                        "probability": float(candidate_probability[row, token_id]),
                    }
                    for token_id in candidate_top_k[row]
                ],
            }
        )
    return samples


def compare_logits(
    fp_logits: np.ndarray,
    anti_outlier_logits: np.ndarray,
    *,
    atol: float,
    rtol: float,
    min_cosine_similarity: float,
    logits_scope: str = "last_token",
    top_k: int = DEFAULT_TOP_K,
    min_top_k_overlap: float = DEFAULT_MIN_TOP_K_OVERLAP,
    max_js_divergence: float = DEFAULT_MAX_JS_DIVERGENCE,
    top1_margin_tolerance: float = DEFAULT_TOP1_MARGIN_TOLERANCE,
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= min_top_k_overlap <= 1.0:
        raise ValueError("min_top_k_overlap must be between 0 and 1")
    if max_js_divergence < 0 or not math.isfinite(max_js_divergence):
        raise ValueError("max_js_divergence must be finite and non-negative")
    if top1_margin_tolerance < 0 or not math.isfinite(top1_margin_tolerance):
        raise ValueError("top1_margin_tolerance must be finite and non-negative")

    shape_matches = fp_logits.shape == anti_outlier_logits.shape
    fp_finite = bool(np.isfinite(fp_logits).all())
    candidate_finite = bool(np.isfinite(anti_outlier_logits).all())

    metrics: dict[str, float | None] = {
        "max_abs_error": None,
        "mean_abs_error": None,
        "cosine_similarity": None,
    }
    error_quantiles: dict[str, Any] = {
        "absolute": {},
        "relative": {},
    }
    elementwise_tolerance_passed = False
    cosine_diagnostic_passed = False
    top1_consistent = False
    top_k_overlap_passed = False
    js_divergence_passed = False
    error_limit: float | None = None
    passed = False
    distribution_metrics: dict[str, Any] = {}
    top_k_output: list[dict[str, Any]] = []

    if shape_matches and fp_finite and candidate_finite:
        delta = np.abs(fp_logits - anti_outlier_logits)
        max_abs_error = float(delta.max(initial=0.0))
        mean_abs_error = float(delta.mean()) if delta.size else 0.0
        fp_scale = float(np.abs(fp_logits).max(initial=0.0))
        error_limit = atol + rtol * fp_scale
        elementwise_limit = atol + rtol * np.abs(fp_logits)
        elementwise_tolerance_passed = bool(
            np.less_equal(delta, elementwise_limit).all()
        )
        relative_delta = delta / np.maximum(np.abs(fp_logits), np.finfo(np.float64).eps)
        error_quantiles = {
            "absolute": _error_quantiles(delta),
            "relative": _error_quantiles(relative_delta),
        }

        fp_flat = fp_logits.ravel()
        candidate_flat = anti_outlier_logits.ravel()
        denominator = float(np.linalg.norm(fp_flat) * np.linalg.norm(candidate_flat))
        if denominator == 0.0:
            cosine_similarity = 1.0 if np.array_equal(fp_flat, candidate_flat) else 0.0
        else:
            cosine_similarity = float(np.dot(fp_flat, candidate_flat) / denominator)
            cosine_similarity = max(-1.0, min(1.0, cosine_similarity))

        metrics = {
            "max_abs_error": max_abs_error,
            "mean_abs_error": mean_abs_error,
            "cosine_similarity": cosine_similarity,
        }
        cosine_diagnostic_passed = cosine_similarity >= min_cosine_similarity

        baseline_selected = _select_distribution_logits(fp_logits, logits_scope)
        candidate_selected = _select_distribution_logits(
            anti_outlier_logits, logits_scope
        )
        baseline_probability = _softmax(baseline_selected)
        candidate_probability = _softmax(candidate_selected)
        effective_top_k = min(top_k, baseline_probability.shape[-1])
        baseline_ranking = np.argsort(-baseline_probability, axis=-1, kind="stable")
        candidate_ranking = np.argsort(-candidate_probability, axis=-1, kind="stable")
        baseline_top_k = baseline_ranking[:, :effective_top_k]
        candidate_top_k = candidate_ranking[:, :effective_top_k]
        baseline_top1 = baseline_top_k[:, 0]
        candidate_top1 = candidate_top_k[:, 0]
        top1_exact = baseline_top1 == candidate_top1

        if baseline_probability.shape[-1] > 1:
            rows = np.arange(baseline_probability.shape[0])
            baseline_margin = (
                baseline_probability[rows, baseline_ranking[:, 0]]
                - baseline_probability[rows, baseline_ranking[:, 1]]
            )
        else:
            baseline_margin = np.full(baseline_probability.shape[0], np.inf)

        overlap_ratios = np.asarray(
            [
                len(set(baseline).intersection(candidate)) / effective_top_k
                for baseline, candidate in zip(baseline_top_k, candidate_top_k)
            ],
            dtype=np.float64,
        )
        mutual_top_k = np.asarray(
            [
                candidate_top1[row] in baseline_top_k[row]
                and baseline_top1[row] in candidate_top_k[row]
                for row in range(baseline_probability.shape[0])
            ],
            dtype=bool,
        )
        margin_accepted = (
            ~top1_exact & (baseline_margin <= top1_margin_tolerance) & mutual_top_k
        )
        top1_consistent = bool(np.all(top1_exact | margin_accepted))
        top_k_overlap_passed = bool(np.all(overlap_ratios >= min_top_k_overlap))

        midpoint = (baseline_probability + candidate_probability) / 2.0
        js_divergence = 0.5 * (
            _kl_divergence(baseline_probability, midpoint)
            + _kl_divergence(candidate_probability, midpoint)
        )
        js_divergence_passed = bool(np.all(js_divergence <= max_js_divergence))
        probability_l1 = np.sum(
            np.abs(baseline_probability - candidate_probability), axis=-1
        )
        distribution_metrics = {
            "rows": int(baseline_probability.shape[0]),
            "vocabulary_size": int(baseline_probability.shape[-1]),
            "effective_top_k": effective_top_k,
            "top1_exact_match_rate": float(np.mean(top1_exact)),
            "top1_margin_accepted_rate": float(np.mean(margin_accepted)),
            "top_k_overlap_min": float(np.min(overlap_ratios)),
            "top_k_overlap_mean": float(np.mean(overlap_ratios)),
            "js_divergence_max": float(np.max(js_divergence)),
            "js_divergence_mean": float(np.mean(js_divergence)),
            "probability_l1_max": float(np.max(probability_l1)),
            "probability_l1_mean": float(np.mean(probability_l1)),
            "baseline_top1_margin_min": float(np.min(baseline_margin)),
            "baseline_top1_margin_max": float(np.max(baseline_margin)),
        }
        top_k_output = _top_k_samples(
            baseline_probability,
            candidate_probability,
            baseline_top_k,
            candidate_top_k,
        )
        # The workflow validates that the model reaches effectively the same
        # token decision.  JS divergence remains useful for spotting changes in
        # confidence, but confidence drift alone must not reject an otherwise
        # stable Top-1/Top-K result.
        passed = top1_consistent and top_k_overlap_passed

    return {
        "schema": "msagent.anti_outlier_final_logits_comparison/v1",
        "gate_policy": "last_token_token_consistency/v1",
        "passed": passed,
        "comparison": {
            "baseline": "fp_before_anti_outlier",
            "candidate": "fp_after_anti_outlier",
            "quantization_run": False,
        },
        "checks": {
            "shape_matches": shape_matches,
            "fp_logits_finite": fp_finite,
            "anti_outlier_logits_finite": candidate_finite,
            "elementwise_tolerance_passed": elementwise_tolerance_passed,
            "cosine_diagnostic_passed": cosine_diagnostic_passed,
            "top1_consistent": top1_consistent,
            "top_k_overlap_passed": top_k_overlap_passed,
            "js_divergence_passed": js_divergence_passed,
        },
        "shapes": {
            "fp_logits": list(fp_logits.shape),
            "anti_outlier_logits": list(anti_outlier_logits.shape),
        },
        "thresholds": {
            "atol": atol,
            "rtol": rtol,
            "max_abs_error_limit": error_limit,
            "min_cosine_similarity": min_cosine_similarity,
            "raw_logits_are_diagnostic_only": True,
            "top_k": top_k,
            "min_top_k_overlap": min_top_k_overlap,
            "max_js_divergence": max_js_divergence,
            "js_divergence_is_diagnostic_only": True,
            "top1_margin_tolerance": top1_margin_tolerance,
        },
        "metrics": metrics,
        "error_quantiles": error_quantiles,
        "distribution": {
            "logits_scope": logits_scope,
            "metrics": distribution_metrics,
            "top_k_samples": top_k_output,
        },
    }


def _positive_finite(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def _cosine_threshold(value: str) -> float:
    parsed = float(value)
    if not -1.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between -1 and 1")
    return parsed


def _unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _failed_comparison(
    algorithm: str,
    atol: float,
    rtol: float,
    min_cosine_similarity: float,
    error: str,
    *,
    top_k: int,
    min_top_k_overlap: float,
    max_js_divergence: float,
    top1_margin_tolerance: float,
) -> dict[str, Any]:
    return {
        "schema": "msagent.anti_outlier_final_logits_comparison/v1",
        "gate_policy": "last_token_token_consistency/v1",
        "passed": False,
        "algorithm": algorithm,
        "error": error,
        "comparison": {
            "baseline": "fp_before_anti_outlier",
            "candidate": "fp_after_anti_outlier",
            "quantization_run": False,
        },
        "checks": {"provenance_valid": False},
        "shapes": {},
        "thresholds": {
            "atol": atol,
            "rtol": rtol,
            "max_abs_error_limit": None,
            "min_cosine_similarity": min_cosine_similarity,
            "raw_logits_are_diagnostic_only": True,
            "top_k": top_k,
            "min_top_k_overlap": min_top_k_overlap,
            "max_js_divergence": max_js_divergence,
            "js_divergence_is_diagnostic_only": True,
            "top1_margin_tolerance": top1_margin_tolerance,
        },
        "metrics": {
            "max_abs_error": None,
            "mean_abs_error": None,
            "cosine_similarity": None,
        },
        "error_quantiles": {"absolute": {}, "relative": {}},
        "distribution": {"metrics": {}, "top_k_samples": []},
    }


def _update_run_record(
    record_path: Path,
    record: dict[str, Any],
    comparison_path: Path,
    passed: bool,
    error: str | None = None,
) -> None:
    if (
        record.get("status") not in ("PASS", "COMPARISON_FAILED")
        or record.get("processor_status") != "PASS"
    ):
        return
    status = "PASS" if passed else "COMPARISON_FAILED"
    record["comparison_status"] = status
    record["comparison_artifact"] = str(comparison_path.expanduser().resolve())
    if passed:
        if (
            isinstance(record.get("failure"), Mapping)
            and record["failure"].get("stage") == "comparison"
        ):
            record.pop("failure", None)
        record["status"] = "PASS"
    else:
        record["status"] = "COMPARISON_FAILED"
        record["failure"] = {
            "stage": "comparison",
            "error_type": "ComparisonError",
            "error": error or "final logits comparison failed",
        }
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare float-model logits before and after anti-outlier processing."
    )
    parser.add_argument(
        "--algorithm",
        required=True,
        help="Anti-outlier algorithm applied to the candidate model.",
    )
    parser.add_argument("--fp-logits", type=Path, required=True)
    parser.add_argument("--anti-outlier-logits", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=_positive_finite, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=_positive_finite, default=DEFAULT_RTOL)
    parser.add_argument(
        "--min-cosine-similarity",
        type=_cosine_threshold,
        default=DEFAULT_MIN_COSINE,
        help="Diagnostic raw-logits cosine threshold; it does not gate PASS/FAIL.",
    )
    parser.add_argument("--top-k", type=_positive_integer, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--min-top-k-overlap",
        type=_unit_interval,
        default=DEFAULT_MIN_TOP_K_OVERLAP,
    )
    parser.add_argument(
        "--max-js-divergence",
        type=_positive_finite,
        default=DEFAULT_MAX_JS_DIVERGENCE,
        help="Diagnostic JS-divergence threshold; it does not gate PASS/FAIL.",
    )
    parser.add_argument(
        "--top1-margin-tolerance",
        type=_positive_finite,
        default=DEFAULT_TOP1_MARGIN_TOLERANCE,
    )
    parser.add_argument(
        "--threshold-reason",
        help="Required when any default comparison threshold is changed.",
    )
    args = parser.parse_args()

    thresholds_changed = (
        args.atol != DEFAULT_ATOL
        or args.rtol != DEFAULT_RTOL
        or args.min_cosine_similarity != DEFAULT_MIN_COSINE
        or args.top_k != DEFAULT_TOP_K
        or args.min_top_k_overlap != DEFAULT_MIN_TOP_K_OVERLAP
        or args.max_js_divergence != DEFAULT_MAX_JS_DIVERGENCE
        or args.top1_margin_tolerance != DEFAULT_TOP1_MARGIN_TOLERANCE
    )
    if thresholds_changed and not args.threshold_reason:
        parser.error("--threshold-reason is required when thresholds are changed")

    record_path = args.run_record.expanduser().resolve()
    record: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    try:
        record = _load_run_record(record_path)
        provenance = validate_run_provenance(
            record,
            record_path,
            args.algorithm,
            args.fp_logits,
            args.anti_outlier_logits,
        )
        fp_logits = _load_logits(args.fp_logits)
        anti_outlier_logits = _load_logits(args.anti_outlier_logits)
        result = compare_logits(
            fp_logits,
            anti_outlier_logits,
            atol=args.atol,
            rtol=args.rtol,
            min_cosine_similarity=args.min_cosine_similarity,
            logits_scope=provenance["patch"]["logits_scope"],
            top_k=args.top_k,
            min_top_k_overlap=args.min_top_k_overlap,
            max_js_divergence=args.max_js_divergence,
            top1_margin_tolerance=args.top1_margin_tolerance,
        )
        result["checks"]["provenance_valid"] = True
        result["provenance"] = provenance
    except (OSError, ValueError) as exc:
        result = _failed_comparison(
            args.algorithm,
            args.atol,
            args.rtol,
            args.min_cosine_similarity,
            str(exc),
            top_k=args.top_k,
            min_top_k_overlap=args.min_top_k_overlap,
            max_js_divergence=args.max_js_divergence,
            top1_margin_tolerance=args.top1_margin_tolerance,
        )
        if provenance is not None:
            result["provenance"] = provenance

    result["inputs"] = {
        "fp_logits": str(args.fp_logits),
        "anti_outlier_logits": str(args.anti_outlier_logits),
    }
    result["algorithm"] = args.algorithm
    result["threshold_reason"] = args.threshold_reason
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if record is not None:
        _update_run_record(
            record_path,
            record,
            args.output,
            bool(result["passed"]),
            result.get("error"),
        )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
