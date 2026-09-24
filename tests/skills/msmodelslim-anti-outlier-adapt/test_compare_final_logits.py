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

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "skills/quantizer/msmodelslim-anti-outlier-adapt/scripts/compare_final_logits.py"
)
SPEC = importlib.util.spec_from_file_location("compare_final_logits", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_identical_float_logits_pass_without_quantization():
    logits = np.array([[[0.1, -0.2, 3.0]]], dtype=np.float32)

    result = MODULE.compare_logits(
        logits,
        logits.copy(),
        atol=1e-5,
        rtol=1e-4,
        min_cosine_similarity=0.99999,
    )

    assert result["passed"] is True
    assert result["comparison"] == {
        "baseline": "fp_before_anti_outlier",
        "candidate": "fp_after_anti_outlier",
        "quantization_run": False,
    }
    assert result["metrics"]["max_abs_error"] == 0.0


def test_large_change_after_anti_outlier_fails():
    before = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    after = np.array([[1.0, 2.0, 2.0]], dtype=np.float32)

    result = MODULE.compare_logits(
        before,
        after,
        atol=1e-5,
        rtol=1e-4,
        min_cosine_similarity=0.99999,
    )

    assert result["passed"] is False
    assert result["metrics"]["max_abs_error"] == 1.0


def test_raw_logits_shift_is_diagnostic_when_distribution_is_unchanged():
    before = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float64)
    after = before + 100.0

    result = MODULE.compare_logits(
        before,
        after,
        atol=1e-5,
        rtol=1e-4,
        min_cosine_similarity=0.99999,
    )

    assert result["passed"] is True
    assert result["checks"]["elementwise_tolerance_passed"] is False
    assert result["checks"]["cosine_diagnostic_passed"] is False
    assert result["checks"]["top1_consistent"] is True
    assert result["checks"]["top_k_overlap_passed"] is True
    assert result["checks"]["js_divergence_passed"] is True


def test_full_sequence_gate_compares_only_last_token_distribution():
    before = np.array([[[100.0, -100.0, 0.0], [1.0, 2.0, 3.0]]])
    after = np.array([[[-100.0, 100.0, 0.0], [1.0, 2.0, 3.0]]])

    result = MODULE.compare_logits(
        before,
        after,
        atol=1e-5,
        rtol=1e-4,
        min_cosine_similarity=0.99999,
        logits_scope="full_sequence",
    )

    assert result["passed"] is True
    assert result["distribution"]["metrics"]["rows"] == 1
    assert result["distribution"]["top_k_samples"][0]["baseline"][0]["token_id"] == 2


def test_distribution_gate_rejects_top_rank_and_probability_drift():
    before = np.array([[10.0, 9.0, 8.0, 7.0, 6.0, -6.0, -7.0, -8.0, -9.0, -10.0]])
    after = np.array([[-10.0, -9.0, -8.0, -7.0, -6.0, 6.0, 7.0, 8.0, 9.0, 10.0]])

    result = MODULE.compare_logits(
        before,
        after,
        atol=1e-5,
        rtol=1e-4,
        min_cosine_similarity=0.99999,
    )

    assert result["passed"] is False
    assert result["checks"]["top1_consistent"] is False
    assert result["checks"]["top_k_overlap_passed"] is False
    assert result["checks"]["js_divergence_passed"] is False


def test_js_divergence_is_diagnostic_when_token_ranking_is_consistent():
    before = np.array([[10.0, 9.0, 8.0, 7.0, 6.0, -6.0, -7.0, -8.0, -9.0, -10.0]])
    after = np.array([[0.10, 0.09, 0.08, 0.07, 0.06, -0.06, -0.07, -0.08, -0.09, -0.10]])

    result = MODULE.compare_logits(
        before,
        after,
        atol=1e-5,
        rtol=1e-4,
        min_cosine_similarity=0.99999,
    )

    assert result["passed"] is True
    assert result["gate_policy"] == "last_token_token_consistency/v1"
    assert result["checks"]["top1_consistent"] is True
    assert result["checks"]["top_k_overlap_passed"] is True
    assert result["checks"]["js_divergence_passed"] is False
    assert result["thresholds"]["js_divergence_is_diagnostic_only"] is True


def test_shape_mismatch_fails_with_empty_metrics():
    result = MODULE.compare_logits(
        np.zeros((1, 2)),
        np.zeros((2, 1)),
        atol=1e-5,
        rtol=1e-4,
        min_cosine_similarity=0.99999,
    )

    assert result["passed"] is False
    assert result["checks"]["shape_matches"] is False
    assert result["metrics"]["cosine_similarity"] is None


def test_cli_records_the_independently_validated_algorithm(tmp_path, monkeypatch):
    before_path = tmp_path / "before.npy"
    after_path = tmp_path / "after.npy"
    output_path = tmp_path / "final_logits_comparison.quarot.json"
    patch_path = tmp_path / "final_logits_capture.py"
    patch_path.write_text("capture patch", encoding="utf-8")
    patch_sha256 = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    checkpoint_identity = "sha256:test-checkpoint"
    input_summary = {"type": "ndarray", "shape": [1, 3], "dtype": "float32"}
    interface_validation_path = tmp_path / "interface_validation.QuaRotInterface.json"
    interface_validation_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "interface": "QuaRotInterface",
                "checkpoint_identity": checkpoint_identity,
            }
        ),
        encoding="utf-8",
    )
    interface_validation_sha256 = hashlib.sha256(interface_validation_path.read_bytes()).hexdigest()
    metadata = {
        "schema": "msagent.anti_outlier_logits_capture_patch/v1",
        "model_type": "fake_model",
        "adapter_class": "fake:Adapter",
        "checkpoint_identity": checkpoint_identity,
        "logits_scope": "full_sequence",
    }
    validation_path = tmp_path / "final_logits_patch_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema": "msagent.anti_outlier_logits_capture_patch_validation/v1",
                "status": "PASS",
                "patch_path": str(patch_path.resolve()),
                "patch_sha256": patch_sha256,
                "metadata": metadata,
                "checkpoint_identity": checkpoint_identity,
                "input_summary": input_summary,
                "checks": {
                    "outputs_finite": True,
                    "shape_matches": True,
                    "dtype_matches": True,
                    "repeatable": True,
                },
            }
        ),
        encoding="utf-8",
    )
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.save(before_path, logits)
    np.save(after_path, logits)
    validation_sha256 = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    run_record_path = tmp_path / "anti_outlier_run.quarot.json"
    run_record_path.write_text(
        json.dumps(
            {
                "schema": "msagent.anti_outlier_single_run/v1",
                "status": "PASS",
                "algorithm": "quarot",
                "model_type": "fake_model",
                "adapter_class": "fake:Adapter",
                "checkpoint_identity": checkpoint_identity,
                "config_source": "official:test",
                "processor_config": {"type": "quarot"},
                "practice": {"spec": {"process": [{"type": "quarot"}], "save": []}},
                "runner_type": "model_wise",
                "processor_status": "PASS",
                "input_summary": input_summary,
                "interface_validation": {
                    "path": str(interface_validation_path.resolve()),
                    "sha256": interface_validation_sha256,
                    "status": "PASS",
                    "interface": "QuaRotInterface",
                },
                "quantization_run": False,
                "patch": {
                    "path": str(patch_path.resolve()),
                    "sha256": patch_sha256,
                    "metadata": metadata,
                    "validation_artifact": str(validation_path.resolve()),
                    "validation_sha256": validation_sha256,
                },
                "artifacts": {
                    "before_logits": str(before_path.resolve()),
                    "after_logits": str(after_path.resolve()),
                },
                "artifact_sha256": {
                    "before_logits": hashlib.sha256(before_path.read_bytes()).hexdigest(),
                    "after_logits": hashlib.sha256(after_path.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--algorithm",
            "quarot",
            "--fp-logits",
            str(before_path),
            "--anti-outlier-logits",
            str(after_path),
            "--run-record",
            str(run_record_path),
            "--output",
            str(output_path),
        ],
    )

    assert MODULE.main() == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["algorithm"] == "quarot"
    assert result["comparison"]["quantization_run"] is False
    assert result["checks"]["provenance_valid"] is True


def test_cli_rejects_missing_run_provenance(tmp_path, monkeypatch):
    before_path = tmp_path / "before.npy"
    after_path = tmp_path / "after.npy"
    output_path = tmp_path / "comparison.json"
    run_record_path = tmp_path / "run.json"
    logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    np.save(before_path, logits)
    np.save(after_path, logits)
    run_record_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--algorithm",
            "quarot",
            "--fp-logits",
            str(before_path),
            "--anti-outlier-logits",
            str(after_path),
            "--run-record",
            str(run_record_path),
            "--output",
            str(output_path),
        ],
    )

    assert MODULE.main() == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["passed"] is False
    assert result["checks"]["provenance_valid"] is False
    record = json.loads(run_record_path.read_text(encoding="utf-8"))
    assert record == {}


def test_provenance_rejects_checkpoint_unbound_interface_evidence(tmp_path):
    interface_path = tmp_path / "interface.json"
    interface_path.write_text(
        json.dumps({"status": "PASS", "interface": "QuaRotInterface"}),
        encoding="utf-8",
    )
    record = {
        "schema": MODULE.RUN_SCHEMA,
        "status": "PASS",
        "processor_status": "PASS",
        "algorithm": "quarot",
        "quantization_run": False,
        "checkpoint_identity": "sha256:current",
        "input_summary": {},
        "interface_validation": {
            "path": str(interface_path),
            "sha256": hashlib.sha256(interface_path.read_bytes()).hexdigest(),
        },
    }

    try:
        MODULE.validate_run_provenance(
            record,
            tmp_path / "run.json",
            "quarot",
            tmp_path / "before.npy",
            tmp_path / "after.npy",
        )
    except ValueError as exc:
        assert "interface validation checkpoint identity does not match" in str(exc)
    else:
        raise AssertionError("checkpoint-unbound interface evidence must be rejected")


def test_comparison_does_not_overwrite_processor_failure(tmp_path, monkeypatch):
    before_path = tmp_path / "before.npy"
    after_path = tmp_path / "after.npy"
    output_path = tmp_path / "comparison.json"
    run_record_path = tmp_path / "run.json"
    np.save(before_path, np.zeros(1, dtype=np.float32))
    np.save(after_path, np.zeros(1, dtype=np.float32))
    original = {
        "schema": MODULE.RUN_SCHEMA,
        "status": "PROCESSOR_FAILED",
        "algorithm": "quarot",
        "failure": {"stage": "processor", "error": "original failure"},
    }
    run_record_path.write_text(json.dumps(original), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--algorithm",
            "quarot",
            "--fp-logits",
            str(before_path),
            "--anti-outlier-logits",
            str(after_path),
            "--run-record",
            str(run_record_path),
            "--output",
            str(output_path),
        ],
    )

    assert MODULE.main() == 1
    assert json.loads(run_record_path.read_text(encoding="utf-8")) == original
