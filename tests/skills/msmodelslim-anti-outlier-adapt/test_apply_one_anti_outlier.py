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
import inspect
import json
from pathlib import Path
import sys
from contextlib import nullcontext

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "skills/quantizer/msmodelslim-anti-outlier-adapt/scripts/apply_one_anti_outlier.py"
)
SPEC = importlib.util.spec_from_file_location("apply_one_anti_outlier", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeModel:
    offset = 0

    def eval(self):
        return self


class FakeAdapter:
    pass


class LayerWiseFakeAdapter:
    def get_layer_wise_offload_device(self):
        return "cpu"


class FakeBFloat16Tensor:
    dtype = "torch.bfloat16"

    def detach(self):
        return self

    def float(self):
        return np.array([1.0, 2.0], dtype=np.float32)


def test_bfloat16_logits_are_converted_to_float32():
    result = MODULE._as_numpy(FakeBFloat16Tensor())
    assert result.dtype == np.float32
    assert np.array_equal(result, [1.0, 2.0])


def test_capture_context_uses_no_grad_not_inference_mode(monkeypatch):
    calls = []

    class FakeTorch:
        @staticmethod
        def no_grad():
            calls.append("no_grad")
            return nullcontext()

        @staticmethod
        def inference_mode():
            raise AssertionError("capture must not create persistent inference tensors")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    with MODULE._inference_context():
        pass

    assert calls == ["no_grad"]


def test_public_api_has_only_four_required_arguments():
    parameters = inspect.signature(MODULE.apply_one_anti_outlier_and_record_logits).parameters.values()
    required = [parameter.name for parameter in parameters if parameter.default is inspect.Parameter.empty]
    assert required == ["algorithm", "model_path", "fixed_input", "output_dir"]


def test_rng_is_reset_before_each_logits_capture(tmp_path):
    samples = []

    def capture(model, inputs):
        samples.append(np.random.random())
        return np.zeros(1)

    MODULE._apply_one_anti_outlier_and_record_logits(
        "quarot",
        "/model",
        None,
        tmp_path,
        load_model_and_adapter=lambda path: (FakeModel(), object()),
        capture_final_logits=capture,
        apply_processor=lambda model, adapter, config: np.random.random(20),
        processor_config={"type": "quarot"},
        config_source="official:test",
        seed=123,
    )

    assert samples[0] == samples[1]


def test_apply_one_reloads_applies_and_records(tmp_path):
    loads = []

    def load(path):
        loads.append(path)
        return FakeModel(), object()

    result = MODULE._apply_one_anti_outlier_and_record_logits(
        "quarot",
        "/model",
        np.array([1.0, 2.0]),
        tmp_path,
        load_model_and_adapter=load,
        capture_final_logits=lambda model, inputs: inputs + model.offset,
        apply_processor=lambda model, adapter, config: setattr(model, "offset", 0.25),
        processor_config={"type": "quarot"},
        config_source="official:test",
    )

    assert loads == ["/model"]
    assert np.allclose(np.load(result["artifacts"]["before_logits"]), [1.0, 2.0])
    assert np.allclose(np.load(result["artifacts"]["after_logits"]), [1.25, 2.25])
    record = json.loads(Path(result["artifacts"]["run_record"]).read_text(encoding="utf-8"))
    assert record["algorithm"] == "quarot"
    assert record["quantization_run"] is False
    assert record["runner_type"] == "model_wise"


def test_rejects_mismatched_processor_config(tmp_path):
    try:
        MODULE._apply_one_anti_outlier_and_record_logits(
            "quarot",
            "/model",
            None,
            tmp_path,
            load_model_and_adapter=lambda path: (FakeModel(), object()),
            capture_final_logits=lambda model, inputs: np.zeros(1),
            apply_processor=lambda model, adapter, config: None,
            processor_config={"type": "iter_smooth"},
            config_source="official:test",
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched processor config should fail")


def test_generated_patch_has_exact_target_and_signature(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "fake_model"}', encoding="utf-8")
    identity = MODULE.checkpoint_identity(model_path)
    metadata = {
        "schema": "msagent.anti_outlier_logits_capture_patch/v1",
        "model_type": "fake_model",
        "adapter_class": f"{FakeAdapter.__module__}:{FakeAdapter.__qualname__}",
        "checkpoint_identity": identity,
        "logits_scope": "full_sequence",
    }
    patch_path = tmp_path / "capture.py"
    patch_path.write_text(
        f"PATCH_METADATA = {metadata!r}\ndef capture_final_logits(self, model, inputs, device):\n    return inputs\n",
        encoding="utf-8",
    )

    generated = MODULE.load_generated_patch(patch_path)
    assert (
        MODULE.validate_patch_target(
            generated.capture,
            FakeAdapter(),
            "fake_model",
            identity,
            metadata=generated.metadata,
        )
        == metadata
    )


def test_patch_metadata_records_dtype_cast_boundaries(tmp_path):
    metadata = {
        "schema": MODULE.PATCH_SCHEMA,
        "model_type": "fake_model",
        "adapter_class": f"{FakeAdapter.__module__}:{FakeAdapter.__qualname__}",
        "checkpoint_identity": "sha256:fixed",
        "logits_scope": "last_token",
        "dtype_casts": [{"boundary": "norm -> q_proj", "from": "float32", "to": "bfloat16"}],
    }

    def capture_final_logits(self, model, inputs, device):
        return inputs

    assert (
        MODULE.validate_patch_target(
            capture_final_logits, FakeAdapter(), "fake_model", "sha256:fixed", metadata=metadata
        )["dtype_casts"]
        == metadata["dtype_casts"]
    )
    metadata["dtype_casts"][0].pop("to")
    with pytest.raises(MODULE.PatchValidationError, match="dtype_casts"):
        MODULE.validate_patch_target(
            capture_final_logits, FakeAdapter(), "fake_model", "sha256:fixed", metadata=metadata
        )


def test_generated_patch_cannot_reload_checkpoint_weights(tmp_path):
    patch_path = tmp_path / "capture.py"
    patch_path.write_text(
        "import torch\n"
        "PATCH_METADATA = {}\n"
        "def capture_final_logits(self, model, inputs, device):\n"
        "    return torch.load('checkpoint.bin')\n",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.PatchUnsupportedError, match="torch.load"):
        MODULE.load_generated_patch(patch_path)


def test_generated_patch_cannot_create_persistent_inference_tensors(tmp_path):
    patch_path = tmp_path / "capture.py"
    patch_path.write_text(
        "import torch\n"
        "PATCH_METADATA = {}\n"
        "def capture_final_logits(self, model, inputs, device):\n"
        "    with torch.inference_mode():\n"
        "        return inputs\n",
        encoding="utf-8",
    )

    with pytest.raises(MODULE.PatchUnsupportedError, match="inference_mode"):
        MODULE.load_generated_patch(patch_path)


def test_auto_runner_preserves_layer_wise_semantics():
    assert MODULE._select_runner("auto", FakeAdapter()) == "layer_wise"
    assert MODULE._select_runner("auto", LayerWiseFakeAdapter()) == "layer_wise"
    assert MODULE._select_runner("model_wise", FakeAdapter()) == "model_wise"


def test_patch_behavior_validator_passes_device_to_bound_patch():
    devices = []

    def capture(model, inputs, device):
        devices.append(device)
        return inputs

    result = MODULE.validate_patch_behavior(capture, FakeModel(), np.array([1.0, 2.0]), "cpu", seed=7)

    assert result["status"] == "PASS"
    assert result["checks"]["repeatable"] is True
    assert devices == ["cpu", "cpu"]


def test_patch_behavior_matches_native_reference_at_last_token_scope():
    class NativeModel(FakeModel):
        def __call__(self, inputs):
            return {"logits": inputs}

    full_logits = np.arange(24, dtype=np.float32).reshape(2, 4, 3)

    result = MODULE.validate_patch_behavior(
        lambda model, inputs, device: inputs[..., -1, :],
        NativeModel(),
        full_logits,
        "cpu",
        seed=7,
        reference_capture=MODULE._capture_native_forward,
        logits_scope="last_token",
    )

    assert result["status"] == "PASS"
    assert result["checks"]["reference_matches"] is True
    assert result["outputs"]["reference"]["shape"] == [2, 3]


def test_interface_validation_requires_current_checkpoint(tmp_path):
    validation_path = tmp_path / "interface.json"
    validation_path.write_text(
        json.dumps({"status": "PASS", "interface": "QuaRotInterface"}),
        encoding="utf-8",
    )

    with pytest.raises(MODULE.PatchUnsupportedError, match="checkpoint identity"):
        MODULE._load_interface_validation(validation_path, "QuaRotInterface", "sha256:current")

    validation_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "interface": "QuaRotInterface",
                "checkpoint_identity": "sha256:current",
            }
        ),
        encoding="utf-8",
    )
    evidence = MODULE._load_interface_validation(validation_path, "QuaRotInterface", "sha256:current")
    assert evidence["checkpoint_identity"] == "sha256:current"


def test_missing_patch_records_unsupported_run(tmp_path):
    output_dir = tmp_path / "output"
    with pytest.raises(MODULE.PatchUnsupportedError):
        MODULE.apply_one_anti_outlier_and_record_logits(
            "quarot",
            tmp_path / "model",
            None,
            output_dir,
        )

    record = json.loads((output_dir / "anti_outlier_run.quarot.json").read_text(encoding="utf-8"))
    validation = json.loads((output_dir / "final_logits_patch_validation.quarot.json").read_text(encoding="utf-8"))
    assert record["status"] == "PATCH_UNSUPPORTED"
    assert validation["status"] == "UNSUPPORTED"


def test_missing_interface_evidence_records_preflight_failure(tmp_path):
    output_dir = tmp_path / "output"
    patch_path = tmp_path / "capture.py"
    patch_path.write_text("PATCH_METADATA = {}\n", encoding="utf-8")
    with pytest.raises(MODULE.InterfaceValidationError, match="interface-validation"):
        MODULE.apply_one_anti_outlier_and_record_logits(
            "iter_smooth",
            tmp_path / "model",
            None,
            output_dir,
            logits_capture_patch=patch_path,
        )
    record = json.loads((output_dir / "anti_outlier_run.iter_smooth.json").read_text(encoding="utf-8"))
    assert record["status"] == "INTERFACE_VALIDATION_FAILED"
    assert record["failure"]["stage"] == "interface_validation"


def test_processor_failure_is_persisted(tmp_path):
    with pytest.raises(RuntimeError, match="processor failed"):
        MODULE._apply_one_anti_outlier_and_record_logits(
            "quarot",
            "/model",
            np.array([1.0]),
            tmp_path,
            load_model_and_adapter=lambda path: (FakeModel(), FakeAdapter()),
            capture_final_logits=lambda model, inputs: inputs,
            apply_processor=lambda model, adapter, config: (_ for _ in ()).throw(RuntimeError("processor failed")),
            processor_config={"type": "quarot"},
            config_source="official:test",
        )

    record = json.loads((tmp_path / "anti_outlier_run.quarot.json").read_text(encoding="utf-8"))
    assert record["status"] == "PROCESSOR_FAILED"
    assert record["failure"]["stage"] == "processor"


def test_before_capture_failure_is_persisted(tmp_path):
    with pytest.raises(RuntimeError, match="before failed"):
        MODULE._apply_one_anti_outlier_and_record_logits(
            "quarot",
            "/model",
            None,
            tmp_path,
            load_model_and_adapter=lambda path: (FakeModel(), FakeAdapter()),
            capture_final_logits=lambda model, inputs: (_ for _ in ()).throw(RuntimeError("before failed")),
            apply_processor=lambda model, adapter, config: None,
            processor_config={"type": "quarot"},
            config_source="official:test",
        )

    record = json.loads((tmp_path / "anti_outlier_run.quarot.json").read_text(encoding="utf-8"))
    assert record["status"] == "BEFORE_CAPTURE_FAILED"
    assert record["failure"]["stage"] == "before_capture"


def test_after_capture_failure_is_persisted(tmp_path):
    calls = 0

    def capture(model, inputs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("after failed")
        return np.zeros(1)

    with pytest.raises(RuntimeError, match="after failed"):
        MODULE._apply_one_anti_outlier_and_record_logits(
            "quarot",
            "/model",
            None,
            tmp_path,
            load_model_and_adapter=lambda path: (FakeModel(), FakeAdapter()),
            capture_final_logits=capture,
            apply_processor=lambda model, adapter, config: None,
            processor_config={"type": "quarot"},
            config_source="official:test",
        )

    record = json.loads((tmp_path / "anti_outlier_run.quarot.json").read_text(encoding="utf-8"))
    assert record["status"] == "AFTER_CAPTURE_FAILED"
    assert record["failure"]["stage"] == "after_capture"
