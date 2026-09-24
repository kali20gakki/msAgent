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
"""Apply exactly one msModelSlim anti-outlier processor and record final logits.

The model is loaded through msModelSlim's registered ``PluginModelFactory``.
The adapter's ``handle_dataset`` prepares a fixed input, and an Agent-generated
model-specific patch owns the complete final-logits capture path. No generic
norm/head inference or model-weight save step is used.

The processor is applied through the official ``modelslim_v1`` practice path:
the script builds a minimal practice document (a single anti-outlier process
entry and an empty ``save``), validates it with ``PracticeConfig``, and runs it
through the same runner selection as ``msmodelslim quant``.  This keeps the
transform identical to production quantization while never emitting quantized
weights.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import random
from contextlib import nullcontext
from pathlib import Path
from types import MethodType
from typing import Any, Callable, Mapping, NamedTuple
from unittest.mock import patch as mock_patch

import numpy as np
import yaml


DEFAULT_CONFIG_NAMES = {
    "quarot": "quarot_default",
    "flex_smooth_quant": "flex_smooth_quant_default",
    "flex_awq_ssz": "flex_awq_ssz_default",
    "iter_smooth": "iter_smooth_default",
}
ALGORITHM_INTERFACE_NAMES = {
    "quarot": "QuaRotInterface",
    "flex_smooth_quant": "FlexSmoothQuantInterface",
    "flex_awq_ssz": "FlexSmoothQuantInterface",
    "iter_smooth": "IterSmoothInterface",
}

PATCH_SCHEMA = "msagent.anti_outlier_logits_capture_patch/v1"
PATCH_VALIDATION_SCHEMA = "msagent.anti_outlier_logits_capture_patch_validation/v1"
RUN_SCHEMA = "msagent.anti_outlier_single_run/v1"
LOGITS_SCOPES = {"full_sequence", "last_token"}


class PatchUnsupportedError(RuntimeError):
    """The supplied patch cannot be trusted for this model instance."""


class PatchValidationError(RuntimeError):
    """The supplied patch failed syntax, import, or behavioral validation."""


class InterfaceValidationError(PatchUnsupportedError):
    """The installed Adapter is not ready for the selected processor."""


class GeneratedPatch(NamedTuple):
    path: Path
    capture: Callable[..., Any]
    metadata: dict[str, Any]
    sha256: str
    source: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _qualified_class_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}:{cls.__qualname__}"


def checkpoint_identity(model_path: str | Path) -> str:
    """Return a stable identity for a local checkpoint directory.

    The path, config content, and checkpoint file hashes distinguish the
    checkpoint used to generate a model-specific capture patch.
    """
    resolved = Path(model_path).expanduser().resolve()
    config_path = resolved / "config.json"
    config_sha256 = _sha256_file(config_path) if config_path.is_file() else "missing"
    weight_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
    files = [
        {
            "path": str(item.relative_to(resolved)),
            "sha256": _sha256_file(item),
        }
        for item in sorted(resolved.rglob("*"))
        if item.is_file() and item.suffix in weight_suffixes
    ]
    payload = {
        "path": str(resolved),
        "config_sha256": config_sha256,
        "weight_files": files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _patch_call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        if (
            isinstance(node.func.value, ast.Name)
            and node.func.value.id == "torch"
            and node.func.attr == "load"
        ):
            return "torch.load"
        return node.func.attr
    return None


def _validate_patch_source(source: str, path: Path) -> None:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise PatchValidationError(f"patch has invalid Python syntax: {exc}") from exc

    forbidden_calls = {
        "init_model",
        "from_pretrained",
        "load_state_dict",
        "torch.load",
        "load_file",
        "safe_open",
        "inference_mode",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _patch_call_name(node) in forbidden_calls:
            raise PatchUnsupportedError(
                f"patch must not call {_patch_call_name(node)} or reload checkpoint weights"
            )


def load_generated_patch(patch_path: str | Path) -> GeneratedPatch:
    """Load a model-specific capture patch without installing it globally."""
    path = Path(patch_path).expanduser().resolve()
    if not path.is_file():
        raise PatchUnsupportedError(f"logits capture patch does not exist: {path}")
    source = path.read_text(encoding="utf-8")
    _validate_patch_source(source, path)
    try:
        compile(source, str(path), "exec")
    except SyntaxError as exc:
        raise PatchValidationError(f"patch failed compilation: {exc}") from exc

    module_name = (
        f"_msagent_logits_capture_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PatchValidationError(f"cannot import generated patch: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise PatchValidationError(f"generated patch import failed: {exc}") from exc

    metadata = getattr(module, "PATCH_METADATA", None)
    capture = getattr(module, "capture_final_logits", None)
    if not isinstance(metadata, Mapping):
        raise PatchValidationError("patch must export mapping PATCH_METADATA")
    if not callable(capture):
        raise PatchValidationError("patch must export callable capture_final_logits")
    try:
        setattr(capture, "PATCH_METADATA", dict(metadata))
    except (AttributeError, TypeError):
        pass
    return GeneratedPatch(
        path=path,
        capture=capture,
        metadata=dict(metadata),
        sha256=_sha256_file(path),
        source=source,
    )


def validate_patch_target(
    capture_fn: Callable[..., Any],
    adapter: Any,
    model_type: str,
    checkpoint_identity_value: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate patch identity and its fixed unbound function signature."""
    if metadata is None:
        metadata = getattr(capture_fn, "PATCH_METADATA", None)
    if not isinstance(metadata, Mapping):
        raise PatchValidationError("patch metadata is missing")

    expected_adapter = _qualified_class_name(adapter)
    checks = {
        "schema": metadata.get("schema") == PATCH_SCHEMA,
        "model_type": metadata.get("model_type") == model_type,
        "adapter_class": metadata.get("adapter_class") == expected_adapter,
        "checkpoint_identity": metadata.get("checkpoint_identity")
        == checkpoint_identity_value,
        "logits_scope": metadata.get("logits_scope") in LOGITS_SCOPES,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise PatchUnsupportedError("patch target mismatch: " + ", ".join(failed))

    dtype_casts = metadata.get("dtype_casts", [])
    if not isinstance(dtype_casts, list) or any(
        not isinstance(item, Mapping)
        or not all(isinstance(item.get(key), str) and item[key] for key in ("boundary", "from", "to"))
        for item in dtype_casts
    ):
        raise PatchValidationError(
            "patch dtype_casts must list boundary/from/to strings"
        )

    try:
        parameters = list(inspect.signature(capture_fn).parameters.values())
    except (TypeError, ValueError) as exc:
        raise PatchValidationError(
            "capture_final_logits signature cannot be inspected"
        ) from exc
    expected_parameters = ["self", "model", "inputs", "device"]
    if [parameter.name for parameter in parameters] != expected_parameters:
        raise PatchValidationError(
            "capture_final_logits must have signature (self, model, inputs, device)"
        )
    if any(
        parameter.kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for parameter in parameters
    ):
        raise PatchValidationError("capture_final_logits parameters must be positional")
    if any(
        parameter.default is not inspect.Parameter.empty for parameter in parameters
    ):
        raise PatchValidationError(
            "capture_final_logits parameters must not have defaults"
        )
    return dict(metadata)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    dtype_name = str(getattr(value, "dtype", "")).lower()
    if "bfloat16" in dtype_name or "float8" in dtype_name:
        if not hasattr(value, "float"):
            raise TypeError(
                f"cannot convert unsupported floating dtype {dtype_name} to float32"
            )
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.number) or np.issubdtype(
        result.dtype, np.complexfloating
    ):
        raise TypeError("capture_final_logits must return a numeric tensor or array")
    return result


def _inference_context():
    try:
        import torch

        # Capture may lazily load decoder parameters or replace tensors on the
        # persistent model instance. ``inference_mode`` would make tensors
        # created in that path permanently lack version counters, which breaks
        # later processor/capture passes. ``no_grad`` avoids autograd storage
        # without changing tensor semantics across the three-pass lifecycle.
        return torch.no_grad()
    except ImportError:
        return nullcontext()


def _invoke_capture(
    capture_final_logits: Callable[..., Any],
    model: Any,
    inputs: Any,
    device: str | None = None,
) -> np.ndarray:
    with _inference_context():
        if device is None:
            value = capture_final_logits(model, inputs)
        else:
            value = capture_final_logits(model, inputs, device)
        return _as_numpy(value)


def _extract_reference_logits(outputs: Any) -> Any:
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, Mapping) and "logits" in outputs:
        return outputs["logits"]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise TypeError("model forward output does not contain reference logits")


def _capture_native_forward(model: Any, inputs: Any) -> np.ndarray:
    with _inference_context():
        if isinstance(inputs, Mapping):
            outputs = model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            outputs = model(*inputs)
        else:
            outputs = model(inputs)
    return _as_numpy(_extract_reference_logits(outputs))


def _apply_logits_scope(logits: np.ndarray, logits_scope: str) -> np.ndarray:
    if logits_scope == "full_sequence":
        return logits
    if logits_scope == "last_token":
        if logits.ndim < 2:
            raise PatchValidationError(
                "last_token reference logits must have at least two dimensions"
            )
        return logits[..., -1, :]
    raise PatchValidationError(f"unsupported logits scope: {logits_scope}")


def _summarize_input(value: Any, depth: int = 0) -> Any:
    if depth > 2:
        return {"type": type(value).__name__}
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "keys": sorted(str(key) for key in value),
            "values": {
                str(key): _summarize_input(item, depth + 1)
                for key, item in value.items()
            },
        }
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [_summarize_input(item, depth + 1) for item in value[:8]],
        }
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    summary: dict[str, Any] = {"type": type(value).__name__}
    if shape is not None:
        summary["shape"] = list(shape)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    return summary


def _describe_array(value: np.ndarray) -> dict[str, Any]:
    description: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
    }
    if value.size:
        description["min"] = float(np.min(value))
        description["max"] = float(np.max(value))
    else:
        description["min"] = None
        description["max"] = None
    return description


def validate_patch_behavior(
    capture_final_logits: Callable[..., Any],
    model: Any,
    fixed_input: Any,
    device: str,
    seed: int,
    reference_capture: Callable[[Any, Any], Any] | None = None,
    logits_scope: str = "full_sequence",
) -> dict[str, Any]:
    """Check that a model-specific patch is repeatable and numerically valid."""
    result: dict[str, Any] = {
        "schema": PATCH_VALIDATION_SCHEMA,
        "status": "FAIL",
        "seed": seed,
        "input_summary": _summarize_input(fixed_input),
        "reference_path": (
            f"model.forward:{logits_scope}"
            if reference_capture is not None
            else "not_available"
        ),
        "checks": {
            "outputs_finite": False,
            "shape_matches": False,
            "dtype_matches": False,
            "repeatable": False,
        },
        "outputs": {},
        "metrics": {},
    }
    try:
        if hasattr(model, "eval"):
            model.eval()
        _reset_rng(seed)
        first = _invoke_capture(capture_final_logits, model, fixed_input, device)
        _reset_rng(seed)
        second = _invoke_capture(capture_final_logits, model, fixed_input, device)
        shape_matches = first.shape == second.shape
        dtype_matches = first.dtype == second.dtype
        first_finite = bool(np.isfinite(first).all())
        second_finite = bool(np.isfinite(second).all())
        repeatable = bool(
            shape_matches and dtype_matches and np.array_equal(first, second)
        )
        result["checks"] = {
            "outputs_finite": first_finite and second_finite,
            "shape_matches": shape_matches,
            "dtype_matches": dtype_matches,
            "repeatable": repeatable,
        }
        result["outputs"] = {
            "first": _describe_array(first),
            "second": _describe_array(second),
        }
        if shape_matches and first_finite and second_finite:
            delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
            result["metrics"] = {
                "max_abs_error": float(delta.max(initial=0.0)),
                "mean_abs_error": float(delta.mean()) if delta.size else 0.0,
            }
        if reference_capture is not None:
            try:
                _reset_rng(seed)
                reference = _invoke_capture(reference_capture, model, fixed_input)
                reference = _apply_logits_scope(reference, logits_scope)
                reference_matches = bool(
                    reference.shape == first.shape
                    and np.isfinite(reference).all()
                    and np.allclose(reference, first, atol=1e-5, rtol=1e-4)
                )
                result["checks"]["reference_matches"] = reference_matches
                result["outputs"]["reference"] = _describe_array(reference)
                if not reference_matches:
                    raise PatchValidationError(
                        "patch output does not match the model's formal forward logits"
                    )
            except PatchValidationError:
                raise
            except Exception as exc:
                result["reference_path"] = "unavailable"
                result["reference_error"] = f"{type(exc).__name__}: {exc}"
        if not all(result["checks"].values()):
            raise PatchValidationError(
                "patch baseline output is not finite, shape/dtype stable, or repeatable"
            )
        result["status"] = "PASS"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _record_failure(
    record: dict[str, Any], status: str, stage: str, error: BaseException
) -> None:
    record["status"] = status
    record["failure"] = {
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "npu"):
            torch.npu.manual_seed_all(seed)
    except ImportError:
        pass


def _apply_one_anti_outlier_and_record_logits(
    algorithm: str,
    model_path: str | Path,
    fixed_input: Any,
    output_dir: str | Path,
    *,
    load_model_and_adapter: Callable[[str | Path], tuple[Any, Any]],
    capture_final_logits: Callable[[Any, Any], Any],
    apply_processor: Callable[[Any, Any, Mapping[str, Any]], None],
    processor_config: Mapping[str, Any],
    config_source: str,
    runner_type: str = "model_wise",
    practice: Mapping[str, Any] | None = None,
    seed: int = 42,
    checkpoint_identity_value: str | None = None,
    patch_provenance: Mapping[str, Any] | None = None,
    input_summary: Any = None,
    adapter_class: str | None = None,
    run_record: dict[str, Any] | None = None,
    interface_validation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run baseline forward, one processor, and post-transform forward.

    ``capture_final_logits`` is supplied by the model-specific patch. The
    processor's calibration forward is deliberately separate and is never used
    as either side of the logits comparison.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    record_path = destination / f"anti_outlier_run.{algorithm}.json"
    record = (
        run_record
        if run_record is not None
        else {
            "schema": RUN_SCHEMA,
            "status": "STARTED",
            "algorithm": algorithm,
            "model_path": str(model_path),
            "checkpoint_identity": checkpoint_identity_value,
            "config_source": config_source,
            "processor_config": dict(processor_config),
            "practice": dict(practice) if practice is not None else None,
            "runner_type": runner_type,
            "logits_seed": seed,
            "quantization_run": False,
            "input_summary": input_summary,
            "patch": (dict(patch_provenance) if patch_provenance is not None else None),
            "adapter_class": adapter_class,
            "interface_validation": (
                dict(interface_validation) if interface_validation is not None else None
            ),
            "artifacts": {"run_record": str(record_path)},
        }
    )
    record.update(
        {
            "schema": RUN_SCHEMA,
            "algorithm": algorithm,
            "model_path": str(model_path),
            "checkpoint_identity": checkpoint_identity_value,
            "config_source": config_source,
            "processor_config": dict(processor_config),
            "practice": dict(practice) if practice is not None else None,
            "runner_type": runner_type,
            "logits_seed": seed,
            "quantization_run": False,
            "input_summary": input_summary,
            "adapter_class": adapter_class,
            "interface_validation": (
                dict(interface_validation) if interface_validation is not None else None
            ),
        }
    )
    if patch_provenance is not None:
        record["patch"] = dict(patch_provenance)
    record.setdefault("artifacts", {})["run_record"] = str(record_path)
    _write_json(record_path, record)

    try:
        if algorithm not in DEFAULT_CONFIG_NAMES:
            raise ValueError(f"unsupported anti-outlier algorithm: {algorithm}")
        if processor_config.get("type") != algorithm:
            raise ValueError(
                "processor config type does not match the requested algorithm"
            )

        # Loading happens inside this function by design: callers cannot pass a
        # model that has already been changed by another algorithm.
        model, adapter = load_model_and_adapter(model_path)
        try:
            if hasattr(model, "eval"):
                model.eval()
            _reset_rng(seed)
            before = _invoke_capture(capture_final_logits, model, fixed_input)
            before_path = destination / f"final_logits.before.{algorithm}.npy"
            np.save(before_path, before)
        except Exception as exc:
            _record_failure(record, "BEFORE_CAPTURE_FAILED", "before_capture", exc)
            raise
        record["artifacts"]["before_logits"] = str(before_path)
        record.setdefault("artifact_sha256", {})["before_logits"] = _sha256_file(
            before_path
        )
        record["outputs"] = {"before": _describe_array(before)}
        _write_json(record_path, record)

        try:
            apply_processor(model, adapter, dict(processor_config))
        except Exception as exc:
            _record_failure(record, "PROCESSOR_FAILED", "processor", exc)
            raise
        record["processor_status"] = "PASS"
        _write_json(record_path, record)

        try:
            if hasattr(model, "eval"):
                model.eval()
            _reset_rng(seed)
            after = _invoke_capture(capture_final_logits, model, fixed_input)
            after_path = destination / f"final_logits.after.{algorithm}.npy"
            np.save(after_path, after)
        except Exception as exc:
            _record_failure(record, "AFTER_CAPTURE_FAILED", "after_capture", exc)
            raise
        record["artifacts"]["after_logits"] = str(after_path)
        record.setdefault("artifact_sha256", {})["after_logits"] = _sha256_file(
            after_path
        )
        record["outputs"]["after"] = _describe_array(after)
        record["comparison_status"] = "PENDING"
        record["status"] = "PASS"
        return record
    except Exception as exc:
        if record["status"] == "STARTED":
            _record_failure(record, "PROCESSOR_FAILED", "run", exc)
        raise
    finally:
        _write_json(record_path, record)


def _load_official_processor_entry(algorithm: str) -> tuple[dict[str, Any], str]:
    """Read one algorithm's default processor entry from msModelSlim's own config.

    The skill forbids copying default parameters into msAgent, so the bundled
    ``expert_experience.yaml`` remains the single source of truth.
    """
    from msmodelslim.core.tune_strategy.common.config_builder.expert_experience import (  # noqa: PLC0415
        expert_experience,
    )

    config_path = Path(expert_experience.__file__).with_name("expert_experience.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    name = DEFAULT_CONFIG_NAMES[algorithm]
    return dict(config["anti_outlier_strategies"][name]), f"{config_path}:{name}"


def _build_practice(
    algorithm: str,
    processor_entry: Mapping[str, Any],
    runner_type: str,
) -> dict[str, Any]:
    """Build a minimal, valid ``modelslim_v1`` practice document.

    The process list holds only the selected anti-outlier processor and ``save``
    is empty, so running this practice transforms the float model without
    emitting quantized weights (``quantization_run`` remains ``False``).
    """
    return {
        "apiversion": "modelslim_v1",
        "metadata": {
            "config_id": f"anti_outlier_{algorithm}",
            "score": 100.0,
            "verified_model_types": [],
            "label": {
                "w_bit": 8,
                "a_bit": 8,
                "is_sparse": False,
                "kv_cache": False,
            },
        },
        "spec": {
            "runner": runner_type,
            "process": [dict(processor_entry)],
            "save": [],
        },
    }


def _apply_with_msmodelslim(
    model: Any,
    adapter: Any,
    config: Mapping[str, Any],
    calib_data: list[Any],
    runner_type: str,
    device: str,
    seed: int = 42,
) -> None:
    # Importing msmodelslim.processor registers every AutoProcessorConfig
    # subclass, so ``type`` can dispatch to the right processor config.
    import msmodelslim.processor  # noqa: F401, PLC0415
    import torch  # noqa: PLC0415

    from msmodelslim.core.const import DeviceType, RunnerType  # noqa: PLC0415
    from msmodelslim.core.context.context_factory import ContextFactory  # noqa: PLC0415
    from msmodelslim.core.context.interface import ContextManager  # noqa: PLC0415
    from msmodelslim.core.practice.interface import PracticeConfig  # noqa: PLC0415
    from msmodelslim.core.quant_service.modelslim_v1.quant_config import (  # noqa: PLC0415
        ModelslimV1QuantConfig,
    )
    from msmodelslim.utils.seed import seed_all  # noqa: PLC0415

    practice = _build_practice(config["type"], config, runner_type)
    quant_config = ModelslimV1QuantConfig.from_base(
        PracticeConfig.model_validate(practice).extract_quant_config()
    )

    device_type = DeviceType(device)
    # Mirror ModelslimV1QuantService.quant_process: deterministic seed and
    # offline operator compilation on NPU.
    seed_all(seed=seed, mode=True)
    if device_type == DeviceType.NPU:
        torch.npu.set_compile_mode(jit_compile=False)

    spec_runner = quant_config.spec.runner
    if spec_runner == RunnerType.MODEL_WISE:
        from msmodelslim.core.runner.pipeline_parallel_runner import PPRunner  # noqa: PLC0415

        runner = PPRunner(adapter=adapter)
    elif spec_runner in (RunnerType.LAYER_WISE, RunnerType.AUTO):
        from msmodelslim.core.runner.layer_wise_runner import LayerWiseRunner  # noqa: PLC0415

        offload_device = "cpu"
        offload_getter = getattr(adapter, "get_layer_wise_offload_device", None)
        if callable(offload_getter):
            try:
                preferred_offload = offload_getter()
            except Exception:
                preferred_offload = None
            if preferred_offload in ("cpu", "meta"):
                offload_device = preferred_offload
        runner = LayerWiseRunner(adapter=adapter, offload_device=offload_device)
    else:
        raise ValueError(f"unsupported runner type for anti-outlier run: {spec_runner}")

    ctx = ContextFactory().create(is_distributed=False)
    with ContextManager(ctx):
        for process_cfg in quant_config.spec.process:
            runner.add_processor(processor_cfg=process_cfg)
        runner.run(model=model, calib_data=calib_data, device=device_type)


def _load_fixed_dataset(fixed_input: Any) -> list[Any]:
    if isinstance(fixed_input, (str, Path)):
        loaded = yaml.safe_load(Path(fixed_input).read_text(encoding="utf-8"))
    else:
        loaded = fixed_input
    return loaded if isinstance(loaded, list) else [loaded]


def _infer_model_type(model_path: Path) -> str:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ValueError(
            "model_type is required when model_path/config.json is unavailable"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_type = config.get("model_type")
    if not model_type:
        raise ValueError("model_type is required when config.json has no model_type")
    return str(model_type)


def _load_interface_validation(
    path: str | Path,
    expected_interface: str,
    expected_checkpoint_identity: str,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise InterfaceValidationError(
            f"interface validation artifact does not exist: {resolved}"
        )
    try:
        loaded = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InterfaceValidationError(
            f"interface validation artifact is not valid JSON: {exc}"
        ) from exc
    if not isinstance(loaded, Mapping):
        raise InterfaceValidationError(
            "interface validation artifact must contain a JSON object"
        )
    if loaded.get("interface") != expected_interface:
        raise InterfaceValidationError(
            "interface validation artifact does not match the selected algorithm"
        )
    if loaded.get("status") != "PASS" or loaded.get("passed") is False:
        raise InterfaceValidationError(
            "interface validation artifact is not a passing validation"
        )
    if loaded.get("checkpoint_identity") != expected_checkpoint_identity:
        raise InterfaceValidationError(
            "interface validation checkpoint identity does not match the current checkpoint"
        )
    evidence: dict[str, Any] = {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
    }
    if isinstance(loaded.get("status"), str):
        evidence["status"] = loaded["status"]
    if isinstance(loaded.get("interface"), str):
        evidence["interface"] = loaded["interface"]
    evidence["checkpoint_identity"] = expected_checkpoint_identity
    return evidence


def _check_installed_adapter_interface(adapter: Any, algorithm: str) -> None:
    """Reject stale evidence when the installed Adapter lacks the interface."""
    from msmodelslim.model import interface_hub  # noqa: PLC0415

    name = ALGORITHM_INTERFACE_NAMES[algorithm]
    interface = getattr(interface_hub, name)
    if not isinstance(adapter, interface):
        raise InterfaceValidationError(
            f"installed Adapter {_qualified_class_name(adapter)} does not implement {name}"
        )


def _select_runner(runner_type: str, adapter: Any, model: Any | None = None) -> str:
    if runner_type in ("model_wise", "layer_wise"):
        return runner_type
    if runner_type != "auto":
        raise ValueError("runner_type must be 'auto', 'model_wise', or 'layer_wise'")
    # msModelSlim AUTO uses the layer-wise runner on a single device. Preserve
    # that memory-safe behavior unless the caller explicitly requests
    # ``model_wise``; loading an ordinary large model on NPU just to detect its
    # state can otherwise OOM before the runner starts.
    return "layer_wise"


def _has_meta_parameters(model: Any) -> bool:
    try:
        parameters = model.parameters()
    except AttributeError:
        return False
    for parameter in parameters:
        if getattr(getattr(parameter, "device", None), "type", None) == "meta":
            return True
    return False


def _layer_count(model: Any, path: str) -> int | None:
    try:
        module = model.get_submodule(path)
    except (AttributeError, RuntimeError):
        return None
    try:
        return len(module)
    except TypeError:
        return None


def _expected_layer_count(adapter: Any) -> int | None:
    config = getattr(adapter, "config", None)
    candidates = (
        config,
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
    )
    for candidate in candidates:
        value = getattr(candidate, "num_hidden_layers", None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _has_partial_decoder_layers(model: Any, adapter: Any) -> bool:
    expected = _expected_layer_count(adapter)
    if expected is None:
        return False
    for path in (
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "layers",
    ):
        count = _layer_count(model, path)
        if count is not None and count < expected:
            return True
    return False


def apply_one_anti_outlier_and_record_logits(
    algorithm: str,
    model_path: str | Path,
    fixed_input: Any,
    output_dir: str | Path,
    *,
    model_type: str | None = None,
    trust_remote_code: bool = True,
    device: str = "npu",
    runner_type: str = "auto",
    logits_capture_patch: str | Path | None = None,
    interface_validation_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply one algorithm through the registered msModelSlim model adapter."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_model_path = Path(model_path).expanduser().resolve()
    record_path = destination / f"anti_outlier_run.{algorithm}.json"
    record: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "STARTED",
        "algorithm": algorithm,
        "model_path": str(resolved_model_path),
        "quantization_run": False,
        "logits_seed": seed,
        "patch": None,
        "interface_validation": None,
        "artifacts": {"run_record": str(record_path)},
    }
    _write_json(record_path, record)

    validation_path = destination / f"final_logits_patch_validation.{algorithm}.json"
    validation: dict[str, Any] = {
        "schema": PATCH_VALIDATION_SCHEMA,
        "status": "UNSUPPORTED",
        "patch_path": str(Path(logits_capture_patch).expanduser().resolve())
        if logits_capture_patch is not None
        else None,
    }

    try:
        if algorithm not in DEFAULT_CONFIG_NAMES:
            raise ValueError(f"unsupported anti-outlier algorithm: {algorithm}")
        if logits_capture_patch is None:
            raise PatchUnsupportedError(
                "--logits-capture-patch is required for the logits gate"
            )
        if interface_validation_path is None:
            raise InterfaceValidationError(
                "--interface-validation is required for the logits gate"
            )

        resolved_model_type = model_type or _infer_model_type(resolved_model_path)
        checkpoint_id = checkpoint_identity(resolved_model_path)
        interface_validation = _load_interface_validation(
            interface_validation_path,
            ALGORITHM_INTERFACE_NAMES[algorithm],
            checkpoint_id,
        )
        from msmodelslim.core.const import DeviceType
        from msmodelslim.model.plugin_factory.plugin_model_factory import (
            PluginModelFactory,
        )

        adapter = PluginModelFactory().create(
            resolved_model_type, resolved_model_path, trust_remote_code
        )
        _check_installed_adapter_interface(adapter, algorithm)
        adapter_class = _qualified_class_name(adapter)
        if interface_validation.get("adapter_class") not in (None, adapter_class):
            raise InterfaceValidationError(
                "interface validation Adapter class does not match the installed Adapter"
            )
        generated_patch = load_generated_patch(logits_capture_patch)
        record.update(
            {
                "model_type": resolved_model_type,
                "checkpoint_identity": checkpoint_id,
                "patch": {
                    "path": str(generated_patch.path),
                    "sha256": generated_patch.sha256,
                    "metadata": generated_patch.metadata,
                    "validation_artifact": str(validation_path),
                },
                "interface_validation": interface_validation,
            }
        )
        record["artifacts"]["patch_validation"] = str(validation_path)
        validation.update(
            {
                "patch_path": str(generated_patch.path),
                "patch_sha256": generated_patch.sha256,
                "metadata": generated_patch.metadata,
                "model_type": resolved_model_type,
                "checkpoint_identity": checkpoint_id,
            }
        )
        _write_json(record_path, record)

        record["adapter_class"] = adapter_class
        validation["adapter_class"] = adapter_class
        validate_patch_target(
            generated_patch.capture,
            adapter,
            resolved_model_type,
            checkpoint_id,
            metadata=generated_patch.metadata,
        )

        selected_runner = _select_runner(runner_type, adapter)
        load_device = (
            DeviceType.CPU if selected_runner == "layer_wise" else DeviceType(device)
        )
        calib_data = _load_fixed_dataset(fixed_input)
        prepared_inputs = adapter.handle_dataset(calib_data, device=load_device)
        if not prepared_inputs:
            raise ValueError(
                "msModelSlim adapter.handle_dataset returned no fixed input"
            )
        prepared_input = prepared_inputs[0]
        record["input_summary"] = _summarize_input(prepared_input)
        validation["input_summary"] = record["input_summary"]
        processor_entry, config_source = _load_official_processor_entry(algorithm)
        model = adapter.init_model(device=load_device)
        detected_runner = _select_runner(runner_type, adapter, model)
        if runner_type == "auto":
            selected_runner = detected_runner
        loading_state = {
            "has_meta_parameters": _has_meta_parameters(model),
            "has_partial_decoder_layers": _has_partial_decoder_layers(model, adapter),
        }
        practice = _build_practice(algorithm, processor_entry, selected_runner)
        record["runner_type"] = selected_runner
        record["runner_selection"] = {
            "requested": runner_type,
            "selected": selected_runner,
            "initial_selection": _select_runner(runner_type, adapter),
            "model_loading_state": loading_state,
        }
        record["config_source"] = config_source
        record["processor_config"] = dict(processor_entry)
        record["practice"] = practice

        bound_capture = MethodType(generated_patch.capture, adapter)
        with mock_patch.object(
            adapter, "capture_final_logits", bound_capture, create=True
        ):
            behavior = validate_patch_behavior(
                adapter.capture_final_logits,
                model,
                prepared_input,
                device,
                seed,
                reference_capture=(
                    _capture_native_forward if selected_runner == "model_wise" else None
                ),
                logits_scope=str(generated_patch.metadata["logits_scope"]),
            )
            validation.update(behavior)
            validation["patch_path"] = str(generated_patch.path)
            validation["patch_sha256"] = generated_patch.sha256
            validation["metadata"] = generated_patch.metadata
            validation["model_type"] = resolved_model_type
            validation["adapter_class"] = adapter_class
            validation["checkpoint_identity"] = checkpoint_id
            _write_json(validation_path, validation)
            record["artifacts"]["patch_validation"] = str(validation_path)
            if validation["status"] != "PASS":
                raise PatchValidationError(
                    str(validation.get("error", "patch behavior validation failed"))
                )
            record["patch"]["validation_status"] = "PASS"
            record["patch"]["validation_sha256"] = _sha256_file(validation_path)
            record["status"] = "PATCH_VALIDATED"
            _write_json(record_path, record)
            patch_provenance = dict(record["patch"])

            return _apply_one_anti_outlier_and_record_logits(
                algorithm,
                resolved_model_path,
                prepared_input,
                destination,
                load_model_and_adapter=lambda path: (model, adapter),
                capture_final_logits=lambda current_model,
                current_input: adapter.capture_final_logits(
                    current_model, current_input, device
                ),
                apply_processor=lambda current_model,
                current_adapter,
                config: _apply_with_msmodelslim(
                    current_model,
                    current_adapter,
                    config,
                    calib_data,
                    selected_runner,
                    device,
                    seed,
                ),
                processor_config=processor_entry,
                config_source=config_source,
                runner_type=selected_runner,
                practice=practice,
                seed=seed,
                checkpoint_identity_value=checkpoint_id,
                patch_provenance=patch_provenance,
                input_summary=record["input_summary"],
                adapter_class=adapter_class,
                run_record=record,
                interface_validation=interface_validation,
            )
    except InterfaceValidationError as exc:
        _record_failure(record, "INTERFACE_VALIDATION_FAILED", "interface_validation", exc)
        validation["status"] = "UNSUPPORTED"
        validation["error"] = f"{type(exc).__name__}: {exc}"
        raise
    except PatchUnsupportedError as exc:
        _record_failure(record, "PATCH_UNSUPPORTED", "patch", exc)
        validation["status"] = "UNSUPPORTED"
        validation["error"] = f"{type(exc).__name__}: {exc}"
        raise
    except PatchValidationError as exc:
        _record_failure(record, "PATCH_VALIDATION_FAILED", "patch_validation", exc)
        validation["status"] = "FAIL"
        validation["error"] = f"{type(exc).__name__}: {exc}"
        raise
    except Exception as exc:
        if record["status"] in ("STARTED", "PATCH_VALIDATED"):
            _record_failure(record, "PROCESSOR_FAILED", "setup", exc)
        raise
    finally:
        _write_json(validation_path, validation)
        _write_json(record_path, record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", required=True, choices=tuple(DEFAULT_CONFIG_NAMES)
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-type")
    parser.add_argument("--fixed-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--logits-capture-patch",
        type=Path,
        help="Agent-generated model-specific final-logits capture patch.",
    )
    parser.add_argument(
        "--interface-validation",
        type=Path,
        help="Matching interface-validation JSON evidence for this checkpoint.",
    )
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--runner",
        choices=("auto", "model_wise", "layer_wise"),
        default="auto",
        help="auto defaults to layer_wise for memory safety",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-trust-remote-code", action="store_true")
    args = parser.parse_args()

    result = apply_one_anti_outlier_and_record_logits(
        args.algorithm,
        args.model_path,
        args.fixed_input,
        args.output_dir,
        model_type=args.model_type,
        trust_remote_code=not args.no_trust_remote_code,
        device=args.device,
        runner_type=args.runner,
        logits_capture_patch=args.logits_capture_patch,
        interface_validation_path=args.interface_validation,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
