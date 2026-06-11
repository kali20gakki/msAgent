#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is part of the MindStudio project.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#    http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Parse and validate MSAGENT_IO v1 blocks from task delegation payloads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_NAME = "msagent.subagent_io"

_MSAGENT_IO_FENCE = re.compile(
    r"```msagent-io\s+v1\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(slots=True)
class ProtocolParseResult:
    """Outcome of parsing one MSAGENT_IO block from free text."""

    parsed: bool
    valid: bool
    payload: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    error_data: dict[str, Any] | None = None
    io_status: str | None = None


def extract_msagent_io_payload(text: str | None) -> dict[str, Any] | None:
    """Return the JSON object inside the first ``msagent-io v1`` fence, if any."""
    if not text or not text.strip():
        return None

    match = _MSAGENT_IO_FENCE.search(text)
    if not match:
        return None

    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None


def parse_delegation_input(
    text: str | None,
    *,
    expected_subagent_type: str,
) -> ProtocolParseResult:
    """Parse and validate the input block from a task ``description``."""
    payload = extract_msagent_io_payload(text)
    if payload is None:
        return ProtocolParseResult(parsed=False, valid=False, errors=["missing_msagent_io_block"])

    errors = _validate_envelope(payload, expected_subagent_type=expected_subagent_type)
    input_data = payload.get("input")
    if not isinstance(input_data, dict):
        errors.append("missing_input_object")
    elif not input_data:
        errors.append("empty_input_object")
    else:
        errors.extend(_validate_delegation_input(expected_subagent_type, input_data))

    return ProtocolParseResult(
        parsed=True,
        valid=not errors,
        payload=payload,
        errors=errors,
        input_data=input_data if isinstance(input_data, dict) else None,
    )


def parse_completion_output(
    text: str | None,
    *,
    expected_subagent_type: str,
) -> ProtocolParseResult:
    """Parse and validate the output block from a task ``ToolMessage``."""
    payload = extract_msagent_io_payload(text)
    if payload is None:
        return ProtocolParseResult(parsed=False, valid=False, errors=["missing_msagent_io_block"])

    errors = _validate_envelope(payload, expected_subagent_type=expected_subagent_type)
    io_status = _coerce_str(payload.get("status"))
    if io_status not in {"ok", "failed"}:
        errors.append("invalid_status")

    output_data = payload.get("output")
    error_data = payload.get("error")
    if io_status == "ok":
        if not isinstance(output_data, dict):
            errors.append("missing_output_object")
        else:
            errors.extend(_validate_completion_output(expected_subagent_type, output_data))
    elif io_status == "failed":
        if not isinstance(error_data, dict):
            errors.append("missing_error_object")

    return ProtocolParseResult(
        parsed=True,
        valid=not errors,
        payload=payload,
        errors=errors,
        output_data=output_data if isinstance(output_data, dict) else None,
        error_data=error_data if isinstance(error_data, dict) else None,
        io_status=io_status,
    )


def _validate_envelope(payload: dict[str, Any], *, expected_subagent_type: str) -> list[str]:
    errors: list[str] = []
    if payload.get("protocol") != PROTOCOL_NAME:
        errors.append("invalid_protocol")

    block_type = _coerce_str(payload.get("subagent_type"))
    if not block_type:
        errors.append("missing_subagent_type")
    elif block_type != expected_subagent_type:
        errors.append("subagent_type_mismatch")

    return errors


def _validate_delegation_input(subagent_type: str, input_data: dict[str, Any]) -> list[str]:
    if subagent_type == "quant-tuning-evaluation-generator":
        return _validate_evaluation_generator_input(input_data)
    if subagent_type == "msmodelslim-model-analysis":
        return _validate_model_analysis_input(input_data)
    if subagent_type == "msmodelslim-model-adapt":
        return _validate_model_adapt_input(input_data)
    return []


def _validate_completion_output(subagent_type: str, output_data: dict[str, Any]) -> list[str]:
    validators = {
        "quant-tuning-practice-generator": _validate_practice_generator_output,
        "quant-tuning-evaluation-generator": _validate_evaluation_generator_output,
        "quant-tuning-quantizer": _validate_quantizer_output,
        "quant-tuning-evaluator": _validate_evaluator_output,
        "msmodelslim-model-analysis": _validate_model_analysis_output,
        "msmodelslim-model-adapt": _validate_model_adapt_output,
    }
    validator = validators.get(subagent_type)
    if validator is None:
        return []
    return validator(output_data)


def _validate_practice_generator_output(output_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _coerce_str(output_data.get("practice_path")):
        errors.append("missing_practice_path")
    validation = output_data.get("validation")
    if not isinstance(validation, dict):
        errors.append("missing_validation")
    errors.extend(
        _validate_commands(
            output_data,
            required_names={"validate_practice_yaml", "sensitive_layer_analysis"},
            optional_names=set(),
        )
    )
    return errors


def _validate_evaluation_generator_output(output_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _coerce_str(output_data.get("evaluate_config_path")):
        errors.append("missing_evaluate_config_path")
    commands = output_data.get("commands")
    if commands is not None:
        errors.extend(_validate_commands(output_data, required_names=set(), optional_names=set()))
    return errors


def _validate_quantizer_output(output_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if output_data.get("success") is None:
        errors.append("missing_success")
    if not _coerce_str(output_data.get("quantized_path")):
        errors.append("missing_quantized_path")
    if output_data.get("exit_code") is None:
        errors.append("missing_exit_code")
    errors.extend(_validate_commands(output_data, required_names={"quantize"}, optional_names=set()))
    return errors


def _validate_evaluator_output(output_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if output_data.get("overall_passed") is None:
        errors.append("missing_overall_passed")
    datasets = output_data.get("datasets")
    if not isinstance(datasets, list):
        errors.append("missing_datasets")
    errors.extend(
        _validate_commands(
            output_data,
            required_names={"inference_service", "evaluation"},
            optional_names=set(),
        )
    )
    return errors


def _validate_commands(
    output_data: dict[str, Any],
    *,
    required_names: set[str],
    optional_names: set[str],
) -> list[str]:
    errors: list[str] = []
    commands = output_data.get("commands")
    if not isinstance(commands, list) or not commands:
        errors.append("missing_commands")
        return errors

    seen: set[str] = set()
    for index, item in enumerate(commands):
        if not isinstance(item, dict):
            errors.append(f"commands[{index}]_invalid")
            continue

        name = _coerce_str(item.get("name"))
        if not name:
            errors.append(f"commands[{index}]_missing_name")
            continue
        seen.add(name)

        skipped = bool(item.get("skipped"))
        if not skipped and not _coerce_str(item.get("command")):
            errors.append(f"commands[{index}]_missing_command")

    for required in sorted(required_names):
        if required not in seen:
            errors.append(f"missing_command:{required}")

    allowed = required_names | optional_names
    if allowed:
        for name in seen:
            if name not in allowed:
                errors.append(f"unexpected_command:{name}")

    return errors


def _validate_model_analysis_input(input_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _coerce_str(input_data.get("model_type")):
        errors.append("missing_model_type")
    if not _coerce_str(input_data.get("model_path")):
        errors.append("missing_model_path")
    return errors


def _validate_model_adapt_input(input_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _coerce_str(input_data.get("model_type")):
        errors.append("missing_model_type")
    if not _coerce_str(input_data.get("model_path")):
        errors.append("missing_model_path")
    if not _coerce_str(input_data.get("analysis_report_path")):
        errors.append("missing_analysis_report_path")
    return errors


def _validate_model_analysis_output(output_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    next_step = _coerce_str(output_data.get("next_step"))
    if not next_step:
        errors.append("missing_next_step")
    elif next_step not in {"model-adapt", "dequant", "blocked", "need_user_input"}:
        errors.append("invalid_next_step")
    implementation_source = _coerce_str(output_data.get("implementation_source"))
    if not implementation_source:
        errors.append("missing_implementation_source")
    elif implementation_source not in {"transformers", "model-local", "blocked"}:
        errors.append("invalid_implementation_source")
    if not _coerce_str(output_data.get("summary")):
        errors.append("missing_summary")
    if not _coerce_str(output_data.get("report_path")):
        errors.append("missing_report_path")
    commands = output_data.get("commands")
    if commands is not None:
        errors.extend(_validate_commands(output_data, required_names=set(), optional_names=set()))
    return errors


def _validate_model_adapt_output(output_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if output_data.get("adapter_registered") is None:
        errors.append("missing_adapter_registered")
    verification_steps = output_data.get("verification_steps")
    if not isinstance(verification_steps, list) or len(verification_steps) != 4:
        errors.append("invalid_verification_steps")
    elif not all(isinstance(step, dict) and step.get("passed") is not None for step in verification_steps):
        errors.append("invalid_verification_steps")
    errors.extend(
        _validate_commands(
            output_data,
            required_names={
                "install",
                "verification_step1",
                "verification_step2",
                "verification_step3",
                "verification_step4",
            },
            optional_names=set(),
        )
    )
    return errors


def _validate_evaluation_generator_input(input_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    deprecated = {"target_datasets", "accuracy_targets", "accuracy_tolerance"} & input_data.keys()
    if deprecated:
        errors.append("deprecated_fields:" + ",".join(sorted(deprecated)))

    datasets = input_data.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        errors.append("missing_datasets")
        return errors

    for index, item in enumerate(datasets):
        if not isinstance(item, dict):
            errors.append(f"datasets[{index}]_invalid")
            continue
        if not _coerce_str(item.get("name")):
            errors.append(f"datasets[{index}]_missing_name")
        if item.get("target") is None:
            errors.append(f"datasets[{index}]_missing_target")
        if not _coerce_str(item.get("config_name")):
            errors.append(f"datasets[{index}]_missing_config_name")

    return errors


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
