#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Type helpers and device parsing for quant-tuning scripts."""

from __future__ import annotations

from typing import List, Optional, Tuple

from lab_paths import get_lab_calib_dir, get_lab_practice_dir

__all__ = [
    "get_lab_calib_dir",
    "get_lab_practice_dir",
    "parse_quant_device",
    "to_analysis_metrics",
    "to_device_type",
    "to_quant_type",
]


def to_device_type(device: str):
    from msmodelslim.core.const import DeviceType

    return DeviceType(device.lower())


def to_quant_type(quant_type: Optional[str]):
    from msmodelslim.core.const import QuantType

    if quant_type is None:
        return None
    return QuantType(quant_type.lower())


def to_analysis_metrics(metrics: str):
    from msmodelslim.app.analysis.application import AnalysisMetrics

    return AnalysisMetrics(metrics.lower())


def parse_quant_device(device: str) -> Tuple[object, Optional[List[int]]]:
    from msmodelslim.core.const import DeviceType
    from msmodelslim.utils.exception import SchemaValidateError

    device = device.strip()
    if not device:
        raise SchemaValidateError("device string cannot be empty")

    parts = device.split(":", 1)
    try:
        device_type = DeviceType(parts[0].strip())
    except ValueError as exc:
        valid_types = ", ".join([f"'{dt.value}'" for dt in DeviceType])
        raise SchemaValidateError(
            f"Invalid device type: '{parts[0].strip()}'. Supported device types: {valid_types}"
        ) from exc

    device_indices = None
    if len(parts) > 1 and parts[1].strip():
        try:
            device_indices = [int(idx.strip()) for idx in parts[1].split(",") if idx.strip()]
        except ValueError as exc:
            raise SchemaValidateError(
                f"Invalid device indices format: '{parts[1].strip()}'. "
                f"Expected comma-separated integers (e.g., '0,1,2,3')"
            ) from exc
    return device_type, device_indices
