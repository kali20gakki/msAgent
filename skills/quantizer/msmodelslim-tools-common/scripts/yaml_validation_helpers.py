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
# -*- coding: utf-8 -*-
"""Practice YAML validation helpers (from msmodelslim.mcp.yaml_validation)."""

from __future__ import annotations

from typing import Any


def build_parse_error(exc: Exception) -> list[dict[str, Any]]:
    return [
        {
            "error_type": "parse_error",
            "field_path": "",
            "message": str(exc),
            "action": "Please fix YAML syntax and try again.",
        }
    ]


def build_schema_errors(exc: Exception) -> list[dict[str, Any]]:
    details = getattr(exc, "errors", None)
    if callable(details):
        parsed = []
        for item in details():
            loc = item.get("loc", [])
            parsed.append(
                {
                    "error_type": "schema_error",
                    "field_path": ".".join([str(x) for x in loc]),
                    "message": item.get("msg", str(exc)),
                    "action": "Please update this field to satisfy PracticeConfig schema.",
                }
            )
        if parsed:
            return parsed
    return [
        {
            "error_type": "schema_error",
            "field_path": "",
            "message": str(exc),
            "action": "Please fix schema violations and try again.",
        }
    ]


def validate_practice_yaml(practice_path: str) -> dict[str, Any]:
    from pydantic import ValidationError

    from msmodelslim.core.practice import PracticeConfig
    from msmodelslim.utils.security import yaml_safe_load

    try:
        content = yaml_safe_load(practice_path)
    except Exception as exc:
        return {"ok": True, "valid": False, "errors": build_parse_error(exc)}

    if not isinstance(content, dict):
        return {
            "ok": True,
            "valid": False,
            "errors": [
                {
                    "error_type": "parse_error",
                    "field_path": "",
                    "message": "YAML root must be a mapping object.",
                    "action": "Please ensure YAML top-level is key-value mapping.",
                }
            ],
        }

    try:
        PracticeConfig.model_validate(content)
        return {"ok": True, "valid": True, "errors": []}
    except ValidationError as exc:
        return {"ok": True, "valid": False, "errors": build_schema_errors(exc)}
    except Exception as exc:
        return {
            "ok": True,
            "valid": False,
            "errors": [
                {
                    "error_type": "business_rule_error",
                    "field_path": "",
                    "message": str(exc),
                    "action": "Please adjust YAML according to skill constraints.",
                }
            ],
        }
