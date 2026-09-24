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

from __future__ import annotations

from typing import Annotated

import pytest
from pydantic import BaseModel, Field, ValidationError

import msagent.utils.validators as validators_module
from msagent.utils.validators import json_list_parser, json_safe_tool


class _Item(BaseModel):
    name: str
    quantity: int


def test_json_list_parser_returns_original_value_when_input_is_not_string() -> None:
    parser = json_list_parser(_Item)
    original = [_Item(name="cpu", quantity=2)]

    assert parser(original) is original


def test_json_list_parser_builds_models_when_json_array_is_valid() -> None:
    parser = json_list_parser(_Item)

    result = parser('[{"name": "npu", "quantity": 8}]')

    assert result == [_Item(name="npu", quantity=8)]


def test_json_list_parser_repairs_payload_when_json_is_malformed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = json_list_parser(_Item)
    repaired_payload = [{"name": "memory", "quantity": 4}]
    repair_calls: list[str] = []

    def fake_repair(value: str) -> list[dict[str, object]]:
        repair_calls.append(value)
        return repaired_payload

    monkeypatch.setattr(validators_module, "repair_loads", fake_repair)

    result = parser("not valid json")

    assert result == [_Item(name="memory", quantity=4)]
    assert repair_calls == ["not valid json"]


def test_json_list_parser_raises_clear_error_when_json_and_repair_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = json_list_parser(_Item)
    monkeypatch.setattr(
        validators_module,
        "repair_loads",
        lambda _value: (_ for _ in ()).throw(RuntimeError("bad")),
    )

    with pytest.raises(ValueError, match=r"Failed to parse JSON \(auto-repair failed\)"):
        parser("not valid json")


def test_json_list_parser_rejects_object_when_json_is_not_array() -> None:
    parser = json_list_parser(_Item)

    with pytest.raises(TypeError, match=r"Expected JSON array for _Item, got dict"):
        parser('{"name": "npu", "quantity": 8}')


def test_json_list_parser_reports_model_error_when_array_item_is_invalid() -> None:
    parser = json_list_parser(_Item)

    with pytest.raises(ValidationError) as exc_info:
        parser('[{"name": "npu", "quantity": "many"}]')

    assert "quantity" in str(exc_info.value)


def test_json_safe_tool_parses_model_list_when_argument_is_json_string() -> None:
    @json_safe_tool
    def summarize_items(
        items: Annotated[list[_Item], Field(description="Items to summarize")],
        prefix: Annotated[str, Field(description="Output prefix")] = "total",
    ) -> str:
        """Summarize item quantities."""
        return f"{prefix}:{sum(item.quantity for item in items)}"

    result = summarize_items.invoke({"items": '[{"name": "npu", "quantity": 3}]'})

    assert result == "total:3"
    assert summarize_items.args_schema.model_fields["items"].description == "Items to summarize"
    assert summarize_items.args_schema.model_fields["prefix"].default == "total"


def test_json_safe_tool_preserves_plain_list_when_item_type_is_not_model() -> None:
    @json_safe_tool
    def join_names(names: list[str]) -> str:
        """Join names."""
        return ",".join(names)

    assert join_names.invoke({"names": ["cpu", "npu"]}) == "cpu,npu"


def test_json_safe_tool_omits_parameter_when_annotation_is_missing() -> None:
    @json_safe_tool
    def annotated_only(name: str, ignored="unused") -> str:
        """Return the annotated value."""
        return name

    assert set(annotated_only.args_schema.model_fields) == {"name"}
    assert annotated_only.invoke({"name": "npu"}) == "npu"
