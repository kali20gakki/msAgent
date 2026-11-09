#!/usr/bin/env python3
"""Generate Markdown and self-contained HTML alignment reports."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import textwrap
import unicodedata
from decimal import Decimal
from pathlib import Path
from typing import Any


# Markdown 表格单元格内的竖线需转义;必须在模块级定义常量再在 f-string 内引用——
# Python <3.12(PEP 498)禁止在 f-string 表达式部分内嵌反斜杠字面量,直接写 '\\|' 会 SyntaxError。
# 表达式内的字符串字面量一律用单引号('|'),避免与外层 f-string 的双引号定界符嵌套——
# 同引号嵌套是 PEP 701(Python 3.12)才允许,Python ≤3.11 会直接 SyntaxError。
PIPE_ESCAPE = "\\|"


STATUSES = {
    "resolved",
    "unresolved_replayed",
    "excluded_known_gap",
    "rejected_hypothesis",
    "pending",
}
PREFILL_STATUSES = {"exact", "nonzero", "not_run", "evidence_missing"}
E2E_STATUSES = {"zero", "nonzero", "not_run", "evidence_missing"}
TRIAGE_STATUSES = {"complete", "partial", "skipped", "failed"}
VALIDATION_STATUSES = {"pass", "fail", "not_run", "evidence_missing", "not_applicable"}
AXIS_STATUSES = {
    "tensor_prefill": PREFILL_STATUSES,
    "weight_sync": {"exact", "mismatch", "not_run", "evidence_missing", "not_applicable"},
    "stateful_training": VALIDATION_STATUSES - {"not_applicable"},
    "long_horizon_e2e": VALIDATION_STATUSES - {"not_applicable"},
    "reduced_model_transfer": VALIDATION_STATUSES,
    "requested_model_final": VALIDATION_STATUSES - {"not_applicable"},
    "production_patch_hygiene": VALIDATION_STATUSES - {"not_applicable"},
    "clean_runtime_activation": VALIDATION_STATUSES - {"not_applicable"},
    "performance_regression": VALIDATION_STATUSES,
}
SOURCE_TEMPLATE = (
    Path(__file__).resolve().parents[1]
    / "references"
    / "templates"
    / "alignment_report_source.zh-CN.json"
)

STATUS_LABELS = {
    "resolved": "已解决",
    "unresolved_replayed": "未解决（Replay 后继续）",
    "excluded_known_gap": "已知未解决差异",
    "rejected_hypothesis": "已排除假设",
    "pending": "待验证",
}
PREFILL_STATUS_LABELS = {
    "exact": "精确对齐",
    "nonzero": "存在非零差异",
    "not_run": "未执行",
    "evidence_missing": "证据不足",
}
E2E_STATUS_LABELS = {
    "zero": "已归零",
    "nonzero": "仍为非零",
    "not_run": "未执行",
    "evidence_missing": "证据不足",
}
METRIC_SCOPE_LABELS = {
    "standard_training": "标准训练",
    "minimal_one_layer": "单层最小路径",
    "full_rollout": "全量 Rollout",
}
OBSERVATION_LABELS = {
    "native": "原生路径",
    "conditional": "条件路径",
}
ROOT_CAUSE_LABELS = {
    "Op path": "算子/计算路径",
    "Input precision": "输入精度",
    "Config": "配置差异",
    "Missing step": "缺失或额外步骤",
    "Framework bug": "框架问题",
    "Test setup mismatch": "测试设置不一致",
    "Weight sync": "权重同步",
    "State transition": "状态转换",
    "Control plane": "控制面",
}
CONFIDENCE_LABELS = {
    "CONFIRMED": "已确认",
    "SUPPORTED": "证据支持",
    "CANDIDATE": "候选",
}
TRIAGE_STATUS_LABELS = {
    "complete": "完整",
    "partial": "部分完成",
    "skipped": "已跳过",
    "failed": "失败",
}
TRIAGE_CLASS_LABELS = {
    "missing_or_extra_op_between_exact_boundary": "精确边界之间存在缺失或额外操作",
    "in_module_impl_difference": "模块内部实现差异",
    "parameter_or_checkpoint_issue": "参数或检查点问题",
    "needs_parameter_evidence": "需要补充参数证据",
    "upstream_propagation": "上游差异传播",
    "needs_exact_input_evidence": "需要补充精确输入证据",
}
TRIAGE_OUTPUT_LABELS = {
    "exact_equal": "精确相等",
    "exact_difference": "精确比较存在差异",
    "statistics_same": "统计摘要相同（非精确证据）",
    "statistics_difference": "统计摘要存在差异",
    "shape_mismatch": "形状不一致",
    "dtype_mismatch": "数据类型不一致",
    "nonfinite": "包含 NaN/Inf",
    "metadata_only": "仅有元数据",
    "unknown": "未知",
    "missing_record": "缺少记录",
}
EVIDENCE_STRENGTH_LABELS = {
    "exact_output_observation": "精确输出观测",
    "comparability_blocker": "可比性阻断项",
    "summary_only": "仅统计摘要",
}
TIMING_LABELS = {
    "session_started_at": "会话开始时间",
    "session_ended_at": "会话结束时间",
    "total_wall_seconds": "总墙钟时间",
    "discovery_seconds": "差异发现",
    "module_1_seconds": "模块 1",
    "module_2_seconds": "模块 2",
    "module_3_seconds": "模块 3",
    "module_4_seconds": "模块 4",
    "module_5_seconds": "模块 5",
    "module_6_seconds": "模块 6",
    "repair_wave_wall_seconds": "修复波次墙钟时间",
    "summed_repair_agent_seconds": "修复 Agent 累计时间",
    "repair_queue_wait_seconds": "修复队列等待",
    "integration_seconds": "串行集成",
    "accelerator_run_seconds": "加速器运行",
    "accelerator_run_count": "加速器运行次数",
    "minimum_observed_window_seconds": "最小可观测时间窗口",
    "timing_basis": "计时依据",
    "estimated_parallel_saving_seconds": "估算并行节省时间",
}
TIMING_DISPLAY_KEYS = (
    "session_started_at",
    "session_ended_at",
    "total_wall_seconds",
    "minimum_observed_window_seconds",
    "module_1_seconds",
    "module_2_seconds",
    "module_3_seconds",
    "module_4_seconds",
    "module_5_seconds",
    "module_6_seconds",
    "discovery_seconds",
    "repair_wave_wall_seconds",
    "summed_repair_agent_seconds",
    "repair_queue_wait_seconds",
    "integration_seconds",
    "accelerator_run_seconds",
    "accelerator_run_count",
    "timing_basis",
    "estimated_parallel_saving_seconds",
)
ARTIFACT_LABELS = {
    "case_readme": "案例说明",
    "report_source": "报告事实源",
    "raw_tensor_included": "是否包含原始 Tensor",
    "raw_dump_included": "是否包含完整 Dump",
    "prompt_included": "是否包含原始 Prompt",
    "raw_training_log_included": "是否包含原始训练日志",
    "production_patch": "生产补丁",
    "apply_check": "补丁应用检查",
    "production_patch_hygiene_audit": "生产补丁卫生审计",
    "clean_runtime_audit": "干净运行审计",
    "validation_matrix_evidence": "分层验证矩阵证据",
}
PROVENANCE_LABELS = {
    "vllm_ascend_revision": "vLLM-Ascend 版本",
    "mindspeed_revision": "MindSpeed 版本",
    "megatron_revision_from_probe_metadata": "Megatron 版本（来自探针元数据）",
    "model_digest": "模型摘要",
    "sample_digest": "样本摘要",
    "config_digests": "配置摘要",
}
AXIS_LABELS = {
    "tensor_prefill": "静态 Tensor Prefill",
    "weight_sync": "权重同步",
    "stateful_training": "状态型训练",
    "long_horizon_e2e": "长程 E2E",
    "reduced_model_transfer": "缩减模型迁移",
    "requested_model_final": "目标模型最终验证",
    "production_patch_hygiene": "生产补丁卫生",
    "clean_runtime_activation": "干净运行激活",
    "performance_regression": "性能回归",
}
AXIS_STATUS_LABELS = {
    **PREFILL_STATUS_LABELS,
    "pass": "通过",
    "fail": "失败",
    "exact": "精确一致",
    "mismatch": "不一致",
    "not_applicable": "不适用",
}
VALIDATION_GATE_LABELS = {
    "diagnostic_tensor_replay": "诊断 Tensor Replay",
    "static_tensor_prefill": "静态 Tensor Prefill",
    "weight_sync": "权重同步",
    "stateful_training": "状态型训练",
    "long_horizon_e2e": "长程 E2E",
    "reduced_model_transfer": "缩减模型迁移",
    "requested_model_final": "目标模型最终验证",
    "production_patch_hygiene": "生产补丁卫生",
    "clean_runtime_activation": "干净运行激活",
    "performance_regression": "性能回归",
}
GATE_AXIS_EXPECTED = {
    "static_tensor_prefill": ("tensor_prefill", "exact"),
    "weight_sync": ("weight_sync", "exact"),
    "stateful_training": ("stateful_training", "pass"),
    "long_horizon_e2e": ("long_horizon_e2e", "pass"),
    "reduced_model_transfer": ("reduced_model_transfer", "pass"),
    "requested_model_final": ("requested_model_final", "pass"),
    "production_patch_hygiene": ("production_patch_hygiene", "pass"),
    "clean_runtime_activation": ("clean_runtime_activation", "pass"),
    "performance_regression": ("performance_regression", "pass"),
}
CANONICAL_REQUIRED_GATES = frozenset(GATE_AXIS_EXPECTED)


class ReportError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read report source {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError("report source root must be an object")
    return value


def string(value: Any, default: str = "未记录") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else default


def seconds(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "未记录"
    total = max(0, int(round(float(value))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def scalar(value: Any, default: str = "未记录") -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def label(raw: Any, labels: dict[str, str], default: str = "未记录") -> str:
    text = string(raw, default)
    return labels.get(text, text)


def status_label(raw: Any) -> str:
    return label(raw, STATUS_LABELS)


def prefill_status_label(raw: Any) -> str:
    return label(raw, PREFILL_STATUS_LABELS)


def e2e_status_label(raw: Any) -> str:
    return label(raw, E2E_STATUS_LABELS)


def observation_label(raw: Any) -> str:
    return label(raw, OBSERVATION_LABELS)


def root_cause_label(raw: Any) -> str:
    return label(raw, ROOT_CAUSE_LABELS)


def confidence_label(raw: Any) -> str:
    return label(raw, CONFIDENCE_LABELS)


def object_value(source: dict[str, Any], key: str) -> dict[str, Any]:
    value = source.get(key, {})
    return value if isinstance(value, dict) else {}


def object_list(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ReportError(f"{label} must be a list of objects")
    return value


def string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ReportError(f"{label} must be a list of non-empty strings")
    return [item.strip() for item in value]


def validate_triage(source: dict[str, Any]) -> None:
    if "triage" not in source:
        return
    triage = source.get("triage")
    if not isinstance(triage, dict):
        raise ReportError("triage must be an object")
    if triage.get("truth_role") != "triage_only":
        raise ReportError("triage.truth_role must be triage_only")
    if triage.get("status") not in TRIAGE_STATUSES:
        raise ReportError("triage.status is invalid")
    if triage.get("may_create_difference_packet") is not False:
        raise ReportError("triage.may_create_difference_packet must be false")
    if triage.get("candidate_ranking_must_not_define_first_divergence") is not True:
        raise ReportError(
            "triage.candidate_ranking_must_not_define_first_divergence must be true"
        )
    if triage.get("exact_comparison_must_follow_semantic_execution_order") is not True:
        raise ReportError(
            "triage.exact_comparison_must_follow_semantic_execution_order must be true"
        )

    candidates = object_list(triage.get("candidates", []), "triage.candidates")
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        label = f"triage.candidates[{index}]"
        candidate_id = string(candidate.get("candidate_id"), "")
        if not candidate_id or candidate_id in candidate_ids:
            raise ReportError(f"{label}.candidate_id is missing or duplicated")
        candidate_ids.add(candidate_id)
        if not string(candidate.get("mapping_id"), ""):
            raise ReportError(f"{label}.mapping_id is required")
        priority = candidate.get("priority_rank")
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 1:
            raise ReportError(f"{label}.priority_rank must be a positive integer")
        if candidate.get("formal_verification_required") is not True:
            raise ReportError(f"{label}.formal_verification_required must be true")

    object_list(
        triage.get("structural_false_positive_candidates", []),
        "triage.structural_false_positive_candidates",
    )
    object_list(
        triage.get("excluded_candidates", []), "triage.excluded_candidates"
    )
    string_list(triage.get("next_checks", []), "triage.next_checks")


def normalized_timing(value: Any) -> dict[str, Any]:
    timing = dict(value) if isinstance(value, dict) else {}
    if "accelerator_run_seconds" not in timing and "npu_run_seconds" in timing:
        timing["accelerator_run_seconds"] = timing["npu_run_seconds"]
    if "accelerator_run_count" not in timing and "npu_run_count" in timing:
        timing["accelerator_run_count"] = timing["npu_run_count"]
    timing.pop("npu_run_seconds", None)
    timing.pop("npu_run_count", None)
    return timing


def ordered_timing_items(timing: dict[str, Any]) -> list[tuple[str, Any]]:
    known = [(key, timing[key]) for key in TIMING_DISPLAY_KEYS if key in timing]
    extras = sorted(
        (key, value) for key, value in timing.items() if key not in TIMING_DISPLAY_KEYS
    )
    return known + extras


def topology_text(scope: dict[str, Any]) -> str:
    topology = object_value(scope, "topology")
    return "TP={} / PP={} / EP={}".format(
        topology.get("tp", "?"),
        topology.get("pp", "?"),
        topology.get("ep", "?"),
    )


def backend_lines(scope: dict[str, Any]) -> list[str]:
    return [
        f"训练后端：{string(scope.get('training_backend'))}",
        f"推理后端：{string(scope.get('inference_backend'))}",
        f"硬件后端：{string(scope.get('hardware_backend'))}",
        f"证据提供方式：{string(scope.get('evidence_provider'))}",
    ]


def validate_exact_gate(value: Any, label: str, extra_true: tuple[str, ...] = ()) -> None:
    if not isinstance(value, dict):
        raise ReportError(f"{label} must be an object")
    for key in (
        "same_shape",
        "same_dtype",
        "no_nonfinite",
        "torch_equal",
        "downstream_checked",
        *extra_true,
    ):
        if value.get(key) is not True:
            raise ReportError(f"{label}.{key} must be true")
    max_abs = value.get("max_abs_diff")
    if isinstance(max_abs, bool) or max_abs != 0:
        raise ReportError(f"{label}.max_abs_diff must be numeric zero")


def validate_metric_progression(
    source: dict[str, Any],
    difference_ids: set[str],
) -> None:
    progression = source.get("metric_progression")
    if progression is None:
        return
    if not isinstance(progression, dict):
        raise ReportError("metric_progression must be an object or null")
    if progression.get("truth_role") != "e2e_impact_only":
        raise ReportError("metric_progression.truth_role must be e2e_impact_only")
    metric_name = string(progression.get("metric_name"), "")
    if not metric_name:
        raise ReportError("metric_progression.metric_name is required")
    if not string(progression.get("evidence"), ""):
        raise ReportError("metric_progression.evidence is required")
    entries = progression.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ReportError("metric_progression.entries must be a non-empty list")
    stage_ids: set[str] = set()
    values_by_stage: dict[str, float] = {}
    for index, entry in enumerate(entries):
        label_text = f"metric_progression.entries[{index}]"
        if not isinstance(entry, dict):
            raise ReportError(f"{label_text} must be an object")
        stage_id = string(entry.get("stage_id"), "")
        if not stage_id or stage_id in stage_ids:
            raise ReportError(f"{label_text}.stage_id is missing or duplicated")
        stage_ids.add(stage_id)
        if not string(entry.get("label"), ""):
            raise ReportError(f"{label_text}.label is required")
        if entry.get("run_scope") not in METRIC_SCOPE_LABELS:
            raise ReportError(f"{label_text}.run_scope is invalid")
        value = entry.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ReportError(f"{label_text}.value must be finite numeric")
        values_by_stage[stage_id] = value
        step = entry.get("step")
        if step is not None and (
            isinstance(step, bool) or not isinstance(step, int) or step < 0
        ):
            raise ReportError(f"{label_text}.step must be a non-negative integer or null")
        related = entry.get("related_difference_ids", [])
        if not isinstance(related, list) or not all(
            isinstance(item, str) for item in related
        ):
            raise ReportError(
                f"{label_text}.related_difference_ids must be a list of strings"
            )
        unknown = sorted(set(related) - difference_ids)
        if unknown:
            raise ReportError(
                f"{label_text}.related_difference_ids contains unknown IDs: "
                + ", ".join(unknown)
            )
    headline_stage_id = string(progression.get("headline_stage_id"), "")
    if headline_stage_id not in values_by_stage:
        raise ReportError(
            "metric_progression.headline_stage_id must reference an entry"
        )
    results = source["results"]
    if results.get("e2e_metric_name") != metric_name:
        raise ReportError(
            "results.e2e_metric_name must match metric_progression.metric_name"
        )
    if results.get("e2e_metric_value") != values_by_stage[headline_stage_id]:
        raise ReportError(
            "results.e2e_metric_value must match the headline progression stage"
        )


def finite_number(value: Any, label_text: str, *, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ReportError(f"{label_text} must be finite numeric")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ReportError(f"{label_text} must be >= {minimum}")
    return number


def validate_acceptance_contract(source: dict[str, Any]) -> None:
    contract = source.get("acceptance_contract")
    if contract is None:
        return
    if not isinstance(contract, dict):
        raise ReportError("acceptance_contract must be an object or null")
    for key in ("objective", "requested_model_scope", "stop_condition"):
        if not string(contract.get(key), ""):
            raise ReportError(f"acceptance_contract.{key} is required")
    gates = string_list(contract.get("required_gates"), "acceptance_contract.required_gates")
    if len(gates) != len(set(gates)):
        raise ReportError("acceptance_contract.required_gates must be unique")
    unknown_gates = sorted(set(gates) - CANONICAL_REQUIRED_GATES)
    if unknown_gates:
        raise ReportError(
            "acceptance_contract.required_gates contains unknown gates: "
            + ", ".join(unknown_gates)
        )
    if not isinstance(contract.get("long_horizon_required"), bool):
        raise ReportError("acceptance_contract.long_horizon_required must be boolean")
    if contract.get("status") not in {"pass", "fail", "not_run", "evidence_missing"}:
        raise ReportError("acceptance_contract.status is invalid")
    historical = contract.get("historical_failure_step")
    if historical is not None and (
        isinstance(historical, bool) or not isinstance(historical, int) or historical < 0
    ):
        raise ReportError(
            "acceptance_contract.historical_failure_step must be non-negative or null"
        )
    if contract.get("long_horizon_required") is True and "long_horizon_e2e" not in gates:
        raise ReportError(
            "long_horizon_required=true requires long_horizon_e2e in required_gates"
        )
    if contract.get("long_horizon_required") is False and "long_horizon_e2e" in gates:
        raise ReportError(
            "long_horizon_e2e required gate requires long_horizon_required=true"
        )


def validate_verification_axes(source: dict[str, Any]) -> None:
    axes = source.get("verification_axes")
    if axes is None:
        return
    if not isinstance(axes, dict):
        raise ReportError("verification_axes must be an object or null")
    if axes.get("truth_role") != "multi_axis_summary":
        raise ReportError("verification_axes.truth_role must be multi_axis_summary")
    for axis, statuses in AXIS_STATUSES.items():
        value = axes.get(axis)
        if not isinstance(value, dict):
            raise ReportError(f"verification_axes.{axis} must be an object")
        if value.get("status") not in statuses:
            raise ReportError(f"verification_axes.{axis}.status is invalid")
        if not string(value.get("evidence"), ""):
            raise ReportError(f"verification_axes.{axis}.evidence is required")
    native_status = source["results"].get("native_prefill_status")
    tensor_status = axes["tensor_prefill"].get("status")
    if tensor_status in {"exact", "nonzero"} and tensor_status != native_status:
        raise ReportError(
            "verification_axes.tensor_prefill must match results.native_prefill_status"
        )
    if axes["long_horizon_e2e"].get("status") == "pass" and source["results"].get("e2e_metric_status") != "zero":
        raise ReportError(
            "long_horizon_e2e pass requires results.e2e_metric_status=zero"
        )


def validate_validation_matrix(source: dict[str, Any]) -> None:
    matrix = source.get("validation_matrix")
    if matrix is None:
        return
    if not isinstance(matrix, dict):
        raise ReportError("validation_matrix must be an object or null")
    if matrix.get("truth_role") != "e2e_and_transfer_evidence":
        raise ReportError(
            "validation_matrix.truth_role must be e2e_and_transfer_evidence"
        )
    entries = object_list(matrix.get("entries", []), "validation_matrix.entries")
    run_ids: set[str] = set()
    for index, entry in enumerate(entries):
        label_text = f"validation_matrix.entries[{index}]"
        run_id = string(entry.get("run_id"), "")
        if not run_id or run_id in run_ids:
            raise ReportError(f"{label_text}.run_id is missing or duplicated")
        run_ids.add(run_id)
        for key in ("label", "gate", "model_scope", "metric_name", "evidence"):
            if not string(entry.get(key), ""):
                raise ReportError(f"{label_text}.{key} is required")
        if entry.get("gate") not in VALIDATION_GATE_LABELS:
            raise ReportError(f"{label_text}.gate is invalid")
        if entry.get("status") not in VALIDATION_STATUSES:
            raise ReportError(f"{label_text}.status is invalid")
        for key in ("expected_steps", "observed_steps", "response_length"):
            value = entry.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ReportError(f"{label_text}.{key} must be non-negative or null")
        for key in ("nonzero_count", "nonfinite_count"):
            value = entry.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ReportError(f"{label_text}.{key} must be non-negative or null")
        max_abs = entry.get("max_abs_diff")
        if max_abs is not None:
            finite_number(max_abs, f"{label_text}.max_abs_diff", minimum=0)
        crossed = entry.get("crossed_historical_failure_step")
        if crossed is not None and not isinstance(crossed, bool):
            raise ReportError(
                f"{label_text}.crossed_historical_failure_step must be boolean or null"
            )
        if entry.get("status") == "pass":
            if entry.get("complete") is not True:
                raise ReportError(f"{label_text}.complete must be true for pass")
            expected = entry.get("expected_steps")
            observed = entry.get("observed_steps")
            if expected is None or expected < 1:
                raise ReportError(
                    f"{label_text}.expected_steps must be positive for pass"
                )
            if observed is None or observed < expected:
                raise ReportError(f"{label_text} did not observe all expected steps")
            if entry.get("nonzero_count") != 0 or entry.get("nonfinite_count") != 0:
                raise ReportError(f"{label_text} pass requires zero failure counts")
            if max_abs != 0:
                raise ReportError(f"{label_text} pass requires max_abs_diff=0")
            if crossed is False:
                raise ReportError(
                    f"{label_text} pass cannot stop before the historical failure step"
                )


def validate_acceptance_closure(source: dict[str, Any]) -> None:
    contract = source.get("acceptance_contract")
    if not isinstance(contract, dict) or contract.get("status") != "pass":
        return
    matrix = source.get("validation_matrix")
    if not isinstance(matrix, dict):
        raise ReportError(
            "acceptance_contract.status=pass requires validation_matrix evidence"
        )
    passed_gates = {
        entry.get("gate")
        for entry in matrix.get("entries", [])
        if isinstance(entry, dict) and entry.get("status") == "pass"
    }
    missing = sorted(set(contract.get("required_gates", [])) - passed_gates)
    if missing:
        raise ReportError(
            "acceptance_contract.status=pass is missing passing gates: "
            + ", ".join(missing)
        )
    axes = source.get("verification_axes")
    if not isinstance(axes, dict):
        raise ReportError(
            "acceptance_contract.status=pass requires verification_axes evidence"
        )
    for gate in contract.get("required_gates", []):
        axis, expected_status = GATE_AXIS_EXPECTED[gate]
        if not isinstance(axes.get(axis), dict) or axes[axis].get("status") != expected_status:
            raise ReportError(
                f"acceptance gate {gate} requires verification_axes.{axis}={expected_status}"
            )
    final_rows = [
        entry
        for entry in matrix.get("entries", [])
        if isinstance(entry, dict)
        and entry.get("gate") == "requested_model_final"
        and entry.get("status") == "pass"
    ]
    if "requested_model_final" in contract.get("required_gates", []) and not any(
        entry.get("model_scope") == contract.get("requested_model_scope")
        for entry in final_rows
    ):
        raise ReportError(
            "requested_model_final pass must use acceptance_contract.requested_model_scope"
        )
    historical = contract.get("historical_failure_step")
    horizon_rows = [
        entry
        for entry in matrix.get("entries", [])
        if isinstance(entry, dict)
        and entry.get("gate") == "long_horizon_e2e"
        and entry.get("status") == "pass"
    ]
    if contract.get("long_horizon_required") is True and not any(
        entry.get("crossed_historical_failure_step") is True
        if historical is not None
        else entry.get("complete") is True
        for entry in horizon_rows
    ):
        raise ReportError("long-horizon acceptance evidence is incomplete")
    if {"long_horizon_e2e", "requested_model_final"} & set(contract.get("required_gates", [])):
        if source["results"].get("e2e_metric_status") != "zero":
            raise ReportError("final E2E acceptance requires e2e_metric_status=zero")
    if "static_tensor_prefill" in contract.get("required_gates", []):
        if source["results"].get("native_prefill_status") != "exact":
            raise ReportError("static Tensor acceptance requires native_prefill_status=exact")
    if "production_patch_hygiene" in contract.get("required_gates", []):
        if not isinstance(source.get("production_alignment_profile"), dict):
            raise ReportError("production acceptance requires production_alignment_profile")
    if "clean_runtime_activation" in contract.get("required_gates", []):
        if not isinstance(source.get("clean_run_gate"), dict):
            raise ReportError("clean runtime acceptance requires clean_run_gate")


def validate_production_runtime(source: dict[str, Any]) -> None:
    profile = source.get("production_alignment_profile")
    if profile is not None:
        if not isinstance(profile, dict):
            raise ReportError("production_alignment_profile must be an object or null")
        if profile.get("truth_role") != "production_alignment_profile":
            raise ReportError(
                "production_alignment_profile.truth_role must be production_alignment_profile"
            )
        if not isinstance(profile.get("active"), bool):
            raise ReportError("production_alignment_profile.active must be boolean")
        string_list(profile.get("features", []), "production_alignment_profile.features")
        for key in ("diagnostic_replay_enabled", "diagnostic_capture_enabled"):
            if profile.get(key) is not False:
                raise ReportError(f"production_alignment_profile.{key} must be false")
        if not string(profile.get("activation_evidence"), ""):
            raise ReportError("production_alignment_profile.activation_evidence is required")
    clean = source.get("clean_run_gate")
    if clean is not None:
        if not isinstance(clean, dict):
            raise ReportError("clean_run_gate must be an object or null")
        if clean.get("truth_role") != "production_activation_only":
            raise ReportError("clean_run_gate.truth_role must be production_activation_only")
        for key in (
            "fresh_process",
            "fresh_weights_loaded",
            "diagnostic_replay_disabled",
            "diagnostic_capture_disabled",
            "activation_proved",
        ):
            if clean.get(key) is not True:
                raise ReportError(f"clean_run_gate.{key} must be true")
        if not string(clean.get("evidence"), ""):
            raise ReportError("clean_run_gate.evidence is required")
        if not isinstance(profile, dict) or profile.get("active") is not True:
            raise ReportError(
                "clean_run_gate requires an active production_alignment_profile"
            )


def validate(source: dict[str, Any]) -> None:
    if source.get("schema_version") != 1:
        raise ReportError("schema_version must be 1")
    if source.get("language") != "zh-CN":
        raise ReportError("language must be zh-CN")
    scope = source.get("scope")
    if not isinstance(scope, dict) or scope.get("phase") != "prefill":
        raise ReportError("scope.phase must be prefill")
    topology = scope.get("topology")
    if not isinstance(topology, dict) or topology.get("tp") != 1:
        raise ReportError("scope.topology.tp must be 1")
    results = source.get("results")
    if not isinstance(results, dict):
        raise ReportError("results must be an object")
    if results.get("native_prefill_status") not in PREFILL_STATUSES:
        raise ReportError("results.native_prefill_status is invalid")
    if results.get("conditional_prefill_status") not in PREFILL_STATUSES:
        raise ReportError("results.conditional_prefill_status is invalid")
    if results.get("e2e_metric_status") not in E2E_STATUSES:
        raise ReportError("results.e2e_metric_status is invalid")
    replay_ids = results.get("unresolved_replay_ids")
    if not isinstance(replay_ids, list) or not all(isinstance(item, str) for item in replay_ids):
        raise ReportError("results.unresolved_replay_ids must be a list of strings")
    if results["native_prefill_status"] == "exact" and replay_ids:
        raise ReportError("native prefill cannot be exact while unresolved replay IDs remain")
    if len(replay_ids) != len(set(replay_ids)):
        raise ReportError("results.unresolved_replay_ids must be unique")
    if results["native_prefill_status"] == "exact":
        validate_exact_gate(
            results.get("native_exact_gate"),
            "results.native_exact_gate",
            ("real_path_activation_proved",),
        )
    if results["conditional_prefill_status"] == "exact":
        validate_exact_gate(
            results.get("conditional_exact_gate"),
            "results.conditional_exact_gate",
            ("replay_activation_proved",),
        )
        if results["native_prefill_status"] != "exact" and not replay_ids:
            raise ReportError(
                "conditional exact with non-exact native prefill requires unresolved replay IDs"
            )
    e2e_status = results["e2e_metric_status"]
    e2e_value = results.get("e2e_metric_value")
    if e2e_status in {"not_run", "evidence_missing"}:
        if e2e_value is not None:
            raise ReportError(f"E2E status {e2e_status} requires e2e_metric_value=null")
    elif (
        isinstance(e2e_value, bool)
        or not isinstance(e2e_value, (int, float))
        or not math.isfinite(e2e_value)
    ):
        raise ReportError("zero/nonzero E2E status requires a numeric metric value")
    elif e2e_status == "zero" and e2e_value != 0:
        raise ReportError("zero E2E status requires e2e_metric_value=0")
    elif e2e_status == "nonzero" and e2e_value == 0:
        raise ReportError("nonzero E2E status requires a nonzero metric value")
    differences = source.get("differences")
    if not isinstance(differences, list):
        raise ReportError("differences must be a list")
    ids: set[str] = set()
    status_by_id: dict[str, str] = {}
    for index, item in enumerate(differences):
        if not isinstance(item, dict):
            raise ReportError(f"differences[{index}] must be an object")
        diff_id = string(item.get("diff_id"), "")
        if not diff_id or diff_id in ids:
            raise ReportError(f"differences[{index}].diff_id is missing or duplicated")
        ids.add(diff_id)
        if item.get("status") not in STATUSES:
            raise ReportError(f"differences[{index}].status is invalid")
        status_by_id[diff_id] = item["status"]
        if item.get("observation") not in {"native", "conditional"}:
            raise ReportError(f"differences[{index}].observation is invalid")
        for key in ("discovery_max_abs_diff", "verification_max_abs_diff"):
            if item.get(key) is not None:
                finite_number(item[key], f"differences[{index}].{key}", minimum=0)
        if item.get("status") == "resolved" and item.get("discovery_max_abs_diff") is not None:
            if item["discovery_max_abs_diff"] <= 0:
                raise ReportError(
                    f"differences[{index}].discovery_max_abs_diff must be positive"
                )
            if item.get("verification_max_abs_diff") != 0:
                raise ReportError(
                    f"differences[{index}].verification_max_abs_diff must be zero"
                )
        if item["status"] == "resolved":
            validate_exact_gate(
                item.get("verification_gate"),
                f"differences[{index}].verification_gate",
                ("real_path_activation_proved",),
            )
        if item["status"] == "unresolved_replayed":
            replay_proof = item.get("replay_proof")
            if not isinstance(replay_proof, dict):
                raise ReportError(f"differences[{index}].replay_proof must be an object")
            if not string(replay_proof.get("manifest"), ""):
                raise ReportError(f"differences[{index}].replay_proof.manifest is required")
            for key in (
                "sha256_verified",
                "activation_proved",
                "immediate_downstream_exact",
            ):
                if replay_proof.get(key) is not True:
                    raise ReportError(f"differences[{index}].replay_proof.{key} must be true")
    for diff_id in replay_ids:
        if status_by_id.get(diff_id) != "unresolved_replayed":
            raise ReportError(
                f"results.unresolved_replay_ids contains {diff_id}, which is not unresolved_replayed"
            )
    omitted = sorted(
        diff_id
        for diff_id, status in status_by_id.items()
        if status == "unresolved_replayed" and diff_id not in replay_ids
    )
    if omitted:
        raise ReportError(
            "results.unresolved_replay_ids omits unresolved replay differences: "
            + ", ".join(omitted)
        )
    validate_metric_progression(source, ids)
    validate_acceptance_contract(source)
    validate_verification_axes(source)
    validate_validation_matrix(source)
    validate_production_runtime(source)
    validate_acceptance_closure(source)
    validate_triage(source)


def status_counts(source: dict[str, Any]) -> dict[str, int]:
    counts = {status: 0 for status in sorted(STATUSES)}
    for item in source.get("differences", []):
        counts[item["status"]] += 1
    return counts


def result_lines(source: dict[str, Any]) -> list[str]:
    results = source["results"]
    replay = "、".join(results.get("unresolved_replay_ids", [])) or "无"
    metric_text = e2e_metric_text(results)
    return [
        f"原生 Prefill：{prefill_status_label(results['native_prefill_status'])}",
        f"条件 Prefill：{prefill_status_label(results['conditional_prefill_status'])}（Replay：{replay}）",
        f"E2E 指标：{e2e_status_label(results['e2e_metric_status'])}（值：{metric_text}）",
    ]


def e2e_metric_text(results: dict[str, Any]) -> str:
    metric = results.get("e2e_metric_value")
    if metric is not None:
        return metric_value_text(metric)
    status = results.get("e2e_metric_status")
    if status == "not_run":
        return "未执行"
    if status == "evidence_missing":
        return "证据不足"
    return "未记录"


def e2e_metric_name(results: dict[str, Any]) -> str:
    return string(results.get("e2e_metric_name"), "E2E 指标")


def metric_value_text(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return scalar(value)
    rendered = format(Decimal(str(value)), "f")
    if "." not in rendered:
        return rendered + ".0"
    rendered = rendered.rstrip("0")
    return rendered if not rendered.endswith(".") else rendered + "0"


def metric_scope_label(raw: Any) -> str:
    return label(raw, METRIC_SCOPE_LABELS)


def metric_related_text(entry: dict[str, Any]) -> str:
    return "、".join(entry.get("related_difference_ids", [])) or "无"


def axis_status_label(raw: Any) -> str:
    return label(raw, AXIS_STATUS_LABELS)


def difference_before_after_text(item: dict[str, Any]) -> str:
    before = item.get("discovery_max_abs_diff")
    after = item.get("verification_max_abs_diff")
    if before is None and after is None:
        return scalar(item.get("max_abs_diff"))
    return f"{scalar(before)} → {scalar(after)}"


def validation_run_text(entry: dict[str, Any]) -> str:
    expected = scalar(entry.get("expected_steps"), "—")
    observed = scalar(entry.get("observed_steps"), "—")
    return (
        "{}；steps={}/{}；complete={}；跨历史失败点={}；max_abs={}；"
        "nonzero={}；nonfinite={}；证据={}"
    ).format(
        axis_status_label(entry.get("status")),
        observed,
        expected,
        scalar(entry.get("complete"), "—"),
        scalar(entry.get("crossed_historical_failure_step"), "—"),
        scalar(entry.get("max_abs_diff"), "—"),
        scalar(entry.get("nonzero_count"), "—"),
        scalar(entry.get("nonfinite_count"), "—"),
        string(entry.get("evidence")),
    )


def comparison_line(source: dict[str, Any]) -> str:
    comparison = object_value(source, "comparison")
    if not comparison:
        return "Prompt 对齐口径：未记录"
    return "Prompt 对齐口径：训练输入={}，Prompt={}，推理 Prefill={}；训练侧截取={}；因果位移={}".format(
        scalar(comparison.get("train_input_token_length")),
        scalar(comparison.get("prompt_token_length")),
        scalar(comparison.get("infer_prefill_token_length")),
        string(comparison.get("train_prompt_slice")),
        string(comparison.get("logit_causal_shift")),
    )


def triage_summary_line(source: dict[str, Any]) -> str:
    triage = object_value(source, "triage")
    if not triage:
        return "结构预分析：未记录（正式结论仍以精确 Tensor 路径为准）"
    return (
        "结构预分析：状态={}，候选={}，结构误报候选={}，已排除={}；"
        "仅用于预分析，不作为正式真值结论"
    ).format(
        label(triage.get("status"), TRIAGE_STATUS_LABELS),
        len(triage.get("candidates", [])),
        len(triage.get("structural_false_positive_candidates", [])),
        len(triage.get("excluded_candidates", [])),
    )


def render_triage_md(source: dict[str, Any]) -> list[str]:
    triage = object_value(source, "triage")
    if not triage:
        return []
    lines = [
        "",
        "## 结构预分析（非正式结论）",
        "",
        f"- 状态：{label(triage['status'], TRIAGE_STATUS_LABELS)}",
        "- 证据角色：仅用于预分析",
        "- 候选排名不定义首个差异；正式精确比较仍严格遵循语义执行顺序。",
        "",
        "| 候选 | 排名 | 映射 | 语义边界 | 观测状态 | 根因假设 | 上一个精确边界 |",
        "|---|---:|---|---|---|---|---|",
    ]
    for candidate in triage.get("candidates", []):
        previous = candidate.get("previous_exact_boundary")
        previous_text = (
            string(previous.get("semantic_node"))
            if isinstance(previous, dict)
            else "无"
        )
        lines.append(
            "| {candidate_id} | {rank} | {mapping} | {boundary} | {observation} | "
            "{classification} | {previous} |".format(
                candidate_id=string(candidate.get("candidate_id")).replace("|", "\\|"),
                rank=scalar(candidate.get("priority_rank")),
                mapping=string(candidate.get("mapping_id")).replace("|", "\\|"),
                boundary=string(candidate.get("semantic_node")).replace("|", "\\|"),
                observation=label(
                    candidate.get("observed_output_state"), TRIAGE_OUTPUT_LABELS
                ).replace("|", "\\|"),
                classification=label(
                    candidate.get("classification"), TRIAGE_CLASS_LABELS
                ).replace("|", "\\|"),
                previous=previous_text.replace("|", "\\|"),
            )
        )
    if not triage.get("candidates"):
        lines.append("| 无 | - | - | - | - | - | - |")

    source_info = object_value(triage, "source")
    if source_info:
        lines.extend(["", "### 预分析来源", ""])
        for key, value in source_info.items():
            rendered = (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else scalar(value)
            )
            lines.append(f"- {key}: `{rendered}`")

    lines.extend(["", "### 候选证据", ""])
    if triage.get("candidates"):
        for candidate in triage["candidates"]:
            train_unmatched = [
                string(item.get("key"))
                for item in candidate.get("unmatched_train_between_boundary", [])
                if isinstance(item, dict)
            ]
            infer_unmatched = [
                string(item.get("key"))
                for item in candidate.get("unmatched_infer_between_boundary", [])
                if isinstance(item, dict)
            ]
            alternatives = candidate.get("alternative_hypotheses", [])
            unproved = [
                string(item.get("semantic_node"))
                for item in candidate.get("unproved_mappings_between", [])
                if isinstance(item, dict)
            ]
            lines.extend(
                [
                    f"#### {string(candidate.get('candidate_id'))} - {string(candidate.get('semantic_node'))}",
                    "",
                    f"- 证据强度：{label(candidate.get('evidence_strength'), EVIDENCE_STRENGTH_LABELS)}",
                    f"- 保留原因：{string(candidate.get('reason_kept'))}",
                    f"- 未匹配训练节点：`{', '.join(train_unmatched) or '无'}`",
                    f"- 未匹配推理节点：`{', '.join(infer_unmatched) or '无'}`",
                    f"- 上一个精确边界之后仍未证明的映射：`{', '.join(unproved) or '无'}`",
                    f"- 备选假设：`{', '.join(str(item) for item in alternatives) or '无'}`",
                    "- 仍需正式精确验证：是",
                    "",
                ]
            )
    else:
        lines.append("- 未记录候选。")

    structural = triage.get("structural_false_positive_candidates", [])
    lines.extend(["", "### 结构误报候选", ""])
    if structural:
        for item in structural:
            lines.append(
                "- `{}` / `{}`: {}".format(
                    string(item.get("mapping_id")),
                    string(item.get("mapping_relation")),
                    string(item.get("disposition")),
                )
            )
    else:
        lines.append("- 无。")

    excluded = triage.get("excluded_candidates", [])
    lines.extend(["", "### 已排除候选", ""])
    if excluded:
        for item in excluded:
            lines.append(
                "- `{}`: {}".format(
                    string(item.get("mapping_id")), string(item.get("reason"))
                )
            )
    else:
        lines.append("- 无。")

    lines.extend(["", "### 后续检查", ""])
    checks = triage.get("next_checks", [])
    lines.extend(f"- {item}" for item in checks)
    return lines


def render_triage_html(source: dict[str, Any]) -> str:
    triage = object_value(source, "triage")
    if not triage:
        return ""
    rows = []
    for candidate in triage.get("candidates", []):
        previous = candidate.get("previous_exact_boundary")
        previous_text = (
            string(previous.get("semantic_node"))
            if isinstance(previous, dict)
            else "无"
        )
        rows.append(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
            "<td>{}</td></tr>".format(
                html.escape(string(candidate.get("candidate_id"))),
                html.escape(scalar(candidate.get("priority_rank"))),
                html.escape(string(candidate.get("mapping_id"))),
                html.escape(string(candidate.get("semantic_node"))),
                html.escape(
                    label(candidate.get("classification"), TRIAGE_CLASS_LABELS)
                ),
                html.escape(previous_text),
            )
        )
    if not rows:
        rows.append("<tr><td colspan='6'>未记录预分析候选。</td></tr>")

    structural = "".join(
        "<li><code>{}</code>: {}</li>".format(
            html.escape(string(item.get("mapping_id"))),
            html.escape(string(item.get("disposition"))),
        )
        for item in triage.get("structural_false_positive_candidates", [])
    )
    excluded = "".join(
        "<li><code>{}</code>: {}</li>".format(
            html.escape(string(item.get("mapping_id"))),
            html.escape(string(item.get("reason"))),
        )
        for item in triage.get("excluded_candidates", [])
    )
    checks = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in triage.get("next_checks", [])
    )
    details = []
    for candidate in triage.get("candidates", []):
        train_unmatched = ", ".join(
            string(item.get("key"))
            for item in candidate.get("unmatched_train_between_boundary", [])
            if isinstance(item, dict)
        ) or "无"
        infer_unmatched = ", ".join(
            string(item.get("key"))
            for item in candidate.get("unmatched_infer_between_boundary", [])
            if isinstance(item, dict)
        ) or "无"
        alternatives = ", ".join(
            str(item) for item in candidate.get("alternative_hypotheses", [])
        ) or "无"
        unproved = ", ".join(
            string(item.get("semantic_node"))
            for item in candidate.get("unproved_mappings_between", [])
            if isinstance(item, dict)
        ) or "无"
        details.append(
            "<details class='card'><summary>{} - {}</summary>"
            "<p><b>证据强度：</b>{}</p>"
            "<p><b>保留原因：</b>{}</p>"
            "<p><b>未匹配训练节点：</b>{}</p>"
            "<p><b>未匹配推理节点：</b>{}</p>"
            "<p><b>上一个精确边界之后仍未证明的映射：</b>{}</p>"
            "<p><b>备选假设：</b>{}</p>"
            "<p><b>仍需正式精确验证：</b>是</p></details>".format(
                html.escape(string(candidate.get("candidate_id"))),
                html.escape(string(candidate.get("semantic_node"))),
                html.escape(
                    label(candidate.get("evidence_strength"), EVIDENCE_STRENGTH_LABELS)
                ),
                html.escape(string(candidate.get("reason_kept"))),
                html.escape(train_unmatched),
                html.escape(infer_unmatched),
                html.escape(unproved),
                html.escape(alternatives),
            )
        )
    source_info = object_value(triage, "source")
    provenance = "".join(
        "<li><code>{}</code>: {}</li>".format(
            html.escape(str(key)),
            html.escape(
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else scalar(value)
            ),
        )
        for key, value in source_info.items()
    )
    return (
        "<h2>结构预分析 <small>（非正式结论）</small></h2>"
        "<section class='card'><p><b>状态：</b>{} &middot; "
        "<b>证据角色：</b>仅用于预分析</p>"
        "<p>候选排名只用于辅助调查；正式精确比较仍遵循语义执行顺序。</p></section>"
        "<div class='table-wrap'><table><thead><tr><th>候选</th><th>排名</th>"
        "<th>映射</th><th>语义边界</th><th>根因假设</th><th>上一个精确边界</th>"
        "</tr></thead><tbody>{}</tbody></table></div>"
        "<section class='card'><h3>预分析来源</h3><ul>{}</ul></section>{}"
        "<section class='card'><h3>结构误报候选</h3><ul>{}</ul>"
        "<h3>已排除候选</h3><ul>{}</ul><h3>后续检查</h3><ul>{}</ul></section>"
    ).format(
        html.escape(label(triage.get("status"), TRIAGE_STATUS_LABELS)),
        "".join(rows),
        provenance or "<li>未记录。</li>",
        "".join(details),
        structural or "<li>无。</li>",
        excluded or "<li>无。</li>",
        checks or "<li>无。</li>",
    )


def artifact_lines(source: dict[str, Any]) -> list[str]:
    artifacts = object_value(source, "artifacts")
    if not artifacts:
        return ["产物：未记录"]
    return ["产物："] + [
        f"  - {ARTIFACT_LABELS.get(key, key)}：{scalar(value)}"
        for key, value in artifacts.items()
    ]


def code_blocks(item: dict[str, Any]) -> str:
    blocks = []
    for code in item.get("code_differences", []):
        if not isinstance(code, dict):
            continue
        label = "{} — {}::{}".format(
            string(code.get("side")), string(code.get("file")), string(code.get("symbol"))
        )
        blocks.append(f"#### {label}\n\n```python\n{string(code.get('snippet'), '')}\n```")
    return "\n\n".join(blocks) or "未记录可公开的代码片段。"


def render_md(source: dict[str, Any]) -> str:
    scope = source["scope"]
    results = source["results"]
    timing = normalized_timing(source.get("timing"))
    counts = status_counts(source)
    comparison = object_value(source, "comparison")
    progression = object_value(source, "metric_progression")
    replay_ids = "、".join(results.get("unresolved_replay_ids", [])) or "无"
    lines = [
        f"# {string(source.get('title'), '训推 Prefill 一致性报告')}",
        "",
        "> 本报告严格区分原生精确对齐、Replay 条件结果、待验证候选和证据不足。",
        "",
        "## 结论概览",
        "",
        "| 检查项 | 结果 | 补充说明 |",
        "|---|---|---|",
        "| 原生 Prefill | **{}** | 关闭所有 Replay 后的真实结果 |".format(
            prefill_status_label(results.get("native_prefill_status"))
        ),
        "| 条件 Prefill | **{}** | 未解决 Replay：{} |".format(
            prefill_status_label(results.get("conditional_prefill_status")),
            replay_ids,
        ),
        "| E2E 指标（{}） | **{}** | 值：{} |".format(
            e2e_metric_name(results).replace("|", "\\|"),
            e2e_status_label(results.get("e2e_metric_status")),
            e2e_metric_text(results),
        ),
    ]
    contract = object_value(source, "acceptance_contract")
    if contract:
        lines.extend(
            [
                "",
                "## 目标与验收契约",
                "",
                f"- 总体状态：{axis_status_label(contract.get('status'))}",
                f"- 最终目标：{string(contract.get('objective'))}",
                f"- 目标模型范围：{string(contract.get('requested_model_scope'))}",
                "- 必过门禁：" + "、".join(contract.get("required_gates", [])),
                f"- 要求长跑：{scalar(contract.get('long_horizon_required'))}",
                f"- 历史失败 Step：{scalar(contract.get('historical_failure_step'), '—')}",
                f"- 停止条件：{string(contract.get('stop_condition'))}",
            ]
        )
    axes = object_value(source, "verification_axes")
    if axes:
        lines.extend(
            [
                "",
                "## 多维真值状态",
                "",
                "| 维度 | 状态 | 证据 |",
                "|---|---|---|",
            ]
        )
        for axis in AXIS_STATUSES:
            record = object_value(axes, axis)
            lines.append(
                "| {} | {} | {} |".format(
                    AXIS_LABELS[axis],
                    axis_status_label(record.get("status")),
                    string(record.get("evidence")).replace("|", "\\|"),
                )
            )
    matrix = object_value(source, "validation_matrix")
    if matrix:
        lines.extend(
            [
                "",
                "## 分层验证矩阵",
                "",
                "| Run | 门禁 | 模型范围 | Response | Steps | 完整 | 跨历史点 | 状态 | 最大差 | 非零/非有限 |",
                "|---|---|---|---:|---:|---|---|---|---:|---:|",
            ]
        )
        for entry in matrix.get("entries", []):
            lines.append(
                "| {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {}/{} |".format(
                    string(entry.get("run_id")),
                    VALIDATION_GATE_LABELS.get(entry.get("gate"), string(entry.get("gate"))),
                    string(entry.get("model_scope")).replace("|", "\\|"),
                    scalar(entry.get("response_length"), "—"),
                    scalar(entry.get("observed_steps"), "—"),
                    scalar(entry.get("expected_steps"), "—"),
                    scalar(entry.get("complete"), "—"),
                    scalar(entry.get("crossed_historical_failure_step"), "—"),
                    axis_status_label(entry.get("status")),
                    scalar(entry.get("max_abs_diff"), "—"),
                    scalar(entry.get("nonzero_count"), "—"),
                    scalar(entry.get("nonfinite_count"), "—"),
                )
            )
        lines.extend(["", "验证矩阵证据："])
        for entry in matrix.get("entries", []):
            lines.append(
                "- `{}`：{}".format(
                    string(entry.get("run_id")),
                    string(entry.get("evidence")),
                )
            )
    profile = object_value(source, "production_alignment_profile")
    clean = object_value(source, "clean_run_gate")
    if profile or clean:
        lines.extend(["", "## 生产激活与干净运行", ""])
    if profile:
        lines.extend(
            [
                f"- Profile 激活：{scalar(profile.get('active'))}",
                "- 生产特性：" + ("、".join(profile.get("features", [])) or "无"),
                "- 诊断 Replay/Capture：{}/{}".format(
                    scalar(profile.get("diagnostic_replay_enabled")),
                    scalar(profile.get("diagnostic_capture_enabled")),
                ),
                f"- 激活证据：{string(profile.get('activation_evidence'))}",
            ]
        )
    if clean:
        lines.extend(
            [
                "- 干净运行："
                + "；".join(
                    f"{key}={scalar(clean.get(key))}"
                    for key in (
                        "fresh_process",
                        "fresh_weights_loaded",
                        "diagnostic_replay_disabled",
                        "diagnostic_capture_disabled",
                        "activation_proved",
                    )
                ),
                f"- 干净运行证据：{string(clean.get('evidence'))}",
            ]
        )
    lines.extend(
        [
            "",
            "## 诊断范围",
            "",
            "| 项目 | 内容 |",
            "|---|---|",
            f"| 模型 | {string(scope.get('model')).replace('|', PIPE_ESCAPE)} |",
            f"| 覆盖范围 | {string(scope.get('coverage')).replace('|', PIPE_ESCAPE)} |",
            f"| 训练后端 | {string(scope.get('training_backend')).replace('|', PIPE_ESCAPE)} |",
            f"| 推理后端 | {string(scope.get('inference_backend')).replace('|', PIPE_ESCAPE)} |",
            f"| 硬件后端 | {string(scope.get('hardware_backend')).replace('|', PIPE_ESCAPE)} |",
            f"| 证据提供方式 | {string(scope.get('evidence_provider')).replace('|', PIPE_ESCAPE)} |",
            f"| 并行拓扑 | {topology_text(scope)} |",
            "",
            "### Prompt 对齐口径",
            "",
            "| 项目 | 内容 |",
            "|---|---|",
            f"| 训练输入长度 | {scalar(comparison.get('train_input_token_length'))} |",
            f"| Prompt 长度 | {scalar(comparison.get('prompt_token_length'))} |",
            f"| 推理 Prefill 长度 | {scalar(comparison.get('infer_prefill_token_length'))} |",
            f"| 训练侧截取 | {string(comparison.get('train_prompt_slice')).replace('|', PIPE_ESCAPE)} |",
            f"| Logit/Logprob 因果位移 | {string(comparison.get('logit_causal_shift')).replace('|', PIPE_ESCAPE)} |",
        ]
    )
    if progression:
        lines.extend(
            [
                "",
                f"## E2E 指标演进：`{string(progression.get('metric_name'))}`",
                "",
                "| 阶段 | 运行口径 | Step | 指标值 | 累积关联差异 |",
                "|---|---|---:|---:|---|",
            ]
        )
        for entry in progression.get("entries", []):
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    string(entry.get("label")).replace("|", "\\|"),
                    metric_scope_label(entry.get("run_scope")),
                    scalar(entry.get("step"), "—"),
                    metric_value_text(entry.get("value")),
                    metric_related_text(entry),
                )
            )
        lines.extend(
            [
                "",
                f"> 证据来源：{string(progression.get('evidence'))}",
                "",
                f"> 口径说明：{string(progression.get('note'))}",
            ]
        )
    lines.extend(
        [
            "",
            "## 差异总览",
            "",
            "| ID | 语义边界 | 观测条件 | 根因 | 状态 | 最大绝对差 |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for item in source.get("differences", []):
        lines.append(
            "| {id} | {boundary} | {obs} | {cls} | {status} | {diff} |".format(
                id=item["diff_id"],
                boundary=string(item.get("semantic_boundary")).replace("|", "\\|"),
                obs=observation_label(item.get("observation")),
                cls=root_cause_label(item.get("root_cause_class")).replace("|", "\\|"),
                status=status_label(item.get("status")),
                diff=difference_before_after_text(item),
            )
        )
    lines.extend(["", "### 一句话摘要", ""])
    for item in source.get("differences", []):
        lines.append(
            "- **{} · {}**：{}".format(
                item["diff_id"],
                status_label(item.get("status")),
                string(item.get("summary")),
            )
        )
    lines.extend(
        [
            "",
            "状态计数：" + "，".join(
                f"{STATUS_LABELS[key]} `{value}`" for key, value in counts.items()
            ),
        ]
    )
    lines.extend(render_triage_md(source))
    lines.extend(
        [
            "",
            "## 耗时",
            "",
            "| 项目 | 数值 |",
            "|---|---:|",
        ]
    )
    for key, value in ordered_timing_items(timing):
        rendered = seconds(value) if key.endswith("_seconds") else scalar(value)
        lines.append(f"| {TIMING_LABELS.get(key, key)} | {rendered} |")
    if timing.get("estimated_parallel_saving_seconds") is not None:
        lines.append(
            "\n计算公式：`修复 Agent 累计时间 - 修复波次墙钟时间`。"
        )

    lines.extend(["", "## 差异详情", ""])
    for item in source.get("differences", []):
        deps = "、".join(item.get("upstream_replay_ids", [])) or "无"
        conditions = "；".join(item.get("upstream_conditions", [])) or "无"
        lines.extend(
            [
                f"### {item['diff_id']} — {string(item.get('semantic_boundary'))}",
                "",
                "| 项目 | 内容 |",
                "|---|---|",
                f"| 状态 | {status_label(item.get('status'))} |",
                f"| 观测条件 | {observation_label(item.get('observation'))} |",
                "| 根因 | {}（{}） |".format(
                    root_cause_label(item.get("root_cause_class")),
                    confidence_label(item.get("root_cause_confidence")),
                ),
                f"| 修复前 → 修复后最大绝对差 | {difference_before_after_text(item)} |",
                f"| 上游 Replay | {deps} |",
                f"| 上游条件 | {conditions.replace('|', PIPE_ESCAPE)} |",
                "",
                "#### 差异摘要",
                "",
                string(item.get("summary")),
                "",
                "#### 证据",
                "",
                string(item.get("proof")),
                "",
                "#### 候选修复或处理结果",
                "",
                string(item.get("repair_result")),
                "",
                "#### Replay 说明",
                "",
                string(item.get("replay_summary")),
                "",
                "#### 验证状态",
                "",
                string(item.get("verification")),
                "",
                "#### 代码对比",
                "",
                code_blocks(item),
            ]
        )
        attempts = item.get("rejected_attempts", [])
        if attempts:
            lines.extend(["", "#### 已拒绝尝试", ""])
            lines.extend(f"- {attempt}" for attempt in attempts)
        limits = item.get("limitations", [])
        if limits:
            lines.extend(["", "#### 当前限制", ""])
            lines.extend(f"- {limit}" for limit in limits)
        lines.append("")

    provenance = source.get("provenance", {})
    if provenance:
        lines.extend(["## 版本与证据来源", "", "| 项目 | 内容 |", "|---|---|"])
        for key, value in provenance.items():
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
            else:
                rendered = scalar(value)
            lines.append(
                f"| {PROVENANCE_LABELS.get(key, key)} | {rendered.replace('|', PIPE_ESCAPE)} |"
            )
        lines.append("")
    lines.extend(["## 报告产物", "", "| 项目 | 内容 |", "|---|---|"])
    for key, value in object_value(source, "artifacts").items():
        lines.append(
            "| {} | {} |".format(
                ARTIFACT_LABELS.get(key, key),
                scalar(value).replace("|", "\\|"),
            )
        )
    limitations = source.get("limitations", [])
    if limitations:
        lines.extend(["", "## 总体限制", ""])
        lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "---",
            "",
            "说明：固定工件探针、统计摘要或 Replay 条件结果不能单独证明原生 Prefill 已精确对齐。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_html(source: dict[str, Any]) -> str:
    scope = source["scope"]
    results = source["results"]
    timing = normalized_timing(source.get("timing"))
    counts = status_counts(source)

    summary_rows = []
    detail_cards = []
    for item_index, item in enumerate(source.get("differences", [])):
        summary_rows.append(
            "<tr><td><a href='#{}'>{}</a></td><td>{}</td><td>{}</td><td>{}</td>"
            "<td><span class='status {}'>{}</span></td><td>{}</td></tr>".format(
                html.escape(item["diff_id"]),
                html.escape(item["diff_id"]),
                html.escape(string(item.get("semantic_boundary"))),
                html.escape(observation_label(item.get("observation"))),
                html.escape(root_cause_label(item.get("root_cause_class"))),
                html.escape(item["status"]),
                html.escape(status_label(item["status"])),
                html.escape(difference_before_after_text(item)),
            )
        )
        code_html = []
        for code in item.get("code_differences", []):
            if not isinstance(code, dict):
                continue
            code_html.append(
                "<h4>{}</h4><pre><code>{}</code></pre>".format(
                    html.escape(
                        "{} — {}::{}".format(
                            string(code.get("side")), string(code.get("file")), string(code.get("symbol"))
                        )
                    ),
                    html.escape(string(code.get("snippet"), "")),
                )
            )
        attempts = "".join(f"<li>{html.escape(str(x))}</li>" for x in item.get("rejected_attempts", []))
        limits = "".join(f"<li>{html.escape(str(x))}</li>" for x in item.get("limitations", []))
        deps = "、".join(item.get("upstream_replay_ids", [])) or "无"
        conditions = "；".join(item.get("upstream_conditions", [])) or "无"
        open_attr = " open" if item_index == 0 else ""
        detail_cards.append(
            textwrap.dedent(
                f"""
            <details class="card difference-card"{open_attr} id="{html.escape(item['diff_id'])}">
              <summary><span>{html.escape(item['diff_id'])} · {html.escape(string(item.get('semantic_boundary')))}</span><span class="status {html.escape(item['status'])}">{html.escape(status_label(item['status']))}</span></summary>
              <div class="meta-grid">
                <div><span>观测条件</span><b>{html.escape(observation_label(item['observation']))}</b></div>
                <div><span>根因</span><b>{html.escape(root_cause_label(item.get('root_cause_class')))}（{html.escape(confidence_label(item.get('root_cause_confidence')))}）</b></div>
                <div><span>修复前 → 修复后最大绝对差</span><b>{html.escape(difference_before_after_text(item))}</b></div>
                <div><span>上游 Replay</span><b>{html.escape(deps)}</b></div>
                <div class="wide"><span>上游条件</span><b>{html.escape(conditions)}</b></div>
                <div><span>耗时</span><b>{html.escape(seconds(item.get('duration_seconds')))}</b></div>
              </div>
              <section class="narrative"><h4>差异摘要</h4><p>{html.escape(string(item.get('summary')))}</p></section>
              <section class="narrative"><h4>证据</h4><p>{html.escape(string(item.get('proof')))}</p></section>
              <section class="narrative"><h4>候选修复或处理结果</h4><p>{html.escape(string(item.get('repair_result')))}</p></section>
              <section class="narrative"><h4>Replay 说明</h4><p>{html.escape(string(item.get('replay_summary')))}</p></section>
              <section class="narrative"><h4>验证状态</h4><p>{html.escape(string(item.get('verification')))}</p></section>
              <h4>代码对比</h4>
              {''.join(code_html) or '<p>未记录可公开的代码片段。</p>'}
              {'<h4>已拒绝尝试</h4><ul>' + attempts + '</ul>' if attempts else ''}
              {'<h4>当前限制</h4><ul>' + limits + '</ul>' if limits else ''}
            </details>
            """
            ).strip()
        )

    replay = "、".join(results.get("unresolved_replay_ids", [])) or "无"
    counts_html = "".join(
        "<span class='pill'>{}: {}</span>".format(
            html.escape(STATUS_LABELS[key]), value
        )
        for key, value in counts.items()
        if value
    )
    timing_html = "".join(
        "<div><span>{}</span><b>{}</b></div>".format(
            html.escape(TIMING_LABELS.get(key, key)),
            html.escape(seconds(value) if key.endswith("_seconds") else scalar(value)),
        )
        for key, value in ordered_timing_items(timing)
    )
    limitations = "".join(f"<li>{html.escape(str(x))}</li>" for x in source.get("limitations", []))
    artifacts_html = "".join(
        f"<div><span>{html.escape(ARTIFACT_LABELS.get(k, str(k)))}</span><b>{html.escape(scalar(v))}</b></div>"
        for k, v in object_value(source, "artifacts").items()
    )
    provenance_html = "".join(
        f"<div><span>{html.escape(PROVENANCE_LABELS.get(k, str(k)))}</span><b>{html.escape(scalar(v))}</b></div>"
        for k, v in object_value(source, "provenance").items()
    )
    comparison = object_value(source, "comparison")
    progression = object_value(source, "metric_progression")
    comparison_html = html.escape(comparison_line(source))
    scope_html = "".join(
        "<div><span>{}</span><b>{}</b></div>".format(
            html.escape(label_text), html.escape(scalar(value))
        )
        for label_text, value in (
            ("模型", scope.get("model")),
            ("覆盖范围", scope.get("coverage")),
            ("训练后端", scope.get("training_backend")),
            ("推理后端", scope.get("inference_backend")),
            ("硬件后端", scope.get("hardware_backend")),
            ("证据提供方式", scope.get("evidence_provider")),
            ("并行拓扑", topology_text(scope)),
            ("训练输入长度", comparison.get("train_input_token_length")),
            ("Prompt 长度", comparison.get("prompt_token_length")),
            ("推理 Prefill 长度", comparison.get("infer_prefill_token_length")),
            ("训练侧截取", comparison.get("train_prompt_slice")),
            ("Logit/Logprob 因果位移", comparison.get("logit_causal_shift")),
        )
    )
    contract = object_value(source, "acceptance_contract")
    contract_html = ""
    if contract:
        contract_html = (
            "<h2>目标与验收契约</h2><section class='info-grid'>"
            "<div><span>总体状态</span><b>{}</b></div>"
            "<div><span>最终目标</span><b>{}</b></div>"
            "<div><span>目标模型范围</span><b>{}</b></div>"
            "<div><span>必过门禁</span><b>{}</b></div>"
            "<div><span>要求长跑</span><b>{}</b></div>"
            "<div><span>历史失败 Step</span><b>{}</b></div>"
            "<div><span>停止条件</span><b>{}</b></div></section>"
        ).format(
            html.escape(axis_status_label(contract.get("status"))),
            html.escape(string(contract.get("objective"))),
            html.escape(string(contract.get("requested_model_scope"))),
            html.escape("、".join(contract.get("required_gates", []))),
            html.escape(scalar(contract.get("long_horizon_required"))),
            html.escape(scalar(contract.get("historical_failure_step"), "—")),
            html.escape(string(contract.get("stop_condition"))),
        )
    axes = object_value(source, "verification_axes")
    axes_html = ""
    if axes:
        rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(AXIS_LABELS[axis]),
                html.escape(axis_status_label(object_value(axes, axis).get("status"))),
                html.escape(string(object_value(axes, axis).get("evidence"))),
            )
            for axis in AXIS_STATUSES
        )
        axes_html = (
            "<h2>多维真值状态</h2><div class='table-wrap'><table><thead>"
            "<tr><th>维度</th><th>状态</th><th>证据</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    matrix = object_value(source, "validation_matrix")
    matrix_html = ""
    if matrix:
        rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}/{}</td><td>{}</td></tr>".format(
                html.escape(string(entry.get("run_id"))),
                html.escape(VALIDATION_GATE_LABELS.get(entry.get("gate"), string(entry.get("gate")))),
                html.escape(string(entry.get("model_scope"))),
                html.escape(scalar(entry.get("response_length"), "—")),
                html.escape(scalar(entry.get("observed_steps"), "—")),
                html.escape(scalar(entry.get("expected_steps"), "—")),
                html.escape(scalar(entry.get("complete"), "—")),
                html.escape(scalar(entry.get("crossed_historical_failure_step"), "—")),
                html.escape(axis_status_label(entry.get("status"))),
                html.escape(scalar(entry.get("max_abs_diff"), "—")),
                html.escape(scalar(entry.get("nonzero_count"), "—")),
                html.escape(scalar(entry.get("nonfinite_count"), "—")),
                html.escape(string(entry.get("evidence"))),
            )
            for entry in matrix.get("entries", [])
        )
        matrix_html = (
            "<h2>分层验证矩阵</h2><div class='table-wrap'><table><thead><tr>"
            "<th>Run</th><th>门禁</th><th>模型范围</th><th>Response</th>"
            "<th>Steps</th><th>完整</th><th>跨历史点</th><th>状态</th>"
            "<th>最大差</th><th>非零/非有限</th><th>证据</th>"
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )
    profile = object_value(source, "production_alignment_profile")
    clean = object_value(source, "clean_run_gate")
    production_html = ""
    if profile or clean:
        items: list[tuple[str, Any]] = []
        if profile:
            items.extend(
                [
                    ("Profile 激活", profile.get("active")),
                    ("生产特性", "、".join(profile.get("features", [])) or "无"),
                    (
                        "诊断 Replay/Capture",
                        "{}/{}".format(
                            scalar(profile.get("diagnostic_replay_enabled")),
                            scalar(profile.get("diagnostic_capture_enabled")),
                        ),
                    ),
                    ("激活证据", profile.get("activation_evidence")),
                ]
            )
        if clean:
            items.extend(
                [
                    (
                        "干净运行门禁",
                        "；".join(
                            f"{key}={scalar(clean.get(key))}"
                            for key in (
                                "fresh_process",
                                "fresh_weights_loaded",
                                "diagnostic_replay_disabled",
                                "diagnostic_capture_disabled",
                                "activation_proved",
                            )
                        ),
                    ),
                    ("干净运行证据", clean.get("evidence")),
                ]
            )
        cells = "".join(
            "<div><span>{}</span><b>{}</b></div>".format(
                html.escape(label_text), html.escape(scalar(value))
            )
            for label_text, value in items
        )
        production_html = (
            "<h2>生产激活与干净运行</h2>"
            f"<section class='info-grid'>{cells}</section>"
        )
    progression_html = ""
    if progression:
        metric_rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td>"
            "<td>{}</td></tr>".format(
                html.escape(string(entry.get("label"))),
                html.escape(metric_scope_label(entry.get("run_scope"))),
                html.escape(scalar(entry.get("step"), "—")),
                html.escape(metric_value_text(entry.get("value"))),
                html.escape(metric_related_text(entry)),
            )
            for entry in progression.get("entries", [])
        )
        progression_html = (
            "<h2>E2E 指标演进：<code>{}</code></h2>"
            "<div class='table-wrap'><table><thead><tr><th>阶段</th>"
            "<th>运行口径</th><th>Step</th><th>指标值</th><th>累积关联差异</th>"
            "</tr></thead><tbody>{}</tbody></table></div>"
            "<section class='card metric-note'><p><b>证据来源：</b>{}</p>"
            "<p><b>口径说明：</b>{}</p></section>"
        ).format(
            html.escape(string(progression.get("metric_name"))),
            metric_rows,
            html.escape(string(progression.get("evidence"))),
            html.escape(string(progression.get("note"))),
        )
    triage_html = render_triage_html(source)
    language = "zh-CN"
    return f"""<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(string(source.get('title'), '训推 Prefill 一致性报告'))}</title>
<style>
:root{{--bg:#f3f6fb;--panel:#fff;--text:#172033;--muted:#667085;--line:#d8e0ec;--accent:#275dad;--accent-soft:#edf4ff;--ok:#16794f;--warn:#a15c00;--danger:#b42318}}
*{{box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.7 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
main{{max-width:1240px;margin:auto;padding:32px 24px 64px}}
h1{{font-size:clamp(26px,4vw,40px);line-height:1.2;margin:0 0 12px}}
h2{{margin:38px 0 14px;font-size:24px}}
h3{{margin:24px 0 10px}}
h4{{margin:20px 0 8px}}
a{{color:var(--accent);text-decoration:none}} a:hover{{text-decoration:underline}}
.muted{{color:var(--muted)}} small{{color:var(--muted);font-weight:500}}
.hero,.card,.info-grid,.table-wrap{{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:0 8px 28px rgba(24,39,75,.06)}}
.hero{{padding:28px;background:linear-gradient(135deg,#fff 0%,#f3f7ff 100%);border-top:5px solid var(--accent)}}
.scope-line{{margin:5px 0;color:var(--muted);overflow-wrap:anywhere}}
.result-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:22px 0}}
.result{{padding:16px;background:rgba(255,255,255,.82);border:1px solid var(--line);border-radius:12px;min-width:0}}
.result span{{display:block;color:var(--muted);font-size:13px;margin-bottom:6px}}
.result b{{font-size:17px;overflow-wrap:anywhere}}
.notice{{margin:16px 0 0;padding:13px 15px;border-left:4px solid var(--warn);background:#fff8e8;border-radius:8px}}
.pill,.status{{display:inline-block;padding:3px 9px;margin:3px;border-radius:999px;background:#e9eef6;font-size:12px;white-space:nowrap}}
.status.resolved{{color:var(--ok);background:#e8f7f0}}
.status.unresolved_replayed,.status.excluded_known_gap{{color:var(--warn);background:#fff1d6}}
.status.pending{{color:#475467;background:#eef1f6}}
.status.rejected_hypothesis{{color:#6941c6;background:#f2edff}}
.table-wrap{{overflow:auto;margin:14px 0 22px}}
table{{border-collapse:collapse;width:100%;min-width:760px}}
th{{background:#f7f9fc;color:#475467;font-size:13px}}
th,td{{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}}
tr:last-child td{{border-bottom:0}}
.info-grid{{padding:18px;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 18px}}
.info-grid div,.meta-grid div{{display:flex;flex-direction:column;gap:3px;padding:10px 4px;border-bottom:1px dashed var(--line);min-width:0}}
.info-grid span,.meta-grid span{{color:var(--muted);font-size:13px}}
.info-grid b,.meta-grid b{{font-weight:600;overflow-wrap:anywhere}}
.difference-card{{padding:0 20px;margin:14px 0;scroll-margin-top:16px}}
summary{{cursor:pointer;font-weight:700;padding:18px 0;display:flex;align-items:center;justify-content:space-between;gap:12px}}
summary span:first-child{{overflow-wrap:anywhere}}
.meta-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0 18px;padding:4px 0 14px}}
.meta-grid .wide{{grid-column:span 2}}
.narrative{{border-left:3px solid #bed0ec;padding:1px 14px;margin:16px 0;background:#fafcff;border-radius:0 8px 8px 0}}
.narrative p{{margin:6px 0 12px}}
.metric-note{{padding:14px 20px}} .metric-note p{{margin:5px 0}}
pre{{overflow:auto;background:#101828;color:#e7edf5;padding:16px;border-radius:10px;line-height:1.55}}
code{{font-family:"Cascadia Code","SFMono-Regular",Consolas,monospace}}
ul{{padding-left:22px}}
.footer-note{{margin-top:30px;padding:16px;border:1px solid var(--line);border-radius:12px;color:var(--muted);background:#fff}}
@media(max-width:900px){{.result-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.info-grid,.meta-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.meta-grid .wide{{grid-column:span 2}}}}
@media(max-width:620px){{main{{padding:16px 12px 40px}}.hero{{padding:20px 16px}}.result-grid,.info-grid,.meta-grid{{grid-template-columns:1fr}}.meta-grid .wide{{grid-column:auto}}table{{min-width:680px}}summary{{align-items:flex-start;flex-direction:column}}}}
</style>
</head>
<body><main>
<section class="hero">
  <h1>{html.escape(string(source.get('title'), '训推 Prefill 一致性报告'))}</h1>
  <p class="scope-line">{html.escape(string(scope.get('model')))} · Prefill · {html.escape(topology_text(scope))}</p>
  <p class="scope-line">覆盖范围：{html.escape(string(scope.get('coverage')))}</p>
  <p class="scope-line">训练后端：{html.escape(string(scope.get('training_backend')))} · 推理后端：{html.escape(string(scope.get('inference_backend')))} · 硬件后端：{html.escape(string(scope.get('hardware_backend')))}</p>
  <div class="result-grid">
    <div class="result"><span>原生 Prefill</span><b>{html.escape(prefill_status_label(results['native_prefill_status']))}</b></div>
    <div class="result"><span>条件 Prefill</span><b>{html.escape(prefill_status_label(results['conditional_prefill_status']))}</b></div>
    <div class="result"><span>未解决 Replay</span><b>{html.escape(replay)}</b></div>
    <div class="result"><span>{html.escape(e2e_metric_name(results))}</span><b>{html.escape(e2e_status_label(results['e2e_metric_status']))} · {html.escape(e2e_metric_text(results))}</b></div>
  </div>
  <div class="notice">{comparison_html}</div>
  <p>{counts_html or '<span class="pill">未记录独立差异</span>'}</p>
</section>
{contract_html}
{axes_html}
{matrix_html}
{production_html}
<h2>诊断范围</h2>
<section class="info-grid">{scope_html}</section>
{progression_html}
{triage_html}
<h2>差异总览</h2>
<div class="table-wrap"><table><thead><tr><th>ID</th><th>语义边界</th><th>观测条件</th><th>根因</th><th>状态</th><th>最大绝对差</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div>
<h2>耗时</h2><section class="info-grid">{timing_html or '<div><span>状态</span><b>未记录</b></div>'}</section>
<h2>报告产物</h2><section class="info-grid">{artifacts_html or '<div><span>状态</span><b>未记录</b></div>'}</section>
<h2>版本与证据来源</h2><section class="info-grid">{provenance_html or '<div><span>状态</span><b>未记录</b></div>'}</section>
<h2>差异详情</h2>{''.join(detail_cards) or '<section class="card"><p>未记录独立差异。</p></section>'}
{'<h2>总体限制</h2><section class="card"><ul>' + limitations + '</ul></section>' if limitations else ''}
<p class="footer-note">说明：条件结果、固定工件探针和 Replay 结果均不会自动升级为原生精确对齐；正式结论仍以真实路径的精确 Tensor 门禁为准。</p>
</main></body></html>
"""


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", type=Path)
    mode.add_argument(
        "--init-source",
        type=Path,
        help="从 Skill 内置中文模板初始化 report_source.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--force",
        action="store_true",
        help="初始化事实源时允许覆盖目标文件",
    )
    args = parser.parse_args()
    if args.init_source is not None:
        if args.output_dir is not None:
            parser.error("--init-source 不能与 --output-dir 同时使用")
        target = args.init_source.resolve()
        if target.exists() and not args.force:
            print(f"ERROR: {target} already exists; use --force to replace it", file=sys.stderr)
            return 2
        try:
            template = load_json(SOURCE_TEMPLATE)
            validate(template)
            target.parent.mkdir(parents=True, exist_ok=True)
            write_text(target, json.dumps(template, ensure_ascii=False, indent=2) + "\n")
        except (OSError, ReportError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(
            json.dumps(
                {
                    "initialized": True,
                    "language": "zh-CN",
                    "template": str(SOURCE_TEMPLATE),
                    "output": str(target),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.output_dir is None:
        parser.error("--input 模式必须提供 --output-dir")
    try:
        source = load_json(args.input.resolve())
        validate(source)
    except ReportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_text(output / "alignment_report.md", render_md(source))
    write_text(output / "alignment_report.html", render_html(source))
    print(json.dumps({"valid": True, "output_dir": str(output), "files": sorted(p.name for p in output.iterdir())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
