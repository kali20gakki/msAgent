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

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from msagent.audit.events import AuditEvent, AuditEventType, format_audit_timestamp
from msagent.audit.protocol import parse_completion_output, parse_delegation_input
from msagent.audit.read import AuditReader, iter_json_values
from msagent.audit.tracker import SubagentAuditTracker
from msagent.audit.user_interaction import build_user_response_fields, extract_last_agent_prompt
from msagent.audit.writer import AuditWriter, build_audit_filename, resolve_audit_log_enabled
from msagent.configs import AuditLogConfig
from msagent.core.constants import CONFIG_AUDIT_DIR

MSAGENT_IO_INPUT = """\
Generate practice YAML.

```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-practice-generator",
  "input": {
    "model_type": "qwen3",
    "model_path": "/data/models/Qwen3-8B/",
    "save_path": "/tmp/record/",
    "device": "npu:2",
    "round": 1
  }
}
```"""

MSAGENT_IO_OUTPUT = """\
Practice YAML ready.

```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-practice-generator",
  "status": "ok",
  "output": {
    "practice_path": "/tmp/practice_round_1.yaml",
    "validation": { "ok": true, "valid": true, "errors": [] },
    "commands": [
      {
        "name": "sensitive_layer_analysis",
        "skipped": true,
        "reason": "analysis_result.yaml already exists"
      },
      {
        "name": "validate_practice_yaml",
        "command": "python skills/tune-practice-cfg/scripts/validate_practice_yaml.py --practice-path /tmp/practice_round_1.yaml"
      }
    ]
  }
}
```"""

PRACTICE_GENERATOR_OUTPUT = {
    "practice_path": "/tmp/practice_round_1.yaml",
    "validation": {"ok": True, "valid": True, "errors": []},
    "commands": [
        {
            "name": "sensitive_layer_analysis",
            "skipped": True,
            "reason": "analysis_result.yaml already exists",
        },
        {
            "name": "validate_practice_yaml",
            "command": "python skills/tune-practice-cfg/scripts/validate_practice_yaml.py --practice-path /tmp/practice_round_1.yaml",
        },
    ],
}


def _writer(
    tmp_path: Path,
    *,
    thread_id: str,
    enabled: bool = True,
    agent_name: str = "Auto-tuning",
) -> AuditWriter:
    return AuditWriter(
        working_dir=tmp_path,
        thread_id=thread_id,
        agent_name=agent_name,
        enabled=enabled,
    )


def _tracker(tmp_path: Path, *, thread_id: str, enabled: bool = True) -> SubagentAuditTracker:
    return SubagentAuditTracker(_writer(tmp_path, thread_id=thread_id, enabled=enabled))


def _task_call(
    *,
    subagent_type: str,
    description: str,
    call_id: str,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {"subagent_type": subagent_type, "description": description},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _task_result(*, content: str, call_id: str, status: str | None = None) -> ToolMessage:
    message = ToolMessage(content=content, tool_call_id=call_id, name="task")
    if status is not None:
        message.status = status
    return message


def _observe_delegation(
    tracker: SubagentAuditTracker,
    *,
    subagent_type: str,
    description: str,
    call_id: str,
    result: str,
    result_status: str | None = None,
) -> None:
    tracker.observe(
        _task_call(subagent_type=subagent_type, description=description, call_id=call_id),
        namespace=(),
    )
    tracker.observe(
        _task_result(content=result, call_id=call_id, status=result_status),
        namespace=(),
    )


def _audit_file(tmp_path: Path, *, thread_id: str, agent_name: str = "Auto-tuning") -> Path:
    return tmp_path / CONFIG_AUDIT_DIR / build_audit_filename(agent_name=agent_name, thread_id=thread_id)


def test_delegation_event_records_structured_io_when_protocol_valid(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-1")
    tracker.begin_run("run-1")
    _observe_delegation(
        tracker,
        subagent_type="quant-tuning-practice-generator",
        description=MSAGENT_IO_INPUT,
        call_id="call-task-1",
        result=MSAGENT_IO_OUTPUT,
    )
    tracker.observe(
        AIMessage(content="ignored", tool_calls=[{"name": "run_command", "args": {}, "id": "x"}]),
        namespace=("subagent",),
    )

    reader = AuditReader(working_dir=tmp_path, thread_id="thread-1")
    event = list(reader.iter_events())[0]
    assert event["event"] == AuditEventType.SUBAGENT_DELEGATION
    assert event["run_id"] == "run-1"
    assert event["subagent_type"] == "quant-tuning-practice-generator"
    assert event["status"] == "ok"
    assert event["input_valid"] is True
    assert event["output_valid"] is True
    assert event["input"]["round"] == 1
    assert event["output"] == PRACTICE_GENERATOR_OUTPUT
    assert "task_description_raw" not in event
    assert "result_raw" not in event

    summary = reader.list_delegations()[0]
    assert summary["output"]["practice_path"] == "/tmp/practice_round_1.yaml"


def test_delegation_event_stores_raw_text_when_protocol_missing(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-raw")
    tracker.begin_run("run-raw")
    _observe_delegation(
        tracker,
        subagent_type="quant-tuning-practice-generator",
        description="Generate practice YAML for round 1",
        call_id="call-task-raw",
        result="Generated practice_round_1.yaml",
    )

    event = list(AuditReader(working_dir=tmp_path, thread_id="thread-raw").iter_events())[0]
    assert event["input_valid"] is False
    assert event["output_valid"] is False
    assert "Generate practice YAML" in event["task_description_raw"]
    assert "practice_round_1.yaml" in event["result_raw"]
    assert "input" not in event
    assert "output" not in event


def test_delegation_event_marks_failed_when_task_errors(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-2")
    tracker.begin_run("run-2")
    tracker.observe(
        _task_result(
            content="We cannot invoke subagent missing because it does not exist",
            call_id="call-task-2",
            status="error",
        ),
        namespace=(),
    )

    event = list(AuditReader(working_dir=tmp_path, thread_id="thread-2").iter_events())[0]
    assert event["status"] == "failed"
    assert event["subagent_type"] == "unknown"


def test_audit_writer_skips_file_when_disabled(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-3", enabled=False)
    tracker.begin_run("run-3")
    _observe_delegation(
        tracker,
        subagent_type="quant-tuning-evaluator",
        description="x",
        call_id="call-task-3",
        result="done",
    )

    assert not _audit_file(tmp_path, thread_id="thread-3").exists()


def test_audit_writer_updates_path_when_thread_rebound(tmp_path: Path) -> None:
    writer = _writer(tmp_path, thread_id="thread-a")
    writer.rebind(thread_id="thread-b")
    assert writer.path.name == build_audit_filename(agent_name="Auto-tuning", thread_id="thread-b")


def test_audit_writer_writes_pretty_json_when_event_appended(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-pretty")
    tracker.begin_run("run-pretty")
    _observe_delegation(
        tracker,
        subagent_type="quant-tuning-quantizer",
        description="quantize model",
        call_id="call-pretty-1",
        result="quantized",
    )

    content = _audit_file(tmp_path, thread_id="thread-pretty").read_text(encoding="utf-8")
    payload = list(iter_json_values(content))[0]
    assert payload["event"] == AuditEventType.SUBAGENT_DELEGATION
    assert list(payload.keys())[0] == "agent_name"
    assert "timestamp" not in payload
    assert "protocol_version" not in payload
    assert '\n  "' in content


def test_delegation_event_omits_null_fields_when_serializing() -> None:
    payload = AuditEvent.delegation(
        run_id="run-1",
        agent_name="Auto-tuning",
        delegation_id="call-1",
        subagent_type="quant-tuning-evaluator",
        start_time="2026-06-02 07:42:23",
        end_time="2026-06-02 07:42:45",
        duration_ms=22000,
        status="ok",
    ).to_json_dict()
    assert list(payload.keys())[:3] == ["agent_name", "event", "subagent_type"]
    assert "input" not in payload


def test_protocol_marks_invalid_when_evaluation_generator_uses_legacy_fields() -> None:
    legacy_input = """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-evaluation-generator",
  "input": {
    "model_name": "Qwen3-8B-w8a8",
    "save_path": "/tmp/record/",
    "target_datasets": ["gpqa"],
    "accuracy_targets": {"gpqa": 79.0}
  }
}
```"""
    result = parse_delegation_input(
        legacy_input,
        expected_subagent_type="quant-tuning-evaluation-generator",
    )
    assert result.valid is False
    assert any("deprecated_fields" in error for error in result.errors)
    assert "missing_datasets" in result.errors


def test_delegation_event_records_datasets_when_evaluation_generator_input_valid(tmp_path: Path) -> None:
    eval_input = """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-evaluation-generator",
  "input": {
    "model_name": "Qwen3-8B-w8a8",
    "save_path": "/tmp/record/",
    "datasets": [
      {
        "name": "gpqa",
        "config_name": "gpqa_gen_0_shot_cot_str",
        "target": 79.0,
        "tolerance": 1.0
      }
    ],
    "device_count": 2
  }
}
```"""
    eval_output = """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-evaluation-generator",
  "status": "ok",
  "output": {
    "evaluate_config_path": "/tmp/record/evaluate.yaml"
  }
}
```"""
    tracker = _tracker(tmp_path, thread_id="thread-eval")
    tracker.begin_run("run-eval")
    _observe_delegation(
        tracker,
        subagent_type="quant-tuning-evaluation-generator",
        description=eval_input,
        call_id="call-eval-1",
        result=eval_output,
    )

    event = list(AuditReader(working_dir=tmp_path, thread_id="thread-eval").iter_events())[0]
    assert event["input_valid"] is True
    assert event["input"]["datasets"][0]["name"] == "gpqa"


def test_audit_timestamp_formats_local_wall_clock_when_given_datetime() -> None:
    assert format_audit_timestamp(now=datetime(2026, 6, 2, 7, 42, 23)) == "2026-06-02 07:42:23"


def test_protocol_extracts_structured_io_when_practice_generator_blocks_valid() -> None:
    input_result = parse_delegation_input(
        MSAGENT_IO_INPUT,
        expected_subagent_type="quant-tuning-practice-generator",
    )
    assert input_result.valid is True
    assert input_result.input_data["round"] == 1

    output_result = parse_completion_output(
        MSAGENT_IO_OUTPUT,
        expected_subagent_type="quant-tuning-practice-generator",
    )
    assert output_result.valid is True
    assert output_result.output_data == PRACTICE_GENERATOR_OUTPUT


def test_protocol_marks_invalid_when_subagent_type_mismatches() -> None:
    result = parse_delegation_input(
        MSAGENT_IO_INPUT,
        expected_subagent_type="quant-tuning-evaluator",
    )
    assert result.valid is False
    assert "subagent_type_mismatch" in result.errors


def test_protocol_marks_invalid_when_quantizer_output_missing_commands() -> None:
    result = parse_completion_output(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-quantizer",
  "status": "ok",
  "output": {
    "success": true,
    "quantized_path": "/tmp/quantized",
    "exit_code": 0
  }
}
```""",
        expected_subagent_type="quant-tuning-quantizer",
    )
    assert result.valid is False
    assert "missing_commands" in result.errors


def test_protocol_marks_valid_when_quantizer_output_includes_commands() -> None:
    result = parse_completion_output(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-quantizer",
  "status": "ok",
  "output": {
    "success": true,
    "quantized_path": "/tmp/quantized",
    "exit_code": 0,
    "commands": [
      {
        "name": "quantize",
        "command": "msmodelslim quant --model_path /m --save_path /tmp/quantized --device npu:0 --model_type Qwen3-8B --config_path /p.yaml --trust_remote_code True"
      }
    ]
  }
}
```""",
        expected_subagent_type="quant-tuning-quantizer",
    )
    assert result.valid is True


def test_protocol_marks_valid_when_evaluator_output_includes_service_commands() -> None:
    result = parse_completion_output(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-evaluator",
  "status": "ok",
  "output": {
    "overall_passed": true,
    "datasets": [{ "name": "gpqa", "score": 80.0, "target": 79.0, "passed": true }],
    "commands": [
      {
        "name": "inference_service",
        "command": "python -m vllm.entrypoints.openai.api_server --model /tmp/quantized --port 8000"
      },
      {
        "name": "evaluation",
        "command": "python skills/quant-tuning-evaluate/scripts/run_evaluation.py --quant-model-path /tmp/quantized --evaluate-id e1 --evaluate-config-path /tmp/evaluate.yaml --save-path /tmp/work --device npu --device-indices 0,1"
      }
    ]
  }
}
```""",
        expected_subagent_type="quant-tuning-evaluator",
    )
    assert result.valid is True


def test_protocol_marks_valid_when_model_analysis_io_complete() -> None:
    input_result = parse_delegation_input(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-analysis",
  "input": {
    "model_type": "Qwen3-8B",
    "model_path": "/data/models/Qwen3-8B/"
  }
}
```""",
        expected_subagent_type="msmodelslim-model-analysis",
    )
    output_result = parse_completion_output(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-analysis",
  "status": "ok",
  "output": {
    "next_step": "model-adapt",
    "implementation_source": "transformers",
    "summary": "Model uses standard transformers architecture.",
    "report_path": "/tmp/work/analysis_report.json"
  }
}
```""",
        expected_subagent_type="msmodelslim-model-analysis",
    )
    assert input_result.valid is True
    assert output_result.valid is True


def test_protocol_marks_invalid_when_model_analysis_source_unknown() -> None:
    result = parse_completion_output(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-analysis",
  "status": "ok",
  "output": {
    "implementation_source": "unknown",
    "summary": "x",
    "report_path": "/tmp/r.json",
    "next_step": "model-adapt"
  }
}
```""",
        expected_subagent_type="msmodelslim-model-analysis",
    )
    assert "invalid_implementation_source" in result.errors


def test_protocol_marks_invalid_when_model_analysis_next_step_invalid() -> None:
    result = parse_completion_output(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-analysis",
  "status": "ok",
  "output": {
    "next_step": "stop",
    "implementation_source": "transformers",
    "summary": "x",
    "report_path": "/tmp/r.json"
  }
}
```""",
        expected_subagent_type="msmodelslim-model-analysis",
    )
    assert "invalid_next_step" in result.errors


def test_protocol_marks_valid_when_model_adapt_io_complete() -> None:
    input_result = parse_delegation_input(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-adapt",
  "input": {
    "model_type": "Qwen3-8B",
    "model_path": "/data/models/Qwen3-8B/",
    "analysis_report_path": "/tmp/work/analysis_report.json"
  }
}
```""",
        expected_subagent_type="msmodelslim-model-adapt",
    )
    output_result = parse_completion_output(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-adapt",
  "status": "ok",
  "output": {
    "adapter_registered": true,
    "verification_steps": [
      { "step": 1, "name": "generate_test_model", "passed": true },
      { "step": 2, "name": "run_quantization", "passed": true },
      { "step": 3, "name": "verify_weights", "passed": true },
      { "step": 4, "name": "verify_quant_description", "passed": true }
    ],
    "artifact_paths": { "adapter_module": "msmodelslim/model/qwen3_8b.py" },
    "commands": [
      { "name": "install", "command": "pip install -e ." },
      { "name": "verification_step1", "command": "python -c 'import msmodelslim'" },
      { "name": "verification_step2", "command": "python verify_config.py" },
      { "name": "verification_step3", "command": "python verify_load.py" },
      { "name": "verification_step4", "command": "python verify_forward.py" }
    ]
  }
}
```""",
        expected_subagent_type="msmodelslim-model-adapt",
    )
    assert input_result.valid is True
    assert output_result.valid is True


def test_protocol_marks_invalid_when_model_adapt_missing_commands() -> None:
    result = parse_completion_output(
        """\
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-adapt",
  "status": "ok",
  "output": {
    "adapter_registered": true,
    "verification_steps": [
      { "step": 1, "name": "generate_test_model", "passed": true },
      { "step": 2, "name": "run_quantization", "passed": true },
      { "step": 3, "name": "load_model", "passed": false },
      { "step": 4, "name": "verify_quant_description", "passed": false }
    ],
    "artifact_paths": {}
  }
}
```""",
        expected_subagent_type="msmodelslim-model-adapt",
    )
    assert "missing_commands" in result.errors


def test_audit_log_resolves_enabled_from_agent_yaml_when_config_present() -> None:
    assert resolve_audit_log_enabled(SimpleNamespace(audit_log=AuditLogConfig(enabled=True))) is True
    assert resolve_audit_log_enabled(SimpleNamespace(audit_log=None)) is False
    assert AuditLogConfig().enabled is False


def test_agent_prompt_extractor_returns_latest_assistant_text_when_messages_present() -> None:
    prompt = extract_last_agent_prompt([HumanMessage(content="start"), AIMessage(content="请确认配置是否无误。")])
    assert prompt == "请确认配置是否无误。"


def test_user_turn_event_records_message_when_begin_run_includes_text(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-user-turn")
    tracker.begin_run("run-user-1", user_message="Tune Qwen3-8B with GPQA target 79%")

    event = list(AuditReader(working_dir=tmp_path, thread_id="thread-user-turn").iter_events())[0]
    assert event["event"] == AuditEventType.USER_TURN
    assert event["run_id"] == "run-user-1"
    assert "Qwen3-8B" in event["message"]


def test_user_response_event_records_choice_when_emitted(tmp_path: Path) -> None:
    writer = _writer(tmp_path, thread_id="thread-user-response")
    writer.begin_run("run-response-1")
    writer.emit_user_response(
        kind="choice",
        prompt="Continue Round 3?",
        options=["continue", "stop"],
        response="continue",
        context={"interrupt_id": "interrupt-1"},
    )

    event = list(AuditReader(working_dir=tmp_path, thread_id="thread-user-response").iter_events())[0]
    assert event["event"] == AuditEventType.USER_RESPONSE
    assert event["response"] == "continue"


def test_user_response_fields_map_hitl_reject_when_execute_interrupt() -> None:
    class _Interrupt:
        id = "int-1"
        value = {
            "action_requests": [
                {
                    "name": "execute",
                    "description": "Delete round_4 artifacts",
                    "args": {"command": "rm -rf /tmp/round_4"},
                }
            ],
            "review_configs": [],
        }

    fields = build_user_response_fields(_Interrupt(), {"decisions": [{"type": "reject"}]})
    assert fields is not None
    assert fields["kind"] == "approval"
    assert fields["response"] == "reject"
    assert fields["context"]["tool_name"] == "execute"


def test_user_turn_event_includes_prompt_when_begin_run_provides_it(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-user-prompt")
    tracker.begin_run(
        "run-user-prompt",
        user_message="确认无误",
        prompt="请确认 base_info 是否无误。",
    )

    event = list(AuditReader(working_dir=tmp_path, thread_id="thread-user-prompt").iter_events())[0]
    assert event["message"] == "确认无误"
    assert event["prompt"] == "请确认 base_info 是否无误。"


def test_user_turn_precedes_delegation_when_same_run(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-timeline")
    tracker.begin_run("run-timeline", user_message="start tuning")
    _observe_delegation(
        tracker,
        subagent_type="quant-tuning-practice-generator",
        description=MSAGENT_IO_INPUT,
        call_id="call-timeline-1",
        result=MSAGENT_IO_OUTPUT,
    )

    events = list(AuditReader(working_dir=tmp_path, thread_id="thread-timeline").iter_events())
    assert events[0]["event"] == AuditEventType.USER_TURN
    assert events[1]["event"] == AuditEventType.SUBAGENT_DELEGATION
    assert events[0]["run_id"] == events[1]["run_id"]


def test_delegation_event_prefers_full_content_when_short_content_truncated(tmp_path: Path) -> None:
    tracker = _tracker(tmp_path, thread_id="thread-full-raw")
    tracker.begin_run("run-full-raw")

    full_result = "x" * 5000
    tracker.observe(
        _task_call(
            subagent_type="quant-tuning-practice-generator",
            description="plain task",
            call_id="call-full-raw",
        ),
        namespace=(),
    )
    message = _task_result(content=full_result, call_id="call-full-raw")
    setattr(message, "short_content", full_result[:200] + "... (truncated)")
    tracker.observe(message, namespace=())

    event = list(AuditReader(working_dir=tmp_path, thread_id="thread-full-raw").iter_events())[0]
    assert event["result_raw"] == full_result


def test_agent_prompt_extractor_preserves_full_text_when_assistant_message_long() -> None:
    long_prompt = "请确认配置。" + ("详细说明。" * 500)
    prompt = extract_last_agent_prompt([HumanMessage(content="start"), AIMessage(content=long_prompt)])
    assert prompt == long_prompt


def test_user_response_fields_preserve_full_args_when_command_long() -> None:
    long_command = "echo " + ("a" * 2000)

    class _Interrupt:
        id = "int-long"
        value = {
            "action_requests": [
                {
                    "name": "execute",
                    "description": "Review shell command execution before running.",
                    "args": {"command": long_command},
                }
            ],
            "review_configs": [],
        }

    fields = build_user_response_fields(_Interrupt(), {"decisions": [{"type": "approve"}]})
    assert fields is not None
    assert long_command in fields["prompt"]
    assert "..." not in fields["prompt"]
