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

"""通过 Mock LLM 和真实 msagent 工具轨迹验证文件系统和本地命令环境。"""

from __future__ import annotations

from pathlib import Path

import pytest

from validator_core.agent_runner import RunResult
from validator_core.assertions import (
    assert_app_log_has_no_fatal_exception,
    assert_session_succeeded,
    assert_tool_invoked,
)
from validator_core.llm_capture_server import LlmCaptureServer, ToolCallScript
from validator_core.trace_parser import get_tool_calls


FILESYSTEM_MARKER = "MSAGENT_FILESYSTEM_VALIDATION_OK"
LOCALSHELL_MARKER = "MSAGENT_LOCALSHELL_VALIDATION_OK"
DUMMY_API_KEY = "local-env-validation-dummy-key"
MISSING_FILE_ERROR_KEYWORDS = (
    "error",
    "no such file",
    "not found",
    "does not exist",
    "不存在",
)

pytestmark = [pytest.mark.local]


def _create_mock_runtime(
    msagent_runtime_factory,
    mock_server: LlmCaptureServer,
    test_workspace,
):
    """创建使用 Mock LLM 的隔离运行时。"""
    return msagent_runtime_factory(
        "openai",
        base_url=mock_server.base_url,
        api_key=DUMMY_API_KEY,
        workspace_seed=test_workspace,
    )


def _tool_output(event: dict, tool_name: str) -> tuple[dict, str]:
    """提取工具输出结构和文本，并在 trace 格式异常时给出直接诊断。"""
    output = event.get("output")
    assert isinstance(output, dict), f"{tool_name} 返回结构无效: {event!r}"
    content = str(output.get("content") or "")
    assert content.strip(), f"{tool_name} 返回内容为空: {output!r}"
    return output, content


def _assert_run_is_healthy(result: RunResult, mock_server: LlmCaptureServer) -> None:
    """工具完成后，msagent 进程、会话和 DEBUG 日志均应正常。"""
    assert not mock_server.errors, "\n".join(mock_server.errors)
    assert_session_succeeded(result)
    assert_app_log_has_no_fatal_exception(result)


def test_filesystem_read_trace_is_successful(
    msagent_runtime_factory,
    scripted_llm_server_factory,
    test_workspace,
) -> None:
    """read_file 应读取已知文件并返回唯一验证标记。"""
    marker_path = (test_workspace / "read_marker.txt").resolve()
    assert marker_path.exists(), f"测试标记文件不存在: {marker_path}"

    mock_server = scripted_llm_server_factory(
        ToolCallScript(
            tool_name="read_file",
            arguments={"file_path": str(marker_path)},
            final_content=FILESYSTEM_MARKER,
        )
    )
    runtime = _create_mock_runtime(msagent_runtime_factory, mock_server, test_workspace)
    result = runtime.run("读取文件并回复内容。", agent_name="Accuracy")

    _, tool_result = assert_tool_invoked(
        result,
        "read_file",
        input_matches=lambda tool_input: _same_path(tool_input.get("file_path"), marker_path),
    )
    assert not get_tool_calls(result.traces, "execute"), "文件读取用例不应绕过 read_file 使用 execute"

    output, content = _tool_output(tool_result, "read_file")
    assert output.get("is_error") is False, f"read_file 执行失败: {output!r}"
    assert FILESYSTEM_MARKER in content, f"read_file 未返回验证标记: {content!r}"
    assert FILESYSTEM_MARKER in result.stdout, f"最终回复未包含文件内容: {result.stdout!r}"
    _assert_run_is_healthy(result, mock_server)


def test_read_file_reports_missing_file(
    msagent_runtime_factory,
    scripted_llm_server_factory,
    test_workspace,
) -> None:
    """read_file 读取不存在的文件时，应返回明确错误且会话正常收口。"""
    missing_path = (test_workspace / "missing-validation-file.txt").resolve()
    assert not missing_path.exists(), f"缺失文件测试路径意外存在: {missing_path}"

    mock_server = scripted_llm_server_factory(
        ToolCallScript(
            tool_name="read_file",
            arguments={"file_path": str(missing_path)},
            final_content="MOCK_READ_MISSING_COMPLETED",
        )
    )
    runtime = _create_mock_runtime(msagent_runtime_factory, mock_server, test_workspace)
    result = runtime.run("读取文件并回复结果。", agent_name="Accuracy")

    _, tool_result = assert_tool_invoked(
        result,
        "read_file",
        input_matches=lambda tool_input: _same_path(tool_input.get("file_path"), missing_path),
    )
    assert not get_tool_calls(result.traces, "execute"), "缺失文件用例不应调用 execute"

    output, content = _tool_output(tool_result, "read_file")
    normalized = content.casefold()
    has_explicit_error = output.get("is_error") is True or any(
        keyword in normalized for keyword in MISSING_FILE_ERROR_KEYWORDS
    )
    assert has_explicit_error, f"read_file 未明确报告文件不存在: {output!r}"
    assert missing_path.name in content, f"错误结果没有指明目标文件 {missing_path.name!r}: {content!r}"
    assert FILESYSTEM_MARKER not in content, f"缺失文件结果错误包含成功标记: {content!r}"
    assert not missing_path.exists(), "read_file 不应创建缺失的目标文件"
    _assert_run_is_healthy(result, mock_server)


def test_localshell_execution_relative_path(
    msagent_runtime_factory,
    scripted_llm_server_factory,
    test_workspace,
) -> None:
    """execute 应以相对路径运行 workspace 内脚本并返回成功退出码和唯一输出标记。"""
    mock_server = scripted_llm_server_factory(
        ToolCallScript(
            tool_name="execute",
            arguments={"command": "bash print_marker.sh"},
            final_content=LOCALSHELL_MARKER,
        )
    )
    runtime = _create_mock_runtime(msagent_runtime_factory, mock_server, test_workspace)
    result = runtime.run("执行脚本并回复输出。", agent_name="Accuracy")

    _, tool_result = assert_tool_invoked(
        result,
        "execute",
        input_matches=lambda tool_input: "print_marker.sh" in str(tool_input.get("command") or ""),
    )
    output, content = _tool_output(tool_result, "execute")
    assert output.get("is_error") is False, f"execute 执行失败: {output!r}"
    assert LOCALSHELL_MARKER in content, f"脚本输出缺少验证标记: {content!r}"
    assert "command succeeded with exit code 0" in content.casefold(), f"execute 没有报告成功退出码 0: {content!r}"
    assert LOCALSHELL_MARKER in result.stdout, f"最终回复未包含脚本输出: {result.stdout!r}"
    _assert_run_is_healthy(result, mock_server)


def test_localshell_execution_absolute_path(
    msagent_runtime_factory,
    scripted_llm_server_factory,
    test_workspace,
    tmp_path,
) -> None:
    """execute 应以绝对路径运行 workspace 外脚本并返回成功退出码和唯一输出标记。"""
    external_dir = tmp_path / "external_scripts"
    external_dir.mkdir()
    script_path = (external_dir / "print_marker.sh").resolve()
    script_path.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' 'MSAGENT_LOCALSHELL_VALIDATION_OK'\n",
        encoding="utf-8",
    )
    script_path.chmod(script_path.stat().st_mode | 0o111)
    assert script_path.is_file(), f"外部脚本未创建: {script_path}"
    assert not script_path.is_relative_to(test_workspace), f"外部脚本不应位于 workspace 内: {script_path}"

    mock_server = scripted_llm_server_factory(
        ToolCallScript(
            tool_name="execute",
            arguments={"command": f"bash {script_path}"},
            final_content=LOCALSHELL_MARKER,
        )
    )
    runtime = _create_mock_runtime(msagent_runtime_factory, mock_server, test_workspace)
    result = runtime.run("执行脚本并回复输出。", agent_name="Accuracy")

    _, tool_result = assert_tool_invoked(
        result,
        "execute",
        input_matches=lambda tool_input: str(script_path) in str(tool_input.get("command") or ""),
    )
    output, content = _tool_output(tool_result, "execute")
    assert output.get("is_error") is False, f"execute 执行失败: {output!r}"
    assert LOCALSHELL_MARKER in content, f"脚本输出缺少验证标记: {content!r}"
    assert "command succeeded with exit code 0" in content.casefold(), f"execute 没有报告成功退出码 0: {content!r}"
    assert LOCALSHELL_MARKER in result.stdout, f"最终回复未包含脚本输出: {result.stdout!r}"
    _assert_run_is_healthy(result, mock_server)


def test_msprof_analyze_cli_help_is_available(
    msagent_runtime_factory,
    scripted_llm_server_factory,
    test_workspace,
) -> None:
    """whl 环境应提供可由 Agent 调用的 msprof-analyze CLI。"""
    mock_server = scripted_llm_server_factory(
        ToolCallScript(
            tool_name="execute",
            arguments={"command": "msprof-analyze --version"},
            final_content="MOCK_MSPROF_ANALYZE_COMPLETED",
        )
    )
    runtime = _create_mock_runtime(msagent_runtime_factory, mock_server, test_workspace)
    result = runtime.run("执行命令并回复输出。", agent_name="Accuracy")

    _, tool_result = assert_tool_invoked(
        result,
        "execute",
        input_matches=lambda tool_input: (
            "msprof-analyze" in str(tool_input.get("command") or "")
            and "--version" in str(tool_input.get("command") or "")
        ),
    )
    output, content = _tool_output(tool_result, "execute")
    normalized = content.casefold()
    assert output.get("is_error") is False, f"msprof-analyze --version 执行失败: {output!r}"
    assert "version" in normalized, f"输出不是 msprof-analyze 版本信息: {content!r}"
    assert "command succeeded with exit code 0" in normalized, f"msprof-analyze 没有报告成功退出码 0: {content!r}"
    for error_text in ("command not found", "no such file"):
        assert error_text not in normalized, f"msprof-analyze CLI 在目标环境中不可用: {content!r}"
    _assert_run_is_healthy(result, mock_server)


def _same_path(raw_path, expected: Path) -> bool:
    """比较模型工具参数中的路径，并将无效输入视为不匹配。"""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False
    try:
        return Path(raw_path).expanduser().resolve() == expected
    except OSError:
        return False
