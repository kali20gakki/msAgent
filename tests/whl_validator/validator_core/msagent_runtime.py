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

"""Reusable isolated runtime for one pytest validation case."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from validator_core.agent_runner import RunResult, run_msagent


logger = logging.getLogger(__name__)


@dataclass
class MsagentRuntime:
    """A workspace and MSAGENT_HOME shared by calls in one test case."""

    provider: str
    model: str
    workspace_dir: Path
    msagent_home: Path
    artifact_dir: Path
    extra_env: dict[str, str]
    executable: str = "msagent"
    timeout_seconds: float | None = None
    execute_approval_mode: str | None = None
    _invocation_index: int = field(default=0, init=False, repr=False)

    def run(self, prompt: str, *, agent_name: str | None = None) -> RunResult:
        self._invocation_index += 1
        invocation_dir = self.artifact_dir / f"invocation-{self._invocation_index:02d}"
        result = run_msagent(
            prompt=prompt,
            workspace_dir=str(self.workspace_dir),
            extra_env=self.extra_env,
            agent_name=agent_name,
            executable=self.executable,
            artifact_dir=invocation_dir,
            timeout_seconds=self.timeout_seconds,
            execute_approval_mode=self.execute_approval_mode,
        )
        logger.info("msagent trace retained at %s", result.trace_path)
        return result
