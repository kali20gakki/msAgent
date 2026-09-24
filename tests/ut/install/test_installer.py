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

"""UT entry points that run the installer logic test suites.

The decision logic of ``scripts/install.sh`` / ``scripts/install.ps1`` (index
priority, version pinning and announcement, PyPI fallback retry,
MSAGENT_NO_MODIFY_PATH, static syntax) is guarded here by invoking the mock-uv
suites in ``tests/install/``. The suites run in seconds and do not perform a
real package install.

The heavy real-install smoke test (``tests/install/smoke_install.sh``) is kept
out of UT on purpose: it downloads all dependencies and is meant for scheduled
CI or manual verification.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_INSTALL = REPO_ROOT / "tests" / "install"

SH_SUITE = TESTS_INSTALL / "test_install_sh.sh"
PS1_SUITE = TESTS_INSTALL / "test_install_ps1.ps1"


def _run_suite(argv: list[str]) -> None:
    env = dict(os.environ)
    result = subprocess.run(
        argv,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"installer test suite failed ({argv[0]}):\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


def test_install_ps1_logic_suite() -> None:
    """install.ps1 logic: version announcement, index passthrough, PyPI
    fallback retry, NO_MODIFY_PATH (mock uv, no real install).

    The suite itself is cross-platform: it runs under PowerShell Core on
    Linux CI runners (pwsh) and under Windows PowerShell on Windows.
    """
    if not PS1_SUITE.exists():
        pytest.skip("tests/install/test_install_ps1.ps1 not found")
    shell = shutil.which("pwsh") or shutil.which("powershell") or shutil.which("powershell.exe")
    if not shell:
        pytest.skip("PowerShell (pwsh/powershell) not available on this runner")
    _run_suite([shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(PS1_SUITE)])
