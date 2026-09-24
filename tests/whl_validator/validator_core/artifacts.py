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

"""Paths and safe names for diagnostic artifacts."""

from __future__ import annotations

import re
from pathlib import Path


def safe_case_name(nodeid: str) -> str:
    """Convert a pytest node id into a stable filesystem directory name."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", nodeid).strip("._")
    return normalized[:180] or "unknown-test"


def create_case_artifact_dir(run_dir: Path, nodeid: str) -> Path:
    """Create the directory holding every msagent invocation for one case."""
    case_dir = run_dir / "cases" / safe_case_name(nodeid)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir
