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

"""Shared helpers for prototype CLI scripts."""

from __future__ import annotations

import json
import subprocess  # nosec B404
from pathlib import Path
from typing import Any


def read_or_run(file_path: str | None, command: list[str], optional: bool = False) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=5)  # nosec B603
        return output.decode("utf-8", errors="ignore")
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        if optional:
            return ""
        raise


def write_json_output(output_path: str | Path, data: dict[str, Any]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
