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
"""Validate Practice YAML (replaces MCP yaml_validation_validate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: add shared library to sys.path before any cross-skill imports
# ---------------------------------------------------------------------------
_common_dir = Path(__file__).resolve().parents[2] / "msmodelslim-tools-common" / "scripts"
if str(_common_dir) not in sys.path:
    sys.path.insert(0, str(_common_dir))

from script_utils import emit_result, ensure_msmodelslim
from yaml_validation_helpers import validate_practice_yaml  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Practice YAML against PracticeConfig.")
    parser.add_argument("--practice-path", required=True)
    args = parser.parse_args()

    ensure_msmodelslim()
    return emit_result(validate_practice_yaml(args.practice_path))


if __name__ == "__main__":
    sys.exit(main())
