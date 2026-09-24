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

"""Offline parser for MindSpore Profiler data (_ascend_ms)."""
import argparse
import os
import shutil
import sys


def _check_msprof() -> None:
    """Ensure msprof is in PATH; required by CANN/profiler analyse."""
    if shutil.which("msprof") is None:
        print(
            "Error: msprof command not found. Please source the correct CANN environment, e.g.:\n"
            "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline parser for MindSpore Profiler data (_ascend_ms)."
    )
    parser.add_argument(
        "profiler_path",
        type=str,
        help="Path to the profiler result directory (e.g., ./result_data)",
    )
    args = parser.parse_args()

    profiler_path = args.profiler_path
    if not os.path.exists(profiler_path):
        print(f"Error: Profiler path '{profiler_path}' does not exist.")
        sys.exit(1)

    _check_msprof()

    try:
        from mindspore.profiler.profiler import analyse
    except ImportError as e:
        print(f"Error: MindSpore not installed or analyse unavailable: {e}")
        sys.exit(1)

    print(f"Starting offline analysis for: {profiler_path}")
    try:
        analyse(profiler_path=profiler_path)
        print("Analysis completed successfully.")
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
