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

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TEST_TARGETS=(
  "${REPO_ROOT}/tests/benchmark"
  "${REPO_ROOT}/tests/e2e"
  "${REPO_ROOT}/tests/test_benchmark_runner.py"
)

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "${PYTHON_BIN} is required to run system tests." >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install uv

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation failed; cannot run system tests." >&2
  exit 1
fi

cd "${REPO_ROOT}"
echo "[run_st] Running benchmark and end-to-end tests..."
uv run pytest -q "${TEST_TARGETS[@]}" "$@"
