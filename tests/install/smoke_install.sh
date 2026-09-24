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
# Real-install smoke test for scripts/install.sh (opt-in / CI).
#
# Installs mindstudio-agent in a disposable HOME (real network, real packages)
# and verifies the CLI runs and the tool environment contains the msprof-mcp
# executable. This is the slowest guard (downloads all dependencies) and is
# intended for scheduled CI or manual verification, not every commit.
#
# Usage:
#   bash tests/install/smoke_install.sh
# Exit code 0 = smoke passed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTALLER="${REPO_ROOT}/scripts/install.sh"

TEST_HOME="$(mktemp -d)"
trap 'rm -rf "${TEST_HOME}"' EXIT

echo "[smoke] disposable HOME: ${TEST_HOME}"
export HOME="${TEST_HOME}"
export MSAGENT_YES=1
export MSAGENT_NO_MODIFY_PATH=1

echo "[smoke] running installer..."
bash "${INSTALLER}" | tee "${TEST_HOME}/install.log"

MSAGENT_BIN="${TEST_HOME}/.local/bin/msagent"
if [ ! -x "${MSAGENT_BIN}" ]; then
  echo "[smoke] FAIL: ${MSAGENT_BIN} not found" >&2
  exit 1
fi
VERSION_OUTPUT="$("${MSAGENT_BIN}" --version 2>&1)"
echo "${VERSION_OUTPUT}" | grep -qi "msagent" || {
  echo "[smoke] FAIL: msagent --version output unexpected" >&2
  exit 1
}
echo "[smoke] msagent --version OK"

TOOL_BIN="${TEST_HOME}/.local/share/uv/tools/mindstudio-agent/bin"
if [ -d "${TOOL_BIN}" ] && ls "${TOOL_BIN}" | grep -q "msprof-mcp"; then
  echo "[smoke] msprof-mcp present in the tool environment"
else
  echo "[smoke] FAIL: msprof-mcp missing from the tool environment" >&2
  exit 1
fi

echo "[smoke] PASS"
