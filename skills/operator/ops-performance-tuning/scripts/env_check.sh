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
# Ascend operator tuning environment check.
# Usage:
#   bash scripts/env_check.sh [--mode board|sim] [--cann-path PATH]
# Environment:
#   CANN_PATH may point to a versioned CANN or ascend-toolkit directory.

set -u

MODE="board"
CANN_ROOT="${CANN_PATH:-${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}}"
FAIL=0

pass() { echo "[PASS] $1"; }
warn() { echo "[WARN] $1"; }
fail() { echo "[FAIL] $1"; FAIL=1; }
hint() { echo "       $1"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mode)
            [ "$#" -ge 2 ] || { echo "ERROR: --mode requires board or sim" >&2; exit 2; }
            MODE="$2"
            shift 2
            ;;
        --cann-path)
            [ "$#" -ge 2 ] || { echo "ERROR: --cann-path requires a path" >&2; exit 2; }
            CANN_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,8p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

case "$MODE" in
    board|sim) ;;
    *) echo "ERROR: --mode must be board or sim" >&2; exit 2 ;;
esac

SET_ENV=""
for candidate in \
    "$CANN_ROOT/set_env.sh" \
    "$CANN_ROOT/ascend-toolkit/set_env.sh" \
    "$CANN_ROOT/ascend-toolkit/latest/set_env.sh"; do
    if [ -f "$candidate" ]; then
        SET_ENV="$candidate"
        break
    fi
done

if [ -n "$SET_ENV" ]; then
    if source "$SET_ENV" >/dev/null 2>&1; then
        pass "CANN environment: $SET_ENV"
        CANN_ROOT="${ASCEND_HOME_PATH:-$CANN_ROOT}"
    else
        fail "CANN environment cannot be sourced: $SET_ENV"
    fi
else
    fail "CANN set_env.sh not found under: $CANN_ROOT"
    hint "Pass the exact installation with --cann-path or CANN_PATH."
fi

if command -v bisheng >/dev/null 2>&1; then
    pass "BiSheng compiler: $(command -v bisheng)"
else
    warn "BiSheng is not in PATH; Ascend C/CATLASS compilation will be unavailable."
fi

if command -v msprof >/dev/null 2>&1 && msprof op --help >/dev/null 2>&1; then
    pass "msOpProf entry: msprof op"
elif command -v msopprof >/dev/null 2>&1; then
    pass "msOpProf entry: $(command -v msopprof)"
elif command -v msprof >/dev/null 2>&1; then
    warn "msprof exists but 'msprof op' is unavailable; only full msprof fallback is visible."
else
    fail "Neither msprof nor msopprof is available."
fi

SOC_TEXT=""
if command -v npu-smi >/dev/null 2>&1; then
    SOC_TEXT=$(npu-smi info 2>/dev/null | awk -F'|' '/Ascend/ {print $3; exit}' | tr -d ' ')
    [ -n "$SOC_TEXT" ] && pass "NPU detected by npu-smi: $SOC_TEXT"
fi

if [ -z "$SOC_TEXT" ] && command -v python3 >/dev/null 2>&1; then
    SOC_TEXT=$(python3 - <<'PY' 2>/dev/null
try:
    import acl
    print(acl.get_soc_name())
except Exception:
    try:
        import torch_npu
        import torch
        if torch.npu.is_available() and torch.npu.device_count() > 0:
            print(torch.npu.get_device_name(0))
    except Exception:
        pass
PY
)
    [ -n "$SOC_TEXT" ] && pass "NPU detected by runtime: $SOC_TEXT"
fi

if [ -n "$SOC_TEXT" ]; then
    case "$SOC_TEXT" in
        *950*)
            pass "SoC family: A5; common targets are --soc=ascend950 / dav-3510"
            ;;
        *910_93*|*910_9391*|*A3*)
            pass "SoC family: A3; use --soc=ascend910_93 and obtain NPU_ARCH from this toolchain"
            ;;
        *910B*|*A2*)
            pass "SoC family: A2; common targets are --soc=ascend910b / dav-2201"
            ;;
        *)
            warn "Unknown SoC mapping: $SOC_TEXT; inspect project and compiler help before building."
            ;;
    esac
elif [ "$MODE" = "board" ]; then
    fail "No board device was detected by npu-smi, ACL, or torch_npu."
else
    warn "No board device detected; simulator mode may continue."
fi

SIM_LIB=""
if [ -d "$CANN_ROOT" ]; then
    SIM_LIB=$(find "$CANN_ROOT" -path '*/simulator/dav_*/lib/libruntime_camodel.so' -type f -print -quit 2>/dev/null)
fi
if [ -n "$SIM_LIB" ]; then
    pass "Simulator runtime: $SIM_LIB"
elif [ "$MODE" = "sim" ]; then
    fail "Simulator runtime libruntime_camodel.so was not found under $CANN_ROOT."
else
    warn "Simulator runtime not found; this is non-blocking for board mode."
fi

if command -v python3 >/dev/null 2>&1; then
    pass "Python: $(command -v python3)"
    if python3 -m pip --version >/dev/null 2>&1; then
        pass "pip: $(python3 -m pip --version 2>/dev/null | awk '{print $2}')"
    else
        warn "pip is unavailable; --pkg/ES wheel flows may fail."
    fi
    python3 -c 'import setuptools' >/dev/null 2>&1 || \
        warn "setuptools is unavailable; package build flows may fail."
    python3 -c 'import packaging' >/dev/null 2>&1 || \
        warn "Python package 'packaging' is unavailable; some ops-transformer builds require it."
else
    fail "python3 is unavailable."
fi

echo
if [ "$FAIL" -eq 0 ]; then
    echo "[env_check] READY ($MODE mode). Warnings must be resolved only if the selected path needs them."
    exit 0
fi

echo "[env_check] BLOCKED ($MODE mode). Resolve FAIL items before running this path."
exit 1
