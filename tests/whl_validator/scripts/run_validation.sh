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
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

wheel_path=""
python_version="3.11"
conda_command="${CONDA_EXE:-conda}"
run_dir=""
keep_env=false
skip_tests=false

usage() {
    cat <<'EOF'
Usage:
  ./scripts/run_validation.sh --wheel <path-to-whl> [options]

Required:
  --wheel PATH             mindstudio-agent whl to validate

Options:
  --python-version VERSION Python version for the clean Conda environment
                           (default: 3.11)
  --run-dir PATH           Empty directory for the environment and logs
  --keep-env               Do not remove the Conda environment after run
                           (default: removed)
  --skip-tests             Skip the pytest validation stage
  -h, --help               Show this help

Validation stages:
  1. Create a clean Conda environment.
  2. Install the specified whl and its dependencies.
  3. Run pip check.
  4. Install test dependencies and run pytest (-n auto).
EOF
}

while (($# > 0)); do
    case "$1" in
        --wheel)
            [[ $# -ge 2 ]] || { echo "--wheel requires a value" >&2; exit 2; }
            wheel_path="$2"
            shift 2
            ;;
        --python-version)
            [[ $# -ge 2 ]] || { echo "--python-version requires a value" >&2; exit 2; }
            python_version="$2"
            shift 2
            ;;
        --run-dir)
            [[ $# -ge 2 ]] || { echo "--run-dir requires a value" >&2; exit 2; }
            run_dir="$2"
            shift 2
            ;;
        --keep-env)
            keep_env=true
            shift
            ;;
        --skip-tests)
            skip_tests=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$wheel_path" ]]; then
    echo "--wheel is required" >&2
    usage >&2
    exit 2
fi
if [[ ! -f "$wheel_path" ]]; then
    echo "Wheel does not exist: $wheel_path" >&2
    exit 2
fi
if [[ "$wheel_path" != *.whl ]]; then
    echo "Input is not a .whl file: $wheel_path" >&2
    exit 2
fi
if ! command -v "$conda_command" >/dev/null 2>&1; then
    echo "Conda executable was not found: $conda_command" >&2
    exit 2
fi

wheel_path="$(cd -- "$(dirname -- "$wheel_path")" && pwd)/$(basename -- "$wheel_path")"
if [[ -z "$run_dir" ]]; then
    run_id="install-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    run_dir="${PROJECT_ROOT}/artifacts/${run_id}"
else
    mkdir -p -- "$(dirname -- "$run_dir")"
    run_dir="$(cd -- "$(dirname -- "$run_dir")" && pwd)/$(basename -- "$run_dir")"
fi

if [[ -e "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Run directory must be empty: $run_dir" >&2
    exit 2
fi
mkdir -p -- "$run_dir"

env_dir="${run_dir}/conda-env"
conda_log="${run_dir}/conda-create.log"
install_log="${run_dir}/pip-install.log"
check_log="${run_dir}/pip-check.log"
test_deps_log="${run_dir}/pip-install-test-deps.log"

cleanup() {
    local exit_code=$?
    if [[ "${keep_env:-false}" != true ]] && [[ -n "${env_dir:-}" ]] && [[ -d "${env_dir:-}" ]]; then
        echo "[cleanup] Removing Conda environment: $env_dir" >&2
        rm -rf -- "$env_dir"
    fi
    exit "$exit_code"
}
trap cleanup EXIT

echo "[1/4] Creating clean Conda environment: $env_dir"
if ! "$conda_command" create \
    --yes \
    --prefix "$env_dir" \
    "python=${python_version}" \
    pip \
    2>&1 | tee "$conda_log"; then
    echo "[ERROR] Stage 1/4 failed: Conda environment creation. Log: $conda_log" >&2
    exit 1
fi

env_python="${env_dir}/bin/python"
if [[ ! -x "$env_python" ]]; then
    echo "Conda environment did not create an executable Python: $env_python" >&2
    exit 1
fi

echo "[2/4] Installing wheel: $wheel_path"
if ! "$env_python" -m pip install \
    --disable-pip-version-check \
    --no-input \
    "$wheel_path" \
    2>&1 | tee "$install_log"; then
    echo "[ERROR] Stage 2/4 failed: pip install. Log: $install_log" >&2
    exit 1
fi

echo "[3/4] Checking installed dependency consistency"
if ! "$env_python" -m pip check 2>&1 | tee "$check_log"; then
    echo "[ERROR] Stage 3/4 failed: pip check found dependency conflicts. Log: $check_log" >&2
    exit 1
fi

if [[ "$skip_tests" == true ]]; then
    echo "[4/4] Skipping tests (--skip-tests)"
else
    echo "[4/4] Installing test dependencies"
    if ! "$env_python" -m pip install \
        --disable-pip-version-check \
        --no-input \
        -r "$PROJECT_ROOT/requirements-test.txt" \
        2>&1 | tee "$test_deps_log"; then
        echo "[ERROR] Stage 4/4 failed: test dependency installation. Log: $test_deps_log" >&2
        exit 1
    fi

    echo "      Running pytest (-n auto)"
    if ! PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}" \
        MSAGENT_VALIDATION_RUN_DIR="${run_dir}/test-artifacts" \
        "$env_python" -m pytest \
        -n auto \
        --rootdir="$PROJECT_ROOT" \
        "$PROJECT_ROOT/test_case"; then
        echo "[ERROR] Stage 4/4 failed: pytest." >&2
        exit 1
    fi
fi

cat <<EOF

Validation passed.
Validated wheel:   $wheel_path
Logs:              $run_dir
EOF
if [[ "${keep_env}" == true ]]; then
    echo "Conda environment (retained): $env_dir"
fi
