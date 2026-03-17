#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${DIST_DIR:-${REPO_ROOT}/dist}"
SKILLS_SUBMODULE_PATH="resources/configs/default/skills"
SKILLS_DIR="${REPO_ROOT}/${SKILLS_SUBMODULE_PATH}"
VERIFY_WHEEL_INSTALL="${VERIFY_WHEEL_INSTALL:-0}"

log() {
  printf '[build_whl] %s\n' "$*"
}

fail() {
  printf '[build_whl] %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi

  if command_exists python3; then
    printf 'python3\n'
    return
  fi

  if command_exists python; then
    printf 'python\n'
    return
  fi

  fail "Python interpreter not found. Set PYTHON_BIN or install python3/python."
}

ensure_supported_python() {
  local python_bin="$1"

  "${python_bin}" - <<'PY'
import sys

if sys.version_info < (3, 11):
    raise SystemExit("mindstudio-agent requires Python 3.11+ to build wheels.")
PY
}

sync_skills_submodule() {
  if [[ ! -f "${REPO_ROOT}/.gitmodules" ]] || ! command_exists git; then
    return
  fi

  if git -C "${REPO_ROOT}" config --file .gitmodules --get-regexp '^submodule\..*\.path$' \
    | awk '{print $2}' | grep -qx "${SKILLS_SUBMODULE_PATH}"; then
    log "Syncing ${SKILLS_SUBMODULE_PATH} submodule..."
    git -C "${REPO_ROOT}" submodule sync --recursive
    git -C "${REPO_ROOT}" submodule update --init --recursive --force --depth 1 "${SKILLS_SUBMODULE_PATH}"
  fi
}

ensure_skills_payload() {
  if [[ ! -d "${SKILLS_DIR}" ]]; then
    fail "Missing ${SKILLS_SUBMODULE_PATH}. Initialize submodules before building."
  fi

  if ! find "${SKILLS_DIR}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    fail "${SKILLS_SUBMODULE_PATH} is empty. Run git submodule update --init --recursive ${SKILLS_SUBMODULE_PATH}."
  fi
}

prepare_dist_dir() {
  mkdir -p "${DIST_DIR}"
  find "${DIST_DIR}" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' \) -delete
}

build_wheel() {
  local python_bin
  python_bin="$(resolve_python)"
  ensure_supported_python "${python_bin}"

  if command_exists uv; then
    log "Checking uv.lock is in sync with pyproject.toml..."
    (
      cd "${REPO_ROOT}"
      uv lock --check
    )

    log "Building wheel with uv..."
    uv build --wheel --out-dir "${DIST_DIR}" "${REPO_ROOT}"
    return
  fi

  log "uv not found, falling back to python -m build..."
  "${python_bin}" -m pip install --upgrade build
  "${python_bin}" -m build --wheel --outdir "${DIST_DIR}" "${REPO_ROOT}"
}

find_built_wheel() {
  find "${DIST_DIR}" -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1
}

resolve_venv_python() {
  local venv_dir="$1"

  if [[ -x "${venv_dir}/bin/python" ]]; then
    printf '%s\n' "${venv_dir}/bin/python"
    return
  fi

  if [[ -x "${venv_dir}/Scripts/python.exe" ]]; then
    printf '%s\n' "${venv_dir}/Scripts/python.exe"
    return
  fi

  fail "Could not find a Python executable inside ${venv_dir}."
}

verify_wheel_install() {
  local wheel_path="$1"
  local python_bin
  local temp_dir
  local venv_dir
  local venv_python

  [[ "${VERIFY_WHEEL_INSTALL}" == "1" ]] || return 0

  python_bin="$(resolve_python)"
  ensure_supported_python "${python_bin}"
  temp_dir="$(mktemp -d)"
  venv_dir="${temp_dir}/venv"

  cleanup() {
    rm -rf "${temp_dir}"
  }
  trap cleanup EXIT

  log "Running isolated wheel smoke test..."
  "${python_bin}" -m venv "${venv_dir}"
  venv_python="$(resolve_venv_python "${venv_dir}")"
  "${venv_python}" -m pip install --upgrade pip
  "${venv_python}" -m pip install "${wheel_path}"
  "${venv_python}" - <<'PY'
import json
from importlib import import_module
from importlib.resources import files

import_module("msagent.cli.bootstrap.app")
config = json.loads(
    files("resources").joinpath("configs", "default", "config.mcp.json").read_text(
        encoding="utf-8"
    )
)
server = config["mcpServers"]["msprof-mcp"]
assert server["command"] == "uvx"
assert server["args"][:2] == ["--isolated", "--from"]
print("wheel smoke test passed")
PY
}

main() {
  log "Preparing wheel build in ${REPO_ROOT}"
  prepare_dist_dir
  sync_skills_submodule
  ensure_skills_payload
  build_wheel

  local wheel_path
  wheel_path="$(find_built_wheel)"
  if [[ -z "${wheel_path}" ]]; then
    fail "Build failed: no wheel generated in ${DIST_DIR}"
  fi

  verify_wheel_install "${wheel_path}"

  log "Wheel generated: ${wheel_path}"
  log "Install with: pip install \"${wheel_path}\""
}

main "$@"
