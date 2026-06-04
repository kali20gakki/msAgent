#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${DIST_DIR:-${REPO_ROOT}/dist}"
PYPROJECT_PATH="${PYPROJECT_PATH:-${REPO_ROOT}/pyproject.toml}"
DEFAULT_SKILLS_PATH="skills"
SKILLS_PATH="${SKILLS_PATH:-${SKILLS_SUBMODULE_PATH:-}}"
SKILLS_DIR=""
REQUIRE_SKILLS_PAYLOAD="${REQUIRE_SKILLS_PAYLOAD:-auto}"
VERIFY_WHEEL_INSTALL="${VERIFY_WHEEL_INSTALL:-0}"
SMOKE_IMPORT_MODULE="${SMOKE_IMPORT_MODULE:-}"
SMOKE_RESOURCE_PATH="${SMOKE_RESOURCE_PATH:-resources/configs/default/config.mcp.json}"
SMOKE_SKILL_PATH="${SMOKE_SKILL_PATH:-resources/configs/default/skills/README.md}"
MSAGENT_BUNDLE_WEB_UI="${MSAGENT_BUNDLE_WEB_UI:-0}"
PROJECT_NAME=""
REQUIRES_PYTHON=""
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11

sync_project_version() {
  local python_bin="$1"
  local sync_script="${REPO_ROOT}/scripts/sync_version.py"

  log "Syncing package version from WHL_VERSION env or version.info..."
  "${python_bin}" "${sync_script}"
}

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

is_truthy_flag() {
  local value
  value="${1:-}"
  value="${value,,}"
  case "${value}" in
    ""|0|false|no|off) return 1 ;;
    *) return 0 ;;
  esac
}

load_project_metadata() {
  local python_bin="$1"
  local metadata

  [[ -f "${PYPROJECT_PATH}" ]] || fail "Missing ${PYPROJECT_PATH}. Cannot determine build metadata."

  metadata="$("${python_bin}" - "${PYPROJECT_PATH}" "${REPO_ROOT}" <<'PY'
import pathlib
import re
import sys

pyproject_path = pathlib.Path(sys.argv[1])
repo_root = pathlib.Path(sys.argv[2])
text = pyproject_path.read_text(encoding="utf-8")

name = ""
requires_python = ""
entry_module = ""

try:
    import tomllib  # py311+

    data = tomllib.loads(text)
    project = data.get("project", {})
    if isinstance(project, dict):
        raw_name = project.get("name")
        if isinstance(raw_name, str):
            name = raw_name.strip()
        raw_requires = project.get("requires-python")
        if isinstance(raw_requires, str):
            requires_python = raw_requires.strip()
        scripts = project.get("scripts")
        if isinstance(scripts, dict) and scripts:
            raw_entry = next(iter(scripts.values()))
            if isinstance(raw_entry, str):
                entry_module = raw_entry.split(":", 1)[0].strip()
except Exception:
    pass

if not name:
    match = re.search(r'(?m)^name\s*=\s*"([^"]+)"', text)
    if match:
        name = match.group(1).strip()

if not requires_python:
    match = re.search(r'(?m)^requires-python\s*=\s*"([^"]+)"', text)
    if match:
        requires_python = match.group(1).strip()

if not entry_module:
    block = re.search(r'(?ms)^\[project\.scripts\]\s*(.+?)(?:^\[|\Z)', text)
    if block:
        entry = re.search(r'(?m)^[^#\n=]+\s*=\s*"([^":]+)', block.group(1))
        if entry:
            entry_module = entry.group(1).strip()

if not name:
    name = repo_root.name

major = 3
minor = 11
match = re.search(r'>=\s*([0-9]+)(?:\.([0-9]+))?', requires_python or "")
if match:
    major = int(match.group(1))
    minor = int(match.group(2) or 0)

print(f"PROJECT_NAME={name}")
print(f"REQUIRES_PYTHON={requires_python}")
print(f"MIN_PYTHON_MAJOR={major}")
print(f"MIN_PYTHON_MINOR={minor}")
print(f"SMOKE_IMPORT_MODULE={entry_module}")
PY
)" || fail "Failed to parse project metadata from ${PYPROJECT_PATH}"

  while IFS='=' read -r key value; do
    value="${value%$'\r'}"
    case "${key}" in
      PROJECT_NAME) PROJECT_NAME="${value}" ;;
      REQUIRES_PYTHON) REQUIRES_PYTHON="${value}" ;;
      MIN_PYTHON_MAJOR) MIN_PYTHON_MAJOR="${value}" ;;
      MIN_PYTHON_MINOR) MIN_PYTHON_MINOR="${value}" ;;
      SMOKE_IMPORT_MODULE)
        if [[ -z "${SMOKE_IMPORT_MODULE}" ]]; then
          SMOKE_IMPORT_MODULE="${value}"
        fi
        ;;
    esac
  done <<< "${metadata}"
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

  "${python_bin}" - "${MIN_PYTHON_MAJOR}" "${MIN_PYTHON_MINOR}" "${PROJECT_NAME}" <<'PY'
import sys

required_major = int(sys.argv[1])
required_minor = int(sys.argv[2])
project_name = sys.argv[3] or "this project"

if sys.version_info < (required_major, required_minor):
    raise SystemExit(
        f"{project_name} requires Python {required_major}.{required_minor}+ to build wheels."
    )
PY
}

detect_skills_path() {
  if [[ -n "${SKILLS_PATH}" ]]; then
    SKILLS_DIR="${REPO_ROOT}/${SKILLS_PATH}"
    return
  fi

  if [[ -d "${REPO_ROOT}/${DEFAULT_SKILLS_PATH}" ]]; then
    SKILLS_PATH="${DEFAULT_SKILLS_PATH}"
    SKILLS_DIR="${REPO_ROOT}/${SKILLS_PATH}"
  fi
}

ensure_skills_payload() {
  local required
  case "${REQUIRE_SKILLS_PAYLOAD}" in
    1|true|required) required=1 ;;
    0|false|optional|off) required=0 ;;
    auto|"")
      if [[ -n "${SKILLS_PATH}" ]] || [[ -d "${REPO_ROOT}/${DEFAULT_SKILLS_PATH}" ]]; then
        required=1
      else
        required=0
      fi
      ;;
    *)
      fail "Invalid REQUIRE_SKILLS_PAYLOAD='${REQUIRE_SKILLS_PAYLOAD}'. Use auto|required|optional|off."
      ;;
  esac

  if [[ -z "${SKILLS_PATH}" ]]; then
    if [[ "${required}" == "1" ]]; then
      fail "Skills payload is required but no skills directory was detected. Set SKILLS_PATH or create ${DEFAULT_SKILLS_PATH}/."
    fi
    log "No skills payload configured; skipping skills payload checks."
    return
  fi

  if [[ ! -d "${SKILLS_DIR}" ]]; then
    if [[ "${required}" == "1" ]]; then
      fail "Missing ${SKILLS_PATH}. Ensure the merged skills directory exists before building."
    fi
    log "Optional skills payload ${SKILLS_PATH} was not found; skipping."
    return
  fi

  if ! find "${SKILLS_DIR}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
    if [[ "${required}" == "1" ]]; then
      fail "${SKILLS_PATH} is empty. Add bundled skills before building."
    fi
    log "Optional skills payload ${SKILLS_PATH} is empty; skipping."
  fi
}

prepare_dist_dir() {
  mkdir -p "${DIST_DIR}"
  find "${DIST_DIR}" -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' \) -delete
}

build_wheel() {
  local python_bin="$1"

  if command_exists uv; then
    if [[ -f "${REPO_ROOT}/uv.lock" ]]; then
      log "Checking uv.lock is in sync with pyproject.toml..."
      if ! (
        cd "${REPO_ROOT}"
        uv lock --check
      ); then
        fail "uv.lock is out of sync with pyproject.toml. If you changed project.version, dependencies, or dependency-groups, run 'uv lock', commit the updated uv.lock, and rerun the build."
      fi
    else
      log "uv.lock not found; skipping uv lock consistency check."
    fi

    log "Building wheel with uv..."
    (
      cd "${REPO_ROOT}"
      uv build --wheel --out-dir "${DIST_DIR}"
    )
    return
  fi

  log "uv not found, falling back to python -m build..."
  "${python_bin}" -m pip install --upgrade build
  "${python_bin}" -m build --wheel --outdir "${DIST_DIR}" "${REPO_ROOT}"
}

configure_build_flags() {
  export MSAGENT_BUNDLE_WEB_UI

  if is_truthy_flag "${MSAGENT_BUNDLE_WEB_UI}"; then
    log "Bundling web UI payload: enabled (MSAGENT_BUNDLE_WEB_UI=${MSAGENT_BUNDLE_WEB_UI})"
  else
    log "Bundling web UI payload: disabled (set MSAGENT_BUNDLE_WEB_UI=1 to include it)"
  fi
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

verify_resource_in_installed_wheel() {
  local python_bin="$1"
  local work_dir="$2"
  local project_name="$3"
  local import_module_name="$4"
  local resource_path="$5"

  (
    cd "${work_dir}"
    "${python_bin}" - "${project_name}" "${import_module_name}" "${resource_path}" <<'PY'
import importlib
import importlib.metadata
import pathlib
import sys
from importlib.resources import files

project_name, import_module_name, resource_path = sys.argv[1:4]

if project_name:
    importlib.metadata.version(project_name)

if import_module_name:
    importlib.import_module(import_module_name)

if resource_path:
    parts = [segment for segment in pathlib.PurePosixPath(resource_path.replace("\\", "/")).parts if segment not in (".", "")]
    if parts:
        resource = files(parts[0]).joinpath(*parts[1:])
        if not resource.is_file():
            raise SystemExit(f"resource check failed: {resource_path} not found in installed wheel")
        resource.read_bytes()

print("wheel smoke test passed")
PY
  )
}

verify_wheel_install() {
  local python_bin="$1"
  local wheel_path="$2"
  local temp_dir
  local venv_dir

  [[ "${VERIFY_WHEEL_INSTALL}" == "1" ]] || return 0

  temp_dir="$(mktemp -d)"
  venv_dir="${temp_dir}/venv"

  log "Running isolated wheel smoke test..."
  "${python_bin}" -m venv "${venv_dir}"
  VENV_PYTHON="$(resolve_venv_python "${venv_dir}")"
  export VENV_PYTHON
  "${VENV_PYTHON}" -m pip install --upgrade pip
  "${VENV_PYTHON}" -m pip install "${wheel_path}"
  verify_resource_in_installed_wheel "${VENV_PYTHON}" "${temp_dir}" "${PROJECT_NAME}" "${SMOKE_IMPORT_MODULE}" "${SMOKE_RESOURCE_PATH}"
  verify_resource_in_installed_wheel "${VENV_PYTHON}" "${temp_dir}" "${PROJECT_NAME}" "${SMOKE_IMPORT_MODULE}" "${SMOKE_SKILL_PATH}"
  rm -rf "${temp_dir}"
}

main() {
  local python_bin
  python_bin="$(resolve_python)"
  sync_project_version "${python_bin}"
  load_project_metadata "${python_bin}"
  ensure_supported_python "${python_bin}"
  detect_skills_path
  configure_build_flags

  if [[ -n "${SMOKE_RESOURCE_PATH}" ]] && [[ ! -f "${REPO_ROOT}/${SMOKE_RESOURCE_PATH}" ]]; then
    log "Smoke resource path ${SMOKE_RESOURCE_PATH} does not exist in repository; skipping resource check."
    SMOKE_RESOURCE_PATH=""
  fi
  log "Preparing wheel build in ${REPO_ROOT}"
  log "Project: ${PROJECT_NAME} (requires-python: ${REQUIRES_PYTHON:-unknown})"
  prepare_dist_dir
  ensure_skills_payload
  build_wheel "${python_bin}"

  local wheel_path
  wheel_path="$(find_built_wheel)"
  if [[ -z "${wheel_path}" ]]; then
    fail "Build failed: no wheel generated in ${DIST_DIR}"
  fi

  verify_wheel_install "${python_bin}" "${wheel_path}"

  log "Wheel generated: ${wheel_path}"
  log "Install with: pip install \"${wheel_path}\""
}

main "$@"
