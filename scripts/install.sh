#!/usr/bin/env bash
# Install msAgent with uv.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kali20gakki/msAgent/main/scripts/install.sh | bash
#
# Environment variables:
#   MSAGENT_PACKAGE_SPEC           Package spec passed to `uv tool install`
#                                  (default: mindstudio-agent)
#   MSAGENT_PYTHON                Python version for the tool environment
#                                  (default: 3.11)
#   MSAGENT_WITH_EXECUTABLES_FROM Additional package whose executables should
#                                  be exposed alongside msagent
#                                  (default: msprof-mcp)
#   UV_BIN                        Path to uv binary (auto-detected if unset)

set -euo pipefail

if [ -t 1 ] || [ "${FORCE_COLOR:-}" = "1" ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  CYAN='\033[0;36m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' CYAN='' BOLD='' NC=''
fi

log_info() { printf "${CYAN}▸${NC} %s\n" "$*"; }
log_success() { printf "${GREEN}✔${NC} %s\n" "$*"; }
log_warn() { printf "${YELLOW}⚠${NC} %s\n" "$*" >&2; }
log_error() { printf "${RED}✖${NC} %s\n" "$*" >&2; }

cleanup() {
  local exit_code=$?
  if [ "${exit_code}" -ne 0 ]; then
    echo "" >&2
    log_error "msAgent installation failed (exit code ${exit_code})."
    log_error "You can retry with:"
    log_error "  pip install -U mindstudio-agent"
  fi
}
trap cleanup EXIT

detect_os() {
  case "$(uname -s)" in
    Darwin) OS="macos" ;;
    Linux) OS="linux" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
    *) OS="unknown" ;;
  esac
}
detect_os

if [ "${OS}" = "windows" ]; then
  log_error "This installer targets Linux, macOS, and WSL shells."
  log_error "On Windows, please use: pip install -U mindstudio-agent"
  exit 1
fi

# macOS MDM/root installs may run without a usable HOME. Re-home to the real
# console user so uv installs into the expected profile directory.
if [ "${OS}" = "macos" ] && { [ -z "${HOME:-}" ] || [ "$(id -u)" -eq 0 ]; }; then
  CONSOLE_USER="$(stat -f '%Su' /dev/console 2>/dev/null || true)"
  if [ -n "${CONSOLE_USER}" ] && [ "${CONSOLE_USER}" != "root" ] && [ -d "/Users/${CONSOLE_USER}" ]; then
    HOME="/Users/${CONSOLE_USER}"
    export HOME
  fi
fi

if [ "$(id -u)" -eq 0 ]; then
  TARGET_USER="${SUDO_USER:-${CONSOLE_USER:-$(basename "${HOME:-root}")}}"
  if [ -z "${TARGET_USER}" ] || [ "${TARGET_USER}" = "root" ]; then
    fix_owner() { :; }
  else
    fix_owner() {
      chown -R "${TARGET_USER}" "$@" 2>/dev/null || \
        log_warn "Could not fix ownership for $*"
    }
  fi
else
  fix_owner() { :; }
fi

PACKAGE_SPEC="${MSAGENT_PACKAGE_SPEC:-mindstudio-agent}"
PYTHON_VERSION="${MSAGENT_PYTHON:-3.11}"
WITH_EXECUTABLES_FROM="${MSAGENT_WITH_EXECUTABLES_FROM:-msprof-mcp}"
TOOL_BIN_DIR="${UV_TOOL_BIN_DIR:-${HOME}/.local/bin}"
TOOL_DATA_DIR="${UV_TOOL_DIR:-${HOME}/.local/share/uv}"

install_uv() {
  if command -v curl >/dev/null 2>&1; then
    log_info "Installing uv..."
    curl -fsSL https://astral.sh/uv/install.sh | sh
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    log_info "Installing uv..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    return
  fi
  log_error "curl or wget is required to install uv."
  exit 1
}

if ! command -v uv >/dev/null 2>&1; then
  install_uv
  fix_owner "${HOME}/.local/bin"
fi

if [ -z "${UV_BIN:-}" ]; then
  UV_BIN="uv"
  if ! command -v "${UV_BIN}" >/dev/null 2>&1; then
    if [ -f "${HOME}/.local/bin/env" ]; then
      # shellcheck source=/dev/null
      . "${HOME}/.local/bin/env"
    fi
  fi
  if ! command -v uv >/dev/null 2>&1; then
    UV_BIN="${HOME}/.local/bin/uv"
  fi
fi

if [ ! -x "${UV_BIN}" ] && ! command -v "${UV_BIN}" >/dev/null 2>&1; then
  log_error "uv is not available after installation."
  log_error "Please restart your shell, or add ~/.local/bin to PATH, then retry."
  exit 1
fi

INSTALL_ARGS=(
  tool
  install
  -U
  --python
  "${PYTHON_VERSION}"
)

if [ -n "${WITH_EXECUTABLES_FROM}" ]; then
  INSTALL_ARGS+=(--with-executables-from "${WITH_EXECUTABLES_FROM}")
fi

INSTALL_ARGS+=("${PACKAGE_SPEC}")

if command -v msagent >/dev/null 2>&1; then
  log_info "Updating existing msAgent installation..."
else
  log_info "Installing msAgent from ${PACKAGE_SPEC}..."
fi

"${UV_BIN}" "${INSTALL_ARGS[@]}"

fix_owner "${TOOL_BIN_DIR}" "${TOOL_DATA_DIR}"
if [ -d "${HOME}/.cache/uv" ]; then
  fix_owner "${HOME}/.cache/uv"
fi
if [ "${OS}" = "macos" ] && [ -d "${HOME}/Library/Caches/uv" ]; then
  fix_owner "${HOME}/Library/Caches/uv"
fi

MSAGENT_BIN=""
if [ -x "${TOOL_BIN_DIR}/msagent" ]; then
  MSAGENT_BIN="${TOOL_BIN_DIR}/msagent"
elif command -v msagent >/dev/null 2>&1; then
  MSAGENT_BIN="msagent"
fi

if [ -z "${MSAGENT_BIN}" ]; then
  log_warn "msagent was installed but is not on PATH in this shell."
  log_warn "Try: source ~/.zshrc  (or ~/.bashrc)"
  exit 0
fi

VERSION_OUTPUT="$("${MSAGENT_BIN}" --version 2>&1)" || {
  log_error "msagent installed, but '--version' failed:"
  printf '%s\n' "${VERSION_OUTPUT}" >&2
  exit 1
}
log_success "Verified: ${VERSION_OUTPUT}"

MSPROF_BIN=""
if [ -x "${TOOL_BIN_DIR}/msprof-mcp" ]; then
  MSPROF_BIN="${TOOL_BIN_DIR}/msprof-mcp"
elif command -v msprof-mcp >/dev/null 2>&1; then
  MSPROF_BIN="msprof-mcp"
fi

if [ -n "${WITH_EXECUTABLES_FROM}" ] && [ -z "${MSPROF_BIN}" ]; then
  log_warn "msprof-mcp is not on PATH in this shell yet."
  log_warn "If msagent later complains about msprof-mcp, restart your shell first."
else
  log_success "msprof-mcp executable is available."
fi

log_success "msAgent installation complete."
