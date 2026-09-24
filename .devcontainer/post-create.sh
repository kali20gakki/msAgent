#!/bin/bash
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
set -uo pipefail

log() { echo "[post-create] $*"; }
warn() { echo "[post-create] WARN: $*"; }

configure_user_bin() {
    log "Configuring user bin directory..."
    mkdir -p "$HOME/.local/bin"
    npm config set prefix "$HOME/.local" 2>/dev/null || warn "npm config set prefix failed"
    local marker="# msagent-devcontainer-user-bin"
    for rcfile in "$HOME/.bashrc" "$HOME/.bash_profile"; do
        if [ -f "$rcfile" ] && ! grep -qF "$marker" "$rcfile"; then
            cat >> "$rcfile" <<EOF
$marker
export PATH="\$HOME/.local/bin:\$PATH"
EOF
            log "Appended PATH to $rcfile"
        fi
    done
}

configure_python3() {
    log "Configuring Python 3..."
    if [ -f "/etc/profile.d/pyenv.sh" ]; then
        source /etc/profile.d/pyenv.sh 2>/dev/null || warn "Failed to source pyenv.sh"
        log "Loaded pyenv profile"
    fi
    for candidate in /opt/python/cp311-cp* /opt/python/cp310-cp*; do
        if [ -d "$candidate/bin" ] && [ -x "$candidate/bin/python3" ]; then
            local marker="# msagent-pyenv-python"
            for rcfile in "$HOME/.bashrc" "$HOME/.bash_profile"; do
                if [ -f "$rcfile" ] && ! grep -qF "$marker" "$rcfile"; then
                    cat >> "$rcfile" <<EOF
$marker
export PATH="$candidate/bin:\$PATH"
EOF
                    log "Prepended pyenv Python to PATH in $rcfile"
                fi
            done
            export PATH="$candidate/bin:$PATH"
            break
        fi
    done
    if command -v python3 &>/dev/null; then
        log "Python 3 found: $(python3 --version 2>&1)"
    else
        warn "python3 not found in PATH"
    fi
}

install_build_deps() {
    log "Installing build dependencies..."

    if ! command -v gitleaks &>/dev/null; then
        log "  Installing gitleaks..."
        GITLEAKS_ARCH=""
        GITLEAKS_VER="8.18.4"
        case "$(uname -m)" in
            x86_64)  GITLEAKS_ARCH="amd64";;
            aarch64) GITLEAKS_ARCH="arm64";;
            *)       warn "Unsupported architecture, skipping gitleaks";;
        esac
        if [ -n "$GITLEAKS_ARCH" ]; then
            GITLEAKS_INSTALLED=false
            for MIRROR in \
                "https://ghproxy.com/https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VER}/gitleaks_${GITLEAKS_VER}_linux_${GITLEAKS_ARCH}.tar.gz" \
                "https://mirror.ghproxy.com/https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VER}/gitleaks_${GITLEAKS_VER}_linux_${GITLEAKS_ARCH}.tar.gz" \
                "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_VER}/gitleaks_${GITLEAKS_VER}_linux_${GITLEAKS_ARCH}.tar.gz"; do
                log "    Trying: ${MIRROR}"
                curl -fsSL "${MIRROR}" -o /tmp/gitleaks.tar.gz --connect-timeout 10 2>/dev/null && \
                    sudo tar -xzf /tmp/gitleaks.tar.gz -C /usr/local/bin gitleaks 2>/dev/null && \
                    GITLEAKS_INSTALLED=true && break
                rm -f /tmp/gitleaks.tar.gz
            done
            if ${GITLEAKS_INSTALLED}; then
                sudo chmod +x /usr/local/bin/gitleaks
                log "  gitleaks ${GITLEAKS_VER} (${GITLEAKS_ARCH}) installed"
            else
                log "  Download mirrors unreachable, trying pip..."
                python3 -m pip install gitleaks 2>/dev/null && log "  gitleaks installed via pip" || \
                    warn "gitleaks install failed (pre-commit gitleaks will be skipped)"
            fi
        fi
    else
        log "  System pkg OK: gitleaks"
    fi

    local pip_pkgs=(wheel pre-commit "bandit[toml]")
    for PY in $(command -v python3 2>/dev/null) $(command -v python 2>/dev/null); do
        log "Installing pip packages for: $($PY --version 2>&1)"
        "$PY" -m pip install --quiet --upgrade pip setuptools >/dev/null 2>&1 || warn "pip/setuptools upgrade failed for $PY"
        for pkg in "${pip_pkgs[@]}"; do
            local pkg_name="${pkg%%[*}"
            if ! "$PY" -m pip show "$pkg_name" &>/dev/null; then
                log "  Installing for $PY: $pkg"
                "$PY" -m pip install "$pkg" || warn "pip install failed for $PY: $pkg"
            else
                log "  Pip pkg OK ($PY): $pkg"
            fi
        done
    done
    log "Build dependencies check complete"

    # 使用 uv 安装项目开发依赖（对齐项目规定的构建方式）
    if command -v uv &>/dev/null; then
        log "Installing project dev dependencies via uv..."
        uv sync --dev 2>&1 | tail -3 || warn "uv sync --dev failed"
    else
        log "uv not found, falling back to pip..."
        python3 -m pip install -e ".[dev]" 2>&1 | tail -3 || warn "pip install -e .[dev] failed"
    fi
}

sync_git_identity() {
    log "Syncing Git identity..."
    local gitconfig="$HOME/.devcontainer-host-gitconfig"
    if [ -f "$gitconfig" ] && [ -s "$gitconfig" ]; then
        local name email
        name=$(git config --file "$gitconfig" --get user.name 2>/dev/null) || true
        email=$(git config --file "$gitconfig" --get user.email 2>/dev/null) || true
        [ -n "$name" ] && git config --global user.name "$name"
        [ -n "$email" ] && git config --global user.email "$email"
        log "Git identity synced from host"
    else
        warn "No host Git config found, skipping identity sync"
    fi
}

append_dev_hint_once() {
    local marker="# msagent-dev-hint"
    local rcfile="$HOME/.bashrc"
    if grep -qF "$marker" "$rcfile" 2>/dev/null; then return; fi
    cat >> "$rcfile" <<EOF
$marker
# msagent development commands:
#   python3 run.py              Run msagent
#   python3 -m pytest tests/    Run unit tests
EOF
    log "Appended dev hints to $rcfile"
}

install_pre_commit_hook() {
    log "Installing pre-commit hook..."
    export PATH="$HOME/.local/bin:$PATH"
    local pre_commit_cmd=""
    if command -v pre-commit &>/dev/null; then
        pre_commit_cmd="pre-commit"
    elif python3 -m pre_commit --version &>/dev/null 2>&1; then
        pre_commit_cmd="python3 -m pre_commit"
    elif python -m pre_commit --version &>/dev/null 2>&1; then
        pre_commit_cmd="python -m pre_commit"
    else
        warn "pre-commit not found, skipping hook installation"
        return
    fi
    if ! git rev-parse --git-dir &>/dev/null; then
        warn "Not a Git repository, skipping pre-commit hook"
        return
    fi
    $pre_commit_cmd install || warn "pre-commit install failed"
    log "pre-commit hook installed via: $pre_commit_cmd"
}

ignore_local_changes() {
    log "Setting up skip-worktree for local-modifiable files..."
    if [ -f ".vscode/settings.json" ]; then
        git update-index --skip-worktree .vscode/settings.json 2>/dev/null || true
        log "  skip-worktree: .vscode/settings.json"
    fi
}

log "Starting container initialization (msagent)..."
configure_user_bin
configure_python3
install_build_deps
sync_git_identity
append_dev_hint_once
install_pre_commit_hook
ignore_local_changes
log "Container initialization complete!"
