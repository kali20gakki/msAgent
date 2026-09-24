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
# =============================================================================
# msAgent installer for Linux / macOS / WSL
#
# Usage:
#   curl -LsSf https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.sh | bash
#
# What it does:
#   1. Preflight checks (network, existing pip install, downloader availability)
#   2. Bootstraps uv (official installer first, pip + domestic mirror as fallback)
#   3. Installs mindstudio-agent (latest from PyPI) into an isolated uv tool env:
#        uv tool install -U --python 3.11 mindstudio-agent --with-executables-from msprof-mcp
#   4. Adds the tool bin directory to PATH (shell profiles, idempotent)
#   5. Verifies: msagent --version
#
# The installer never pins a version: it always installs the latest
# mindstudio-agent from PyPI, so releases do not require touching this script.
# The uv tool environment is fully isolated, so it will not conflict with any
# existing Python environment (system Python, torch, mindstudio_monitor, ...).
#
# Environment variables:
#   MSAGENT_PYTHON            Python request for the isolated tool env (default: >=3.11, reuses local Python)
#   MSAGENT_VERSION           Exact version to install, e.g. "26.1.0" (default: latest)
#   MSAGENT_INDEX             PyPI index URL override (default: official PyPI, domestic mirrors as fallback)
#   MSAGENT_NO_MODIFY_PATH    set to 1 to skip all PATH modification
#   MSAGENT_PLAIN_UI          set to 1 for ASCII-only output (colours/glyphs off; FORCE_COLOR=1 forces them on)
#   MSAGENT_YES               set to 1 to accept prompts without asking (CI/cron)
#   MSAGENT_NO_FALLBACK       set to 1 to disable the venv fallback
#   MSAGENT_FALLBACK_VENV     venv path used by the fallback install (default: ~/.msagent-venv)
#   MSAGENT_WITH_EXECUTABLES_FROM  package whose executables are exposed too (default: msprof-mcp)
#   MSAGENT_NO_ASCEND_DOC_MCP      set to 1 to skip Node provisioning and the ascend-doc-mcp pre-install
#   MSAGENT_ASCEND_DOC_MCP_REQUIRE set to 1 to abort install when Node/ascend-doc-mcp prep fails
#   MSAGENT_NODE_HOME              user-local Node root used by ascend-doc-mcp (default: ~/.msagent/node)
#   MSAGENT_NODE_MIRROR            Node dist mirror base; must expose latest-v22.x/
#                                  (default: probed huaweicloud -> npmmirror -> nodejs.org)
#   MSAGENT_NPM_REGISTRY           npm registry for the ascend-doc-mcp pre-install
#                                  (default: the registry this machine's npm is
#                                  configured with, then probed npmmirror ->
#                                  huaweicloud -> npmjs)
#   MSAGENT_NPM_REGISTRY_ONLY      set to 1 to use MSAGENT_NPM_REGISTRY exclusively
#   MSAGENT_NPM_REGISTRY_FALLBACKS extra last-resort npm sources (comma/semicolon/
#                                  space separated) for corporate intranets
#   MSAGENT_NPM_MIRRORS_FILE       file with one fallback source per line
#                                  (defaults: ~/.msagent/npm-mirrors, then
#                                  /etc/msagent/npm-mirrors when readable)
#   MSAGENT_NPM_CACHE              npm cache directory (default: ~/.cache/msagent/npm-cache)
#   MSAGENT_ASCEND_DOC_MCP_PREFIX  local ascend-doc-mcp install prefix (default: ~/.msagent/ascend-doc-mcp)
#   UV_DEFAULT_INDEX / UV_INDEX_URL / PIP_INDEX_URL  used as a last-resort fallback
#   UV_PYTHON_INSTALL_MIRROR  mirror for managed CPython downloads (domestic candidates validated for real binary content)
#   UV_NATIVE_TLS              use the system certificate store for uv (default: 1, set 0 to disable)
#
# Uninstall:
#   uv tool uninstall mindstudio-agent
#   Optionally remove the PATH entries this installer added to your profiles.
#
# Upgrade:
#   Just re-run the installer; it always upgrades to the latest release.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# 输出风格与日志
#
# 交互终端启用颜色；输出被重定向（CI、日志、测试）时自动退化为纯文本。
# MSAGENT_PLAIN_UI=1 强制纯文本，FORCE_COLOR=1 强制彩色。不使用 emoji 与制表符，
# 保证在不同终端、不同代码页下都能正常显示。
# ---------------------------------------------------------------------------
if { [ -t 1 ] && [ "${MSAGENT_PLAIN_UI:-0}" != "1" ]; } || [ "${FORCE_COLOR:-}" = "1" ]; then
  UI_FANCY=1
  # $'...' 存放真正的 ESC 字节：调色板也会作为 printf 参数（%s）传入，
  # 那种场景下转义序列不会被解释。
  RED=$'\033[1;31m'; GREEN=$'\033[1;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[1;34m'
  CYAN=$'\033[1;36m'; WHITE=$'\033[1;37m'; DIM=$'\033[2m'; BOLD=$'\033[1m'; NC=$'\033[0m'
  # 章节/卡片标题用反显徽标，重点行用亮白加粗，便于一眼扫到关键信息。
  HL_BADGE=$'\033[1;44;97m'
  HL_TEXT=$'\033[1;97m'
else
  UI_FANCY=0
  RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; WHITE=''; DIM=''; BOLD=''; NC=''
  HL_BADGE=''; HL_TEXT=''
fi

# 行内高亮：强调"安装哪个版本""验证通过"这类重点信息。
bold() { printf '%s%s%s' "${HL_TEXT}" "$*" "${NC}"; }

# 英文 TAG，按 5 列对齐，便于跨终端/不同语言环境阅读。
TAG_INFO='INFO '
TAG_OK='OK   '
TAG_WARN='WARN '
TAG_ERR='ERROR'

log_info()    { printf '  %s%s%s %s\n' "${BLUE}" "${TAG_INFO}" "${NC}" "$*"; }
log_success() { printf '  %s%s%s %s\n' "${GREEN}" "${TAG_OK}" "${NC}" "$*"; }
log_warn()    { printf '  %s%s%s %s\n' "${YELLOW}" "${TAG_WARN}" "${NC}" "$*" >&2; }
log_error()   { printf '  %s%s%s %s\n' "${RED}" "${TAG_ERR}" "${NC}" "$*" >&2; }

# 显示宽度：中文等全角字符按 2 列计，避免用 %-Ns 按字符数补位时错位。
# 用码点数值判断，而不是 case 的字符区间（后者的匹配受 locale 排序规则影响）。
display_width() {
  local s="$1" w=0 i=0 ch cp
  while [ "${i}" -lt "${#s}" ]; do
    ch="${s:${i}:1}"
    cp="$(printf '%d' "'${ch}" 2>/dev/null || printf '0')"
    if [ "${cp}" -gt 127 ]; then
      w=$((w + 2))
    else
      w=$((w + 1))
    fi
    i=$((i + 1))
  done
  printf '%s' "${w}"
}

# pad_display <文本> <目标显示宽度>
pad_display() {
  local text="$1" target="$2" pad
  pad=$((target - $(display_width "${text}")))
  [ "${pad}" -lt 0 ] && pad=0
  printf '%s%*s' "${text}" "${pad}" ""
}

# ---------------------------------------------------------------------------
# 进度与结果汇总
#
# 分阶段编号让长流程可读，SUMMARY_* 收集的值在最后一次性打印，
# 避免被中间的日志冲掉。
# ---------------------------------------------------------------------------
TOTAL_PHASES=6
PHASE=0
SUMMARY_VERSION=""
SUMMARY_PREV_VERSION=""
SUMMARY_TOOL_BIN=""
SUMMARY_PYPI_INDEX=""
SUMMARY_NODE=""
SUMMARY_DOC_MCP=""
SUMMARY_MSPROF=""
SUMMARY_NPM_REGISTRY=""
NODE_VERSION_RESOLVED=""
MCP_PACKAGE_VERSION_RESOLVED=""
NPM_REGISTRY_USED=""

phase() {
  PHASE=$((PHASE + 1))
  if [ "${UI_FANCY}" = "1" ]; then
    # 反显徽标 + 标题，让每个章节在滚动中一眼可见。
    printf '\n%s [%s/%s] %s %s\n' "${HL_BADGE}" "${PHASE}" "${TOTAL_PHASES}" "$*" "${NC}"
  else
    printf '\n[%s/%s] %s\n' "${PHASE}" "${TOTAL_PHASES}" "$*"
  fi
}

BANNER_ART=$(cat <<'MSAGENT_BANNER'
                                        _
   _ __ ___  ___  __ _  __ _  ___ _ __ | |_
  | '_ ` _ \/ __|/ _` |/ _` |/ _ \ '_ \| __|
  | | | | | \__ \ (_| | (_| |  __/ | | | |_
  |_| |_| |_|___/\__,_|\__, |\___|_| |_|\__|
                       |___/
MSAGENT_BANNER
)

print_banner() {
  printf '\n'
  if [ "${UI_FANCY}" = "1" ]; then
    printf '%s%s%s\n' "${CYAN}" "${BANNER_ART}" "${NC}"
  else
    printf '%s\n' "${BANNER_ART}"
  fi
  if [ "${UI_FANCY}" = "1" ]; then
    printf '  %sMindStudio Agent Installer%s\n' "${DIM}" "${NC}"
  else
    printf '  MindStudio Agent Installer\n'
  fi
}

RULE_WIDTH=42
DASHES='------------------------------------------'

# 结果卡片：标题用反显徽标 + 一条分隔线，键值行保持对齐（不用方框字符）。
card_top() {
  local title="$1" total pad
  total=$((RULE_WIDTH - $(display_width "${title}") - 2))
  pad=$((total / 2))
  if [ "${UI_FANCY}" = "1" ]; then
    printf '\n  %s%s %s %s%s\n' "${DIM}" "${DASHES:0:${pad}}" "${HL_BADGE} ${title} ${NC}" "${DASHES:0:$((total - pad))}" "${NC}"
  else
    printf '\n  %s%s %s %s%s\n' "${DIM}" "${DASHES:0:${pad}}" "${title}" "${DASHES:0:$((total - pad))}" "${NC}"
  fi
}

card_row() {
  local key="$1" value="$2" colour="${3:-${NC}}"
  printf '  %s%s%s%s%s%s\n' "${BOLD}" "$(pad_display "${key}" 18)" "${NC}" "${colour}" "${value}" "${NC}"
}

card_bottom() {
  printf '  %s%s%s\n' "${DIM}" "${DASHES}" "${NC}"
}

# 最终汇总：装到了哪里、用了哪个源。
print_summary() {
  card_top "安装结果"
  case "${SUMMARY_VERSION}" in
    *"未加载 PATH"*)     card_row "安装版本" "${SUMMARY_VERSION}" "${YELLOW}" ;;
    *"已是最新"*)        card_row "安装版本" "${SUMMARY_VERSION}" "${GREEN}" ;;
    "")
      card_row "安装版本" "已安装" "${GREEN}" ;;
    *)
      if [ -n "${SUMMARY_PREV_VERSION}" ] && [ "${SUMMARY_PREV_VERSION}" != "${SUMMARY_VERSION}" ]; then
        card_row "安装版本" "${SUMMARY_VERSION}（由 ${SUMMARY_PREV_VERSION} 升级）" "${GREEN}"
      else
        card_row "安装版本" "${SUMMARY_VERSION}" "${GREEN}"
      fi
      ;;
  esac
  [ -n "${SUMMARY_TOOL_BIN}" ] && card_row "安装目录" "${SUMMARY_TOOL_BIN}"
  card_row "PyPI 源" "${SUMMARY_PYPI_INDEX:-${INDEX:-未知}}"
  [ -n "${SUMMARY_NPM_REGISTRY}" ] && card_row "npm 源" "${SUMMARY_NPM_REGISTRY}"
  [ -n "${SUMMARY_NODE}" ] && card_row "Node.js" "${SUMMARY_NODE}" "${GREEN}"
  if [ -n "${SUMMARY_DOC_MCP}" ]; then
    case "${SUMMARY_DOC_MCP}" in
      v*)  card_row "文档查询服务" "${SUMMARY_DOC_MCP}" "${GREEN}" ;;
      *)   card_row "文档查询服务" "${SUMMARY_DOC_MCP}" "${YELLOW}" ;;
    esac
  fi
  if [ -n "${SUMMARY_MSPROF}" ]; then
    case "${SUMMARY_MSPROF}" in
      正常) card_row "msprof-mcp" "${SUMMARY_MSPROF}" "${GREEN}" ;;
      *)    card_row "msprof-mcp" "${SUMMARY_MSPROF}" "${YELLOW}" ;;
    esac
  fi
  card_bottom
}

print_next_steps() {
  card_top "后续步骤"
  printf '  %s1) 配置模型与 API Key%s\n' "${BOLD}" "${NC}"
  printf '       export OPENAI_API_KEY="<你的 API Key>"   # Anthropic: ANTHROPIC_API_KEY，Google: GOOGLE_API_KEY\n'
  printf '       msagent config --llm-provider openai --llm-base-url "<你的服务地址>" --llm-model "<模型名>"\n'
  printf '       # 用服务商官方端点时可省略 --llm-base-url\n'
  printf '       msagent config --show                   # 确认配置已生效\n'
  printf '  %s2) 启动会话%s\n' "${BOLD}" "${NC}"
  printf '       msagent                                 # 更多用法：msagent --help\n'
  printf '  %s3) 若提示 command not found%s\n' "${BOLD}" "${NC}"
  printf '       source %s       # 或新开一个终端窗口（PATH 在本次会话中尚未生效）\n' "${PROFILE_FILE:-~/.bashrc}"
  card_bottom
}

# CLI 会打印整段横幅，这里只取版本号（如 26.1.2）。
# 先剥掉 ANSI 颜色与 \r：某些环境（设置了 FORCE_COLOR 等）下横幅带转义序列，
# "msagent" 与版本号之间会夹着颜色码，直接匹配会失败。
msagent_version_token() {
  sed -e 's/\x1b\[[0-9;]*[A-Za-z]//g' -e 's/\r//g' \
    | sed -n 's/.*msagent[[:space:]]\{1,\}\([0-9][0-9.]*\).*/\1/p' \
    | head -n 1
}

# Works both interactively and under `curl ... | bash` (reads from /dev/tty).
prompt_yn() {
  [ "${MSAGENT_YES:-0}" = "1" ] && return 0
  local answer
  printf "${YELLOW}?${NC} %s [y/N] " "$*" >&2
  if [ -t 0 ]; then
    read -r answer || return 1
  elif [ -r /dev/tty ]; then
    read -r answer < /dev/tty || return 1
  else
    log_warn "非交互式 shell，按"否"处理。"
    return 1
  fi
  case "${answer}" in
    y|Y|yes|YES|Yes) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# OS / environment detection
# ---------------------------------------------------------------------------
detect_os() {
  case "$(uname -s)" in
    Darwin) OS="macos" ;;
    Linux)  OS="linux" ;;
    MINGW*|MSYS*|CYGWIN*) OS="windows" ;;
    *) OS="unknown" ;;
  esac
}
detect_os

case "${OS}" in
  linux|macos) ;;
  windows)
    log_error "本安装脚本支持 Linux、macOS 与 WSL 环境。"
    log_error "原生 Windows 请在 PowerShell 中执行："
    log_error "  irm https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.ps1 | iex"
    exit 1
    ;;
  *)
    log_error "不支持的操作系统，请手动安装 msagent："
    log_error "  uv tool install -U mindstudio-agent"
    log_error "若没有 uv，可参考本脚本中的 venv 兜底安装逻辑。"
    exit 1
    ;;
esac

# macOS MDM/root installs may run without a usable HOME. Re-home to the real
# console user so uv installs into the expected profile directory.
if [ "${OS}" = "macos" ] && { [ -z "${HOME:-}" ] || [ "$(id -u)" -eq 0 ]; }; then
  CONSOLE_USER="$(stat -f '%Su' /dev/console 2>/dev/null || true)"
  if [ -n "${CONSOLE_USER}" ] && [ "${CONSOLE_USER}" != "root" ] && [ -d "/Users/${CONSOLE_USER}" ]; then
    HOME="/Users/${CONSOLE_USER}"
    export HOME
  fi
fi

# root runs: install into the invoking user's profile, then fix ownership.
if [ "$(id -u)" -eq 0 ]; then
  TARGET_USER="${SUDO_USER:-${CONSOLE_USER:-$(basename "${HOME:-root}")}}"
  if [ -z "${TARGET_USER}" ] || [ "${TARGET_USER}" = "root" ]; then
    fix_owner() { :; }
  else
    fix_owner() {
      chown -R "${TARGET_USER}" "$@" 2>/dev/null || \
        log_warn "无法修正 $* 的属主"
    }
  fi
else
  fix_owner() { :; }
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
print_banner
phase "检查环境并选择下载源"
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  log_error "需要 curl 或 wget 才能下载安装所需文件。"
  exit 1
fi

probe_url() {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fsS -o /dev/null --connect-timeout 6 --max-time 12 -I "${url}" 2>/dev/null && return 0
    curl -fsS -o /dev/null --connect-timeout 6 --max-time 12 "${url}" 2>/dev/null && return 0
  elif command -v wget >/dev/null 2>&1; then
    wget -q --spider --timeout=6 --tries=1 "${url}" 2>/dev/null && return 0
  fi
  return 1
}

# Warn about an existing pip-installed msagent that may shadow the uv tool.
if command -v pip >/dev/null 2>&1 && pip show mindstudio-agent >/dev/null 2>&1; then
  log_warn "当前环境已通过 pip 安装 mindstudio-agent。"
  log_warn "下面的 uv 工具安装是隔离环境，不会改动它，但两个可执行文件"
  log_warn "可能在 PATH 中互相遮蔽，建议执行："
  log_warn "  pip uninstall mindstudio-agent"
fi

# ---------------------------------------------------------------------------
# Index selection: domestic mirrors first (fast in China), official PyPI as
# fallback; generic env vars last resort. Mirror sync may lag weekly
# releases, so the install is retried with official PyPI on failure (see the
# install section), and the latest version is pinned from PyPI beforehand.
# ---------------------------------------------------------------------------
select_index() {
  local candidate
  if [ -n "${MSAGENT_INDEX:-}" ]; then
    INDEX="${MSAGENT_INDEX}"
    log_info "使用 MSAGENT_INDEX 指定的源：${INDEX}"
    return 0
  fi
  for candidate in \
    "https://mirrors.huaweicloud.com/repository/pypi/simple" \
    "https://mirrors.aliyun.com/pypi/simple" \
    "https://pypi.tuna.tsinghua.edu.cn/simple" \
    "https://pypi.org/simple"; do
    if probe_url "${candidate}/pip/"; then
      INDEX="${candidate}"
      log_info "已选择 PyPI 源：${candidate}"
      return 0
    fi
    log_warn "源不可达：${candidate}（尝试下一个）"
  done
  # All candidates are unreachable; fall back to a user-configured
  # generic index if one exists, otherwise use official PyPI.
  if [ -n "${UV_DEFAULT_INDEX:-}" ]; then
    INDEX="${UV_DEFAULT_INDEX}"
    log_info "回退到 UV_DEFAULT_INDEX：${INDEX}"
    return 0
  fi
  if [ -n "${UV_INDEX_URL:-}" ]; then
    INDEX="${UV_INDEX_URL}"
    log_info "回退到 UV_INDEX_URL：${INDEX}"
    return 0
  fi
  if [ -n "${PIP_INDEX_URL:-}" ]; then
    INDEX="${PIP_INDEX_URL}"
    log_info "回退到 PIP_INDEX_URL：${INDEX}"
    return 0
  fi
  INDEX="https://pypi.org/simple"
  log_warn "没有可用源，默认使用 ${INDEX}。"
}
select_index

# ---------------------------------------------------------------------------
# 版本号排序工具。
# 不用 GNU 的 `sort -V`：macOS/BSD 自带的 sort 没有 -V（会直接报错），而且
# 镜像目录里同一版本往往有多个文件（.tar.gz/.tar.xz、whl/tar.gz），文本顺序
# 的最后一个既不等于最新版本，也不能在多个版本间作可靠比较。这里用 awk 把
# 版本号逐段数值化，并把 a1/rc/post 等后缀按 PEP 440 的先后关系排序。
# 输入每行 "<版本号>\t<原始文本>"，按版本号降序输出其中的原始文本。
# ---------------------------------------------------------------------------
version_sort_desc() {
  awk -F'\t' '
    function suffix_rank(suffix) {
      if (suffix == "") return 3
      suffix = tolower(suffix)
      if (suffix ~ /^\.?dev/) return 0
      if (suffix ~ /^\.?(a|alpha)/) return 1
      if (suffix ~ /^\.?(post|rev|r[0-9]*$)/) return 4
      return 2
    }
    function suffix_number(suffix,   digits) {
      if (suffix !~ /[0-9]/) return 0
      digits = suffix
      sub(/^[^0-9]*/, "", digits)
      sub(/[^0-9].*$/, "", digits)
      return digits + 0
    }
    function version_key(version,   core, suffix, fields, count, i, key) {
      core = version
      suffix = ""
      if (match(version, /^[0-9][0-9.]*/)) {
        core = substr(version, RSTART, RLENGTH)
        suffix = substr(version, RSTART + RLENGTH)
      }
      count = split(core, fields, ".")
      key = ""
      for (i = 1; i <= 3; i++) key = key sprintf("%05d", fields[i] + 0)
      return key sprintf("%d", suffix_rank(suffix)) sprintf("%05d", suffix_number(suffix))
    }
    { key[NR] = version_key($1); line[NR] = $0 }
    END {
      for (i = 1; i <= NR; i++) {
        best = i
        for (j = i + 1; j <= NR; j++) { if (key[j] > key[best]) best = j }
        if (best != i) {
          swap = key[i]; key[i] = key[best]; key[best] = swap
          swap = line[i]; line[i] = line[best]; line[best] = swap
        }
      }
      for (i = 1; i <= NR; i++) { print line[i] }
    }
  ' | cut -f2-
}

# 从 "node-v22.23.2-linux-x64.tar.xz" 这类文件名里取出版本号（22.23.2）。
archive_version() {
  printf '%s' "$1" | sed -n 's/^.*node-v\([0-9][0-9.]*\)-.*$/\1/p'
}

# ---------------------------------------------------------------------------
# Resolve and announce the msagent version to install. Mirror sync may lag
# weekly releases, so the latest version is pinned straight from PyPI and
# shown up front; the install is retried with official PyPI if the selected
# mirror does not have that version yet.
# ---------------------------------------------------------------------------
# 解析并提前告知要安装的 msagent 版本号。
# 1) MSAGENT_VERSION 显式指定；
# 2) 否则先问 PyPI 官方（拿到全局最新版）；
# 3) 再问所选镜像源的 simple 索引，保证提示里的版本与实际安装源一致。
# 拿不到版本号时明确说明"以源提供的最新版为准"，而不是含糊地说 latest。
latest_version_from_index() {
  # 从 PyPI simple 索引的页面里解析出最高版本号
  local base="$1" html
  [ -n "${base}" ] || return 1
  if command -v curl >/dev/null 2>&1; then
    html="$(curl -fsSL --connect-timeout 6 --max-time 20 "${base%/}/mindstudio-agent/" 2>/dev/null || true)"
  elif command -v wget >/dev/null 2>&1; then
    html="$(wget -qO- --timeout=20 "${base%/}/mindstudio-agent/" 2>/dev/null || true)"
  else
    return 1
  fi
  [ -n "${html}" ] || return 1
  printf '%s\n' "${html}" \
    | grep -oE 'mindstudio[_-]agent-[0-9][0-9A-Za-z.+-]*\.(whl|tar\.gz)' \
    | sed -E 's/^mindstudio[_-]agent-//; s/\.(whl|tar\.gz)$//; s/-(py3|py2|cp3)[^.]*$//' \
    | sort -u \
    | awk '{ print $1 "\t" $1 }' \
    | version_sort_desc | head -n 1
}

MSAGENT_PYTHON="${MSAGENT_PYTHON:->=3.11}"
INSTALL_SPEC="mindstudio-agent"
LATEST_VERSION=""
if [ -n "${MSAGENT_VERSION:-}" ]; then
  LATEST_VERSION="${MSAGENT_VERSION}"
  VERSION_SOURCE="由 MSAGENT_VERSION 指定"
else
  VERSION_SOURCE="最新版，源：${INDEX}"
  if [ -z "${MSAGENT_INDEX:-}" ]; then
    if command -v curl >/dev/null 2>&1; then
      LATEST_VERSION="$(curl -fsSL --connect-timeout 6 --max-time 15 https://pypi.org/pypi/mindstudio-agent/json 2>/dev/null | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
    elif command -v wget >/dev/null 2>&1; then
      LATEST_VERSION="$(wget -qO- --timeout=15 https://pypi.org/pypi/mindstudio-agent/json 2>/dev/null | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4 || true)"
    fi
  fi
  if [ -z "${LATEST_VERSION}" ]; then
    LATEST_VERSION="$(latest_version_from_index "${INDEX}" || true)"
  fi
fi
if [ -n "${LATEST_VERSION}" ]; then
  INSTALL_SPEC="mindstudio-agent==${LATEST_VERSION}"
  log_info "$(bold "即将安装 msagent ${LATEST_VERSION}（${VERSION_SOURCE}）")"
else
  log_warn "未能从 ${INDEX} 解析出版本号，将安装该源提供的最新版 mindstudio-agent。"
fi
SUMMARY_PYPI_INDEX="${INDEX}"

# Record what is currently installed, so the summary can report an update
# instead of a bare version ("updated 26.1.2 -> 26.1.3").
SUMMARY_PREV_VERSION=""
PREV_MSAGENT="$(command -v msagent 2>/dev/null || true)"
if [ -n "${PREV_MSAGENT}" ]; then
  SUMMARY_PREV_VERSION="$("${PREV_MSAGENT}" --version 2>/dev/null | msagent_version_token || true)"
fi

# ---------------------------------------------------------------------------
# uv bootstrap: official installer, then pip + domestic mirror as fallback.
# ---------------------------------------------------------------------------
phase "准备 uv（Python 包管理器）"
UV_BIN="${UV_BIN:-}"
bootstrap_uv() {
  if [ -n "${UV_BIN:-}" ]; then
    if [ -x "${UV_BIN}" ] || command -v "${UV_BIN}" >/dev/null 2>&1; then
      log_info "使用 UV_BIN 指定的 uv：${UV_BIN}"
      return 0
    fi
    log_warn "UV_BIN 已设置但不可执行：${UV_BIN}"
    UV_BIN=""
  fi
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="uv"
    return 0
  fi
  if [ -x "${HOME}/.local/bin/uv" ]; then
    UV_BIN="${HOME}/.local/bin/uv"
    return 0
  fi

  log_info "未检测到 uv，正在自动安装 uv ..."
  if command -v curl >/dev/null 2>&1 && curl -fsSL https://astral.sh/uv/install.sh | sh; then
    :
  elif command -v wget >/dev/null 2>&1 && wget -qO- https://astral.sh/uv/install.sh | sh; then
    :
  else
    log_warn "uv 官方安装脚本失败或超时，改用 pip + ${INDEX} ..."
    local pybin=""
    if command -v python3 >/dev/null 2>&1; then pybin="python3"
    elif command -v python >/dev/null 2>&1; then pybin="python"
    else
      log_error "通过 pip 安装 uv 需要 python3。"
      log_error "请手动安装 uv（https://docs.astral.sh/uv/getting-started/installation/）后重试。"
      return 1
    fi
    local errfile
    errfile="$(mktemp)"
    if ! "${pybin}" -m pip install --user -q -U uv -i "${INDEX}" 2>"${errfile}"; then
      if grep -q "externally-managed" "${errfile}" 2>/dev/null; then
        log_warn "检测到 PEP 668 托管环境，改用 --break-system-packages 重试 ..."
        "${pybin}" -m pip install --user -q -U --break-system-packages uv -i "${INDEX}" || {
          cat "${errfile}" >&2
          rm -f "${errfile}"
          log_error "uv 安装失败，请手动安装后重试本脚本。"
          return 1
        }
      else
        cat "${errfile}" >&2
        rm -f "${errfile}"
        log_error "uv 安装失败，请手动安装后重试本脚本。"
        return 1
      fi
    fi
    rm -f "${errfile}"
  fi

  if [ -x "${HOME}/.local/bin/uv" ]; then
    UV_BIN="${HOME}/.local/bin/uv"
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="uv"
    return 0
  fi
  if [ -f "${HOME}/.local/bin/env" ]; then
    # shellcheck source=/dev/null
    . "${HOME}/.local/bin/env"
    if command -v uv >/dev/null 2>&1; then
      UV_BIN="uv"
      return 0
    fi
  fi
  log_error "uv 仍不可用：请重开终端后重试，或把 ~/.local/bin 加入 PATH。"
  return 1
}
bootstrap_uv

# ---------------------------------------------------------------------------
# Use the platform's native TLS store (Windows Schannel / system CA bundle).
# Corporate proxies re-sign HTTPS with their own CA: system tools (curl/wget)
# trust it, but uv's bundled roots reject it with "UnknownIssuer". Only an
# explicitly set UV_NATIVE_TLS wins.
# ---------------------------------------------------------------------------
if [ -z "${UV_NATIVE_TLS:-}" ]; then
  export UV_NATIVE_TLS=1
  log_info "UV_NATIVE_TLS=1（uv 使用系统证书库，适配企业代理证书）"
fi

# ---------------------------------------------------------------------------
# Managed CPython downloads: try Huawei Cloud first, then NJU, but only when
# the mirror actually serves real binary content with a trusted certificate
# (some public github-release mirrors answer HTML with HTTP 200, and corporate
# proxies may reject certificates). Otherwise leave uv's defaults (official
# source or an existing local Python) to handle it.
# ---------------------------------------------------------------------------
python_platform_triple() {
  case "$(uname -s)/$(uname -m)" in
    Linux/x86_64)  echo "x86_64-unknown-linux-gnu" ;;
    Linux/aarch64) echo "aarch64-unknown-linux-gnu" ;;
    Darwin/x86_64) echo "x86_64-apple-darwin" ;;
    Darwin/arm64)  echo "aarch64-apple-darwin" ;;
    MINGW*|MSYS*|CYGWIN*/x86_64) echo "x86_64-pc-windows-msvc" ;;
    *) echo "" ;;
  esac
}

# Discover one real python-build-standalone file on the mirror from its own
# directory listing (newest tag), e.g.
# "20260825/cpython-3.11.16+20260825-aarch64-unknown-linux-gnu-install_only.tar.gz".
# No tag or version is hardcoded, so the probe stays valid as releases move on.
python_canary_file() {
  local mirror="$1" platform tag listing file
  platform="$(python_platform_triple)"
  [ -n "${platform}" ] || return 1
  if ! command -v curl >/dev/null 2>&1; then
    return 1
  fi
  listing="$(curl -fsSL --connect-timeout 6 --max-time 15 "${mirror}/" 2>/dev/null || true)"
  tag="$(printf '%s\n' "${listing}" | grep -oE '[0-9]{8}/' | tail -1 | tr -d '/')"
  [ -n "${tag}" ] || return 1
  listing="$(curl -fsSL --connect-timeout 6 --max-time 15 "${mirror}/${tag}/" 2>/dev/null || true)"
  file="$(printf '%s\n' "${listing}" | grep -oE "cpython-[0-9.]+%2B[0-9]{8}-${platform}-install_only[^\"<]*\.tar\.gz" | head -1)"
  [ -n "${file}" ] || return 1
  printf '%s/%s\n' "${tag}" "${file}"
}

probe_python_mirror() {
  local mirror="$1" canary hdr
  canary="$(python_canary_file "${mirror}")"
  [ -n "${canary}" ] || return 1
  hdr="$(curl -fsSI --connect-timeout 6 --max-time 12 "${mirror}/${canary}" 2>/dev/null || true)"
  [ -z "${hdr}" ] && return 1
  printf '%s\n' "${hdr}" | grep -qi "content-type:.*text/html" && return 1
  return 0
}

if [ -z "${UV_PYTHON_INSTALL_MIRROR:-}" ]; then
  PY_MIRROR_OK=""
  for mirror in \
    "https://mirrors.huaweicloud.com/github-release/astral-sh/python-build-standalone" \
    "https://mirror.nju.edu.cn/github-release/astral-sh/python-build-standalone"; do
    if probe_python_mirror "${mirror}"; then
      export UV_PYTHON_INSTALL_MIRROR="${mirror}"
      log_info "UV_PYTHON_INSTALL_MIRROR=${mirror}"
      PY_MIRROR_OK=1
      break
    fi
  done
  if [ -z "${PY_MIRROR_OK:-}" ]; then
    log_warn "Python 下载镜像都不可用，将使用 uv 默认源或本地 Python；"
    log_warn "如需内网镜像请设置 UV_PYTHON_INSTALL_MIRROR（证书被代理拦截时可加 UV_INSECURE_HOST=<主机>）。"
  fi
fi

# ---------------------------------------------------------------------------
# Path setup helpers (defined before use)
# ---------------------------------------------------------------------------
ORIGINAL_PATH="${PATH:-}"
TOOL_BIN_DIR="$("${UV_BIN}" tool dir --bin 2>/dev/null || true)"
[ -n "${TOOL_BIN_DIR}" ] || TOOL_BIN_DIR="${HOME}/.local/bin"
PROFILE_FILE="${HOME}/.bashrc"

path_contains() {
  case ":${ORIGINAL_PATH}:" in
    *":$1:"*) return 0 ;;
    *) return 1 ;;
  esac
}

add_to_profile() {
  local profile="$1"
  local line="export PATH=\"${TOOL_BIN_DIR}:\$PATH\""
  if [ -f "${profile}" ] && grep -qF "${line}" "${profile}" 2>/dev/null; then
    return 0
  fi
  touch "${profile}" 2>/dev/null || return 1
  printf '\n# Added by the msagent installer\n%s\n' "${line}" >> "${profile}"
  log_success "已把 ${TOOL_BIN_DIR} 写入 ${profile} 的 PATH"
}

setup_path() {
  if [ "${MSAGENT_NO_MODIFY_PATH:-0}" = "1" ]; then
    log_warn "已设置 MSAGENT_NO_MODIFY_PATH，跳过 PATH 修改。"
    log_warn "请自行把 ${TOOL_BIN_DIR} 加入 PATH，例如："
    log_warn "  export PATH=\"${TOOL_BIN_DIR}:\$PATH\""
    return 0
  fi
  if path_contains "${TOOL_BIN_DIR}"; then
    log_info "${TOOL_BIN_DIR} 已在 PATH 中。"
    return 0
  fi
  PROFILE_FILE=""
  # fish
  if command -v fish >/dev/null 2>&1 || [ -d "${HOME}/.config/fish" ]; then
    local fish_dir="${HOME}/.config/fish/conf.d"
    mkdir -p "${fish_dir}" 2>/dev/null || true
    local fish_file="${fish_dir}/msagent-path.fish"
    if [ ! -f "${fish_file}" ]; then
      printf 'fish_add_path %s\n' "${TOOL_BIN_DIR}" > "${fish_file}" 2>/dev/null || true
      log_success "已通过 ${fish_file} 添加 ${TOOL_BIN_DIR}"
    fi
  fi
  # bash/zsh: cover every profile a login or interactive shell may read.
  # Only create ~/.bashrc (safe everywhere); existing ~/.bash_profile and
  # ~/.profile are updated too so login shells pick the entry up.
  local updated=0
  local candidate
  for candidate in "${HOME}/.bashrc" "${HOME}/.bash_profile" "${HOME}/.profile"; do
    if [ -f "${candidate}" ] || [ "${candidate}" = "${HOME}/.bashrc" ]; then
      if add_to_profile "${candidate}"; then
        [ -z "${PROFILE_FILE}" ] && PROFILE_FILE="${candidate}"
        updated=1
      fi
    fi
  done
  if [ -n "${SHELL:-}" ] && [[ "${SHELL}" == *zsh* ]]; then
    local zshrc="${HOME}/.zshrc"
    [ -n "${ZDOTDIR:-}" ] && zshrc="${ZDOTDIR}/.zshrc"
    if add_to_profile "${zshrc}"; then
      [ -z "${PROFILE_FILE}" ] && PROFILE_FILE="${zshrc}"
      updated=1
    fi
  fi
  if [ "${updated}" = "1" ]; then
    log_success "执行 'source ${PROFILE_FILE}'（或重开终端）后即可使用 msagent。"
  else
    log_warn "未能更新 shell 配置，请自行把 '${TOOL_BIN_DIR}' 加入 PATH。"
  fi
  # uv's own shell-profile updater covers more edge cases; best effort only.
  "${UV_BIN}" tool update-shell >/dev/null 2>&1 || true
  fix_owner "${TOOL_BIN_DIR}" "${HOME}/.local/share/uv" "${HOME}/.cache/uv" 2>/dev/null || true
}

# Last-resort fallback when `uv tool install` fails: an isolated venv.
install_venv_fallback() {
  local venv_dir="${MSAGENT_FALLBACK_VENV:-${HOME}/.msagent-venv}"
  local pybin=""
  if command -v python3 >/dev/null 2>&1; then pybin="python3"
  elif command -v python >/dev/null 2>&1; then pybin="python"
  else
    log_error "venv 兜底安装需要 python3。"
    return 1
  fi
  log_info "正在 ${venv_dir} 创建独立 venv ..."
  if ! "${pybin}" -m venv "${venv_dir}" 2>/dev/null; then
    # python3-venv (ensurepip) may be missing; uv can create venvs without it.
    if [ -n "${UV_BIN:-}" ] && "${UV_BIN}" venv "${venv_dir}" 2>/dev/null; then
      log_success "已用 uv 创建 venv（系统未安装 python3-venv）。"
    else
      log_error "无法创建虚拟环境（可能缺少 python3-venv）。"
      log_error "Debian/Ubuntu 上请先执行：sudo apt install python3-venv"
      log_error "或改用推荐方式重试："
      log_error "  ${UV_BIN} tool install -U --python \"${MSAGENT_PYTHON}\" --default-index \"${INDEX}\" \"${INSTALL_SPEC}\""
      log_error "如需手动兜底安装："
      log_error "  ${pybin} -m venv ${venv_dir} && ${venv_dir}/bin/pip install -U -i ${INDEX} ${INSTALL_SPEC}"
      return 1
    fi
  fi
  log_info "正在把 ${INSTALL_SPEC}（源：${INDEX}）安装到 venv ..."
  if [ -x "${venv_dir}/bin/pip" ]; then
    "${venv_dir}/bin/pip" install -q -U -i "${INDEX}" "${INSTALL_SPEC}" || return 1
  elif [ -n "${UV_BIN:-}" ]; then
    # A venv created by uv has no pip; install with uv pip instead.
    "${UV_BIN}" pip install --python "${venv_dir}/bin/python" -q -U --index-url "${INDEX}" "${INSTALL_SPEC}" || return 1
  else
    return 1
  fi
  TOOL_BIN_DIR="${venv_dir}/bin"
  setup_path
  local vout
  vout="$("${TOOL_BIN_DIR}/msagent" --version 2>&1)" || true
  log_success "验证通过：${vout}"
  log_warn "已通过 venv 兜底方式安装（${venv_dir}）。"
  log_warn "建议改用 'uv tool install'：uv 可用后重跑本脚本即可切换。"
}

# uv sometimes only links the primary tool's executables into the tool bin
# directory. The msprof-mcp MCP server is spawned by name from PATH, so make
# sure its executable (and msprof-analyze) are reachable. Best effort only.
expose_tool_executables() {
  local tool_env_bin
  tool_env_bin="$("${UV_BIN}" tool dir 2>/dev/null || true)/mindstudio-agent/bin"
  [ -d "${tool_env_bin}" ] || return 0
  local exe
  for exe in msprof-mcp msprof-analyze; do
    if [ -x "${tool_env_bin}/${exe}" ] && [ ! -e "${TOOL_BIN_DIR}/${exe}" ]; then
      ln -s "${tool_env_bin}/${exe}" "${TOOL_BIN_DIR}/${exe}" 2>/dev/null && \
        log_success "已通过 ${TOOL_BIN_DIR}/${exe} 暴露 ${exe}"
    fi
  done
}

# ---------------------------------------------------------------------------
# Optional stage: provision a user-local Node.js (npmmirror mirror) and
# pre-install @opencxd/ascend-doc-mcp into ~/.msagent/ascend-doc-mcp. The
# msagent ascend-doc-mcp launcher then runs the pre-installed copy directly:
# no global Node/npx and no on-demand npm fetch are needed on first use.
#
#   MSAGENT_NO_ASCEND_DOC_MCP=1        skip this stage entirely
#   MSAGENT_ASCEND_DOC_MCP_REQUIRE=1   abort the installer if this stage fails
#   MSAGENT_NODE_HOME                  user-local Node root (default ~/.msagent/node)
#   MSAGENT_NODE_MIRROR                Node dist mirror base (default: probed
#                                      huaweicloud -> npmmirror -> nodejs.org)
#   MSAGENT_NPM_REGISTRY               npm registry (default: the machine's
#                                      configured npm registry, then probed
#                                      npmmirror -> huaweicloud -> npmjs)
#   MSAGENT_NPM_CACHE                  npm cache dir (default ~/.cache/msagent/npm-cache)
#   MSAGENT_ASCEND_DOC_MCP_PREFIX      local install prefix (~/.msagent/ascend-doc-mcp)
#
# Every mirror below is probed before use, so the stage works on the public
# internet and inside a corporate intranet (the Huawei Cloud mirrors are usually
# reachable where registry.npmmirror.com / registry.npmjs.org are not).
# ---------------------------------------------------------------------------
NODE_MIN_MAJOR=22
NODE_MIRROR_CANDIDATES=(
  "https://mirrors.huaweicloud.com/nodejs"
  "https://registry.npmmirror.com/-/binary/node"
  "https://nodejs.org/dist"
)
NPM_REGISTRY_CANDIDATES=(
  "https://registry.npmmirror.com"
  "https://mirrors.huaweicloud.com/repository/npm"
  "https://registry.npmjs.org"
)
# ---------------------------------------------------------------------------
# 公司内网（华为）npm 镜像，仅作为"最后兜底"候选：
#   * 排在公网候选链与外部注入的兜底源之后，只有前面全部失败（典型就是内网
#     环境）才会被访问，公网用户不会因此多花任何时间，也不参与首选源探测；
#   * 置空即可去掉这个内置兜底（例如对外发布时）；
#   * 内网地址变更时只需改这一行；
#   * 更通用的做法（脚本不含内网地址）：MSAGENT_NPM_REGISTRY_FALLBACKS 或
#     ~/.msagent/npm-mirrors（见 npm_extra_registries）。
# ---------------------------------------------------------------------------
INTRANET_NPM_REGISTRY="http://cmc-cd-mirror.rnd.huawei.com/npm"
HWCLOUD_NODE_MIRROR="https://mirrors.huaweicloud.com/nodejs"
HWCLOUD_NPM_REGISTRY="https://mirrors.huaweicloud.com/repository/npm"
NODE_BIN_DIR=""

# Echo the major version of a node binary ("22" for v22.23.2); empty on failure.
node_major_version() {
  local version
  version="$("$1" --version 2>/dev/null || true)"
  printf '%s' "${version#v}" | cut -d. -f1 | grep -E '^[0-9]+$' || true
}


is_public_registry() {
  # Registries the probe chain below already covers; used to ignore an npm
  # configuration that only points at a default, keeping the public path clean.
  case "${1%/}" in
    "https://registry.npmjs.org"|"http://registry.npmjs.org"|\
    "https://registry.npmmirror.com"|"http://registry.npmmirror.com"|\
    "https://mirrors.huaweicloud.com/repository/npm"|"http://mirrors.huaweicloud.com/repository/npm")
      return 0 ;;
  esac
  return 1
}

configured_npm_registry() {
  # The registry npm itself would use (env vars, project/user/global .npmrc).
  # Corporate intranets usually point npm at an internal mirror there, so
  # honoring it makes "use the intranet registry when we are on the intranet"
  # automatic without hardcoding any internal host in this script.
  local configured
  command -v npm >/dev/null 2>&1 || return 1
  configured="$(npm config get registry 2>/dev/null | tr -d '\r' | head -n 1 || true)"
  case "${configured}" in
    ""|"undefined"|"null") return 1 ;;
  esac
  is_public_registry "${configured}" && return 1
  printf '%s' "${configured}"
}

select_node_mirror() {
  # MSAGENT_NODE_MIRROR wins; otherwise probe the domestic mirrors first.
  local candidate
  if [ -n "${MSAGENT_NODE_MIRROR:-}" ]; then
    printf '%s' "${MSAGENT_NODE_MIRROR}"
    return 0
  fi
  for candidate in "${NODE_MIRROR_CANDIDATES[@]}"; do
    if probe_url "${candidate}/latest-v22.x/" >/dev/null 2>&1; then
      printf '%s' "${candidate}"
      return 0
    fi
    log_warn "Node 镜像不可达：${candidate}（尝试下一个）"
  done
  printf '%s' "${NODE_MIRROR_CANDIDATES[0]}"
}

# 版本比较（$1 > $2）：用 awk 逐段数值比较，避免 sort -V 在 macOS 上不可用。
version_gt() {
  awk -v a="$1" -v b="$2" 'BEGIN{
    n=split(a,x,"."); m=split(b,y,"."); k=(n>m?n:m);
    for(i=1;i<=k;i++){ ai=x[i]+0; bi=y[i]+0; if(ai>bi) exit 0; if(ai<bi) exit 1 }
    exit 1
  }'
}

# npm 需要的网络参数：内网通常要走代理，而 npm 自己不读系统代理设置。
# 有 HTTP(S)_PROXY/ALL_PROXY 时直接沿用，避免 curl 能通、npm 却 ETIMEDOUT。
npm_proxy_args() {
  local proxy_url="${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-${ALL_PROXY:-${all_proxy:-}}}}}}"
  if [ -n "${proxy_url}" ]; then
    printf '%s\n' --proxy "${proxy_url}" --https-proxy "${proxy_url}"
  fi
}

# 只在可选项失败时打印的精简修复指引。
ascend_doc_mcp_guidance() {
  log_warn "需要 Node.js >= ${NODE_MIN_MAJOR} 与可访问的 npm 源，补齐任一项后重跑即可；"
  log_warn "不需要该功能可加 MSAGENT_NO_ASCEND_DOC_MCP=1 跳过。参考："
  log_warn "  Node：${HWCLOUD_NODE_MIRROR}/latest-v22.x/ 下取最新的 node-v22.x.y-linux-x64.tar.xz"
  log_warn "        tar -xf node-v22.x.y-linux-x64.tar.xz -C /opt && export PATH=/opt/node-v22.x.y-linux-x64/bin:\$PATH"
  log_warn "  npm ：export MSAGENT_NPM_REGISTRY=${HWCLOUD_NPM_REGISTRY}"
}

node_platform_name() {
  local arch
  case "$(uname -m)" in
    x86_64|amd64) arch="x64" ;;
    aarch64|arm64) arch="arm64" ;;
    *) return 1 ;;
  esac
  case "$(uname -s)" in
    Darwin) printf 'darwin-%s' "${arch}" ;;
    Linux) printf 'linux-%s' "${arch}" ;;
    *) return 1 ;;
  esac
}

npm_cache_dir() {
  # 返回一个可写的 npm 缓存目录。默认位置可能因属主不对（例如曾被 root 或容器内
  # 其它 uid 创建过）而不可写，这时必须换一个位置，否则 npm 会直接报错退出。
  local candidate probe
  for candidate in \
    "${MSAGENT_NPM_CACHE:-}" \
    "${HOME}/.cache/msagent/npm-cache" \
    "${HOME}/.msagent/npm-cache" \
    "${TMPDIR:-/tmp}/msagent-npm-cache"; do
    [ -n "${candidate}" ] || continue
    mkdir -p "${candidate}" 2>/dev/null || continue
    probe="${candidate}/.msagent-write-test-$$"
    if { : > "${probe}"; } 2>/dev/null; then
      rm -f "${probe}" 2>/dev/null || true
      printf '%s' "${candidate}"
      return 0
    fi
    rm -f "${probe}" 2>/dev/null || true
  done
  return 1
}

npm_extra_registries() {
  # 额外的兜底 npm 源，供公司内网/离线环境注入，脚本本身不含任何内网地址：
  #   1) MSAGENT_NPM_REGISTRY_FALLBACKS（逗号/分号/空格分隔，可多个）
  #   2) 站点/用户配置文件（每行一个源，# 开头为注释）：
  #        $MSAGENT_NPM_MIRRORS_FILE > ~/.msagent/npm-mirrors > /etc/msagent/npm-mirrors
  # 这些源排在公网候选链之后：公网能装成功时完全不会被访问。
  local file line
  if [ -n "${MSAGENT_NPM_REGISTRY_FALLBACKS:-}" ]; then
    printf '%s\n' "${MSAGENT_NPM_REGISTRY_FALLBACKS}" | tr ',;' '\n' | tr -s '[:space:]' '\n'
  fi
  for file in "${MSAGENT_NPM_MIRRORS_FILE:-}" "${HOME}/.msagent/npm-mirrors" "/etc/msagent/npm-mirrors"; do
    [ -n "${file}" ] || continue
    [ -r "${file}" ] || continue
    while IFS= read -r line; do
      line="${line%%#*}"
      line="$(printf '%s' "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
      [ -n "${line}" ] && printf '%s\n' "${line}"
    done < "${file}"
  done
}

npm_registry_candidates() {
  # 依次尝试的 npm 源（按顺序、去重，每行一个）：
  #   显式指定 → 本机 npm 配置 → npmmirror → 华为云 → npmjs
  #   → 外部注入的兜底源 → 内置的内网兜底源（INTRANET_NPM_REGISTRY）
  #
  # 注意：显式指定只是"优先"，不是"只用"。公司内网源在外网不可达时，必须能
  # 自动回退到公网源；确实只想用指定源时，设置 MSAGENT_NPM_REGISTRY_ONLY=1。
  #
  # 兜底源放在最后有两个好处：内网机器（公网源都不通）能用上公司镜像，
  # 公网用户则因为前面的源已经成功而完全不会去访问它，不增加任何耗时。
  local candidate configured explicit listed=""
  explicit="${MSAGENT_NPM_REGISTRY:-}"
  configured="$(configured_npm_registry || true)"
  if [ -n "${explicit}" ] && [ "${MSAGENT_NPM_REGISTRY_ONLY:-0}" = "1" ]; then
    printf '%s\n' "${explicit}"
    return 0
  fi
  for candidate in \
    ${explicit:+"${explicit}"} \
    ${configured:+"${configured}"} \
    "${NPM_REGISTRY_CANDIDATES[@]}" \
    $(npm_extra_registries) \
    ${INTRANET_NPM_REGISTRY:+"${INTRANET_NPM_REGISTRY}"}; do
    [ -n "${candidate}" ] || continue
    case "|${listed}|" in
      *"|${candidate}|"*) continue ;;
    esac
    listed="${listed}${listed:+|}${candidate}"
    printf '%s\n' "${candidate}"
  done
}

npm_metadata_version() {
  # 取某个源上 @opencxd/ascend-doc-mcp 的最新版本号；同时充当"可达性探测"：
  # 强超时（连接 3s / 总长 6s），不可达的源会很快失败而不是拖住整个安装。
  # 作用域名里的 / 按 npm 自己的请求方式写成 %2F：它是包名的一部分，不是路径分隔符。
  local registry="$1" body
  if command -v curl >/dev/null 2>&1; then
    body="$(curl -fsSL --connect-timeout 3 --max-time 6 \
      "${registry%/}/@opencxd%2Fascend-doc-mcp/latest" 2>/dev/null || true)"
  elif command -v wget >/dev/null 2>&1; then
    body="$(wget -qO- --timeout=6 "${registry%/}/@opencxd%2Fascend-doc-mcp/latest" 2>/dev/null || true)"
  fi
  [ -n "${body}" ] || return 1
  printf '%s' "${body}" | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4
}

npm_probe_registries() {
  # 并行探测候选源，输出 "<源>\t<版本>"（每行一个，保持候选顺序，只输出可达的）。
  # 并行是重点：内网里前面的公网源会一直卡到超时，串行探测会非常慢。
  local candidates_file="$1" work_dir="$2" registry i=0
  while IFS= read -r registry; do
    [ -n "${registry}" ] || continue
    i=$((i + 1))
    (
      version="$(npm_metadata_version "${registry}" || true)"
      [ -n "${version}" ] && printf '%s\t%s\n' "${registry}" "${version}" > "${work_dir}/${i}.out"
    ) &
  done < "${candidates_file}"
  wait || true
  i=0
  while IFS= read -r registry; do
    [ -n "${registry}" ] || continue
    i=$((i + 1))
    [ -s "${work_dir}/${i}.out" ] && cat "${work_dir}/${i}.out"
  done < "${candidates_file}"
  return 0
}

install_user_node() {
  # Downloads the newest Node 22 LTS tarball from the given mirror base into
  # NODE_HOME. Returns 0 and sets NODE_BIN_DIR on success; the bin directory is
  # passed through a global (not stdout) so diagnostics can stay on stdout.
  # Mirror directories carry several patch releases in one of two formats, so
  # the candidates are ranked newest-first and tried in order: a single missing
  # or broken archive (404, truncated, xz unsupported) falls back to the next
  # one instead of failing the whole stage.
  local base="$1" platform listing names candidates best url tmp tarball node_root
  NODE_BIN_DIR=""
  platform="$(node_platform_name)" || return 1
  [ -n "${base}" ] || return 1
  listing="$(curl -fsSL --connect-timeout 8 --max-time 25 "${base}/latest-v22.x/" 2>/dev/null || true)"
  [ -n "${listing}" ] || return 1
  # Mirror indexes come as HTML (huaweicloud, nodejs.org) or JSON (npmmirror);
  # matching the archive name works for both.
  names="$(printf '%s\n' "${listing}" | grep -oE "node-v[0-9]+\.[0-9]+\.[0-9]+-${platform}\.tar\.(xz|gz)" || true)"
  [ -n "${names}" ] || return 1
  candidates="$(printf '%s\n' "${names}" | sort -u | while IFS= read -r name; do
    printf '%s\t%s\n' "$(archive_version "${name}")" "${name}"
  done | version_sort_desc)"
  [ -n "${candidates}" ] || return 1
  tmp="$(mktemp -d)"
  while IFS= read -r best; do
    [ -n "${best}" ] || continue
    url="${base}/latest-v22.x/${best}"
    NODE_TARBALL_URL="${url}"
    log_info "正在从 ${base} 下载 ${best} ..."
    tarball="${tmp}/${best}"
    if ! curl -fsSL --connect-timeout 10 --max-time 600 -o "${tarball}" "${url}"; then
      log_warn "下载失败：${best}（改用下一个可用版本）"
      rm -f "${tarball}"
      continue
    fi
    if ! tar -xf "${tarball}" -C "${tmp}"; then
      log_warn "解压失败：${best}（改用下一个可用版本）"
      rm -f "${tarball}"
      continue
    fi
    node_root="$(find "${tmp}" -maxdepth 1 -type d -name 'node-v*' | head -n 1)"
    if [ -z "${node_root}" ]; then
      log_warn "归档内容异常：${best}（改用下一个可用版本）"
      rm -f "${tarball}"
      continue
    fi
    rm -rf "${NODE_HOME}"
    mkdir -p "${NODE_HOME}"
    mv "${node_root}"/* "${NODE_HOME}/"
    rm -rf "${tmp}"
    [ -x "${NODE_HOME}/bin/node" ] || return 1
    NODE_BIN_DIR="${NODE_HOME}/bin"
    return 0
  done <<EOF
${candidates}
EOF
  rm -rf "${tmp}"
  return 1
}

prepare_ascend_doc_mcp() {
  local node_bin_dir registry prefix cache node_mirror node_major node_version npm_log npm_ok
  local package_version package_spec attempts_file candidate npm_last_failed
  local candidates_file probe_dir probe_file configured
  if [ "${MSAGENT_NO_ASCEND_DOC_MCP:-0}" = "1" ]; then
    log_info "已跳过文档查询服务准备（MSAGENT_NO_ASCEND_DOC_MCP=1）。"
    SUMMARY_DOC_MCP="已跳过（MSAGENT_NO_ASCEND_DOC_MCP=1）"
    return 0
  fi
  NODE_HOME="${MSAGENT_NODE_HOME:-${HOME}/.msagent/node}"
  node_bin_dir=""
  node_version=""
  if [ -x "${NODE_HOME}/bin/node" ]; then
    node_bin_dir="${NODE_HOME}/bin"
    node_version="$("${node_bin_dir}/node" --version 2>/dev/null || true)"
  elif command -v node >/dev/null 2>&1; then
    node_bin_dir="$(dirname "$(command -v node)")"
    node_version="$("${node_bin_dir}/node" --version 2>/dev/null || true)"
    node_major="$(node_major_version "${node_bin_dir}/node")"
    if [ -n "${node_major}" ] && [ "${node_major}" -lt "${NODE_MIN_MAJOR}" ]; then
      log_warn "检测到 Node ${node_version}，但本功能需要 >= ${NODE_MIN_MAJOR}，将安装用户级 Node。"
      node_bin_dir=""
      node_version=""
    fi
  fi
  if [ -z "${node_bin_dir}" ]; then
    node_mirror="$(select_node_mirror)"
    log_info "正在从 ${node_mirror} 下载用户级 Node ${NODE_MIN_MAJOR} LTS ..."
    if ! install_user_node "${node_mirror}"; then
      log_warn "无法从 ${node_mirror} 获取 Node.js。"
      if [ -n "${NODE_TARBALL_URL:-}" ]; then
        log_warn "可用浏览器/下载工具手动获取该压缩包：${NODE_TARBALL_URL}"
      fi
      return 1
    fi
    node_bin_dir="${NODE_BIN_DIR}"
    node_version="$("${node_bin_dir}/node" --version 2>/dev/null || true)"
    log_success "已安装用户级 Node ${node_version} 到 ${NODE_HOME}。"
    log_info "如需在自己的终端中使用：export PATH=\"${NODE_HOME}/bin:\$PATH\""
  else
    log_info "使用已有 Node ${node_version}（${node_bin_dir}）。"
  fi
  export PATH="${node_bin_dir}:${PATH}"

  if ! command -v npm >/dev/null 2>&1 || ! command -v npx >/dev/null 2>&1; then
    log_warn "${node_bin_dir} 下没有 npm/npx，跳过文档查询服务的预安装。"
    return 1
  fi

  # 先并行探测所有候选源（每个 6 秒超时），只在**可达的源**上真正安装：
  # 内网里前面的公网源会一直卡到超时，逐个 npm install 试过去非常慢；探测则
  # 最多等一个超时周期就能定位到可用源。探测请求本身就是包元数据请求，顺带
  # 拿到要安装的版本号。
  candidates_file="$(mktemp)"
  npm_registry_candidates > "${candidates_file}"
  probe_dir="$(mktemp -d)"
  probe_file="$(mktemp)"
  log_info "正在探测 npm 源（$(grep -c . "${candidates_file}" 2>/dev/null || echo 0) 个，并行、每个最多 6 秒）..."
  npm_probe_registries "${candidates_file}" "${probe_dir}" > "${probe_file}"

  attempts_file="$(mktemp)"
  if [ -s "${probe_file}" ]; then
    cut -f1 "${probe_file}" > "${attempts_file}"
    registry="$(head -n 1 "${probe_file}" | cut -f1)"
    package_version="$(head -n 1 "${probe_file}" | cut -f2)"
    log_info "可达的 npm 源：$(cut -f1 "${probe_file}" | tr '\n' ' ')"
    configured="$(configured_npm_registry || true)"
    if [ -n "${configured}" ] && [ "${registry}" = "${configured}" ]; then
      log_info "使用本机 npm 已配置的源：${registry}"
    fi
  else
    # 探测全部失败（例如源只提供包内容、不响应元数据请求）：退回逐个尝试。
    log_warn "所有 npm 源都没有响应元数据请求，改为按顺序逐个尝试安装。"
    cp "${candidates_file}" "${attempts_file}"
    registry="$(head -n 1 "${candidates_file}")"
    package_version=""
  fi
  rm -rf "${probe_dir}"
  rm -f "${probe_file}" "${candidates_file}"

  package_spec="@opencxd/ascend-doc-mcp@latest"
  if [ -n "${package_version}" ]; then
    package_spec="@opencxd/ascend-doc-mcp@${package_version}"
  fi
  prefix="${MSAGENT_ASCEND_DOC_MCP_PREFIX:-${HOME}/.msagent/ascend-doc-mcp}"
  mkdir -p "${prefix}"
  # npm 会因为缓存目录不可写而整体失败（常见于该目录被 root 或其它 uid 创建过），
  # 所以先探测可写性，不可写就换一个位置，而不是白跑完所有候选源。
  cache="$(npm_cache_dir || true)"
  default_cache="${MSAGENT_NPM_CACHE:-${HOME}/.cache/msagent/npm-cache}"
  if [ -z "${cache}" ]; then
    cache="${TMPDIR:-/tmp}/msagent-npm-cache"
    mkdir -p "${cache}" 2>/dev/null || true
    log_warn "没有可写的 npm 缓存目录，改用 ${cache}。"
  elif [ "${cache}" != "${default_cache}" ]; then
    log_warn "npm 缓存目录不可写（${default_cache}），已改用 ${cache}。"
    log_warn "如需修复原目录属主：sudo chown -R \"\$(id -u):\$(id -g)\" \"${default_cache}\""
  fi

  # 只在探测可达的源上安装（正常情况下就是 1 个）；若探测阶段一个都没通，
  # attempts_file 里是所有候选源，沿用"逐个回退"的老行为。
  npm_log="$(mktemp)"
  npm_ok=0
  npm_last_failed=""
  npm_net_args="$(npm_proxy_args)"
  npm_extra=()
  if [ -n "${npm_net_args}" ]; then
    # shellcheck disable=SC2207
    npm_extra=($(printf '%s' "${npm_net_args}"))
    log_info "npm 使用环境变量指定的代理：${HTTPS_PROXY:-${https_proxy:-${HTTP_PROXY:-${http_proxy:-${ALL_PROXY:-${all_proxy}}}}}}"
  fi
  while IFS= read -r candidate; do
    [ -n "${candidate}" ] || continue
    log_info "正在预安装 ${package_spec}（源：${candidate}）..."
    : > "${npm_log}"
    if npm install --prefix "${prefix}" --registry "${candidate}" --cache "${cache}" \
        ${npm_extra[@]+"${npm_extra[@]}"} \
        --fetch-retries=1 --no-audit --no-fund --no-package-lock "${package_spec}" >"${npm_log}" 2>&1 \
       && [ -f "${prefix}/node_modules/@opencxd/ascend-doc-mcp/package.json" ]; then
      npm_ok=1
      registry="${candidate}"
      break
    fi
    npm_last_failed="${candidate}"
    log_warn "在 ${candidate} 上预安装失败，继续尝试下一个源。"
  done < "${attempts_file}"
  rm -f "${attempts_file}"

  if [ "${npm_ok}" = "0" ]; then
    # 某些环境（如 WSL1/容器）npm 的 DNS/IPv6 解析会失败而 curl 正常，最后再强制
    # IPv4 优先重试一次；这条也常能解决 ETIMEDOUT。
    log_info "改用 IPv4 优先重试一次（源：${registry}）..."
    : > "${npm_log}"
    if NODE_OPTIONS="${NODE_OPTIONS:+${NODE_OPTIONS} }--dns-result-order=ipv4first" \
        npm install --prefix "${prefix}" --registry "${registry}" --cache "${cache}" \
          ${npm_extra[@]+"${npm_extra[@]}"} \
          --fetch-retries=1 --no-audit --no-fund --no-package-lock "${package_spec}" >"${npm_log}" 2>&1 \
       && [ -f "${prefix}/node_modules/@opencxd/ascend-doc-mcp/package.json" ]; then
      npm_ok=1
    fi
  fi

  if [ "${npm_ok}" = "0" ]; then
    log_warn "所有候选源均无法预安装文档查询服务（最后一次：${npm_last_failed}）。"
    # 只打印真正的报错：npm 的"日志写不进去 / 加 --loglevel=verbose / sudo chown"
    # 提示会把真正的 network/证书类错误淹没掉。
    if [ -s "${npm_log}" ]; then
      grep -E 'npm (error|ERR!|warn)' "${npm_log}" 2>/dev/null \
        | grep -v -E 'Log files were not written|loglevel=verbose|sudo chown|A complete log' \
        | head -3 | while IFS= read -r line; do
          log_warn "  npm：${line}"
        done || true
    fi
    rm -f "${npm_log}"
    log_warn "内网环境请把 npm 源指向内网镜像（任选其一），然后重跑本脚本："
    log_warn "  npm config set registry <内网 npm 源>        # 永久生效，推荐"
    log_warn "  export MSAGENT_NPM_REGISTRY=<内网 npm 源>   # 仅本次；Windows 用 setx 持久化"
    log_warn "  运维也可放置 ~/.msagent/npm-mirrors 或 /etc/msagent/npm-mirrors（每行一个源）"
    log_warn "如需走代理请设置 HTTPS_PROXY，或 npm config set proxy/https-proxy <代理地址>。"
    log_warn "运行时会按需重试，它同样遵循 MSAGENT_NPM_REGISTRY。"
    return 1
  fi
  rm -f "${npm_log}"
  log_success "文档查询服务已预安装到 ${prefix}。"
  NODE_VERSION_RESOLVED="${node_version}"
  MCP_PACKAGE_VERSION_RESOLVED="${package_version:-latest}"
  NPM_REGISTRY_USED="${registry}"
}

# ---------------------------------------------------------------------------
# Install mindstudio-agent as an isolated uv tool (target version resolved above)
# ---------------------------------------------------------------------------
phase "安装 msagent"
# 不静默降级：本地可能装着更新的内测/自编译版本，源上反而更旧（例如源同步滞后
# 或本地装了 26.1.3 的本地 wheel）。此时保留现有版本并给出强制降级的方法。
SKIP_INSTALL=0
if [ -n "${LATEST_VERSION}" ] && [ -n "${SUMMARY_PREV_VERSION}" ] && [ -z "${MSAGENT_VERSION:-}" ] \
   && version_gt "${SUMMARY_PREV_VERSION}" "${LATEST_VERSION}"; then
  log_warn "已安装的 msagent ${SUMMARY_PREV_VERSION} 高于源上的 ${LATEST_VERSION}，跳过安装以免降级。"
  log_warn "如需强制安装 ${LATEST_VERSION}：MSAGENT_VERSION=${LATEST_VERSION} 重新运行。"
  SKIP_INSTALL=1
elif command -v msagent >/dev/null 2>&1; then
  log_info "检测到已安装，正在更新 msagent ..."
else
  log_info "正在把 ${INSTALL_SPEC} 安装到独立的 uv 工具环境 ..."
fi

UV_TOOL_ARGS=(tool install -U --python "${MSAGENT_PYTHON}" --default-index "${INDEX}")
if [ -n "${MSAGENT_WITH_EXECUTABLES_FROM:-}" ]; then
  UV_TOOL_ARGS+=(--with-executables-from "${MSAGENT_WITH_EXECUTABLES_FROM}")
fi
UV_TOOL_ARGS+=("${INSTALL_SPEC}")

INSTALL_OK=0
if [ "${SKIP_INSTALL}" = "1" ]; then
  INSTALL_OK=1
elif "${UV_BIN}" "${UV_TOOL_ARGS[@]}"; then
  INSTALL_OK=1
fi

# Mirror sync may lag the latest weekly release; retry with official PyPI.
if [ "${INSTALL_OK}" = "0" ] && [ -z "${MSAGENT_INDEX:-}" ] && [ "${INDEX}" != "https://pypi.org/simple" ]; then
  log_warn "使用 ${INDEX} 安装失败，改用 PyPI 官方源重试一次（镜像可能尚未同步）..."
  UV_TOOL_ARGS=(tool install -U --python "${MSAGENT_PYTHON}" --default-index "https://pypi.org/simple")
  if [ -n "${MSAGENT_WITH_EXECUTABLES_FROM:-}" ]; then
    UV_TOOL_ARGS+=(--with-executables-from "${MSAGENT_WITH_EXECUTABLES_FROM}")
  fi
  UV_TOOL_ARGS+=("${INSTALL_SPEC}")
  if "${UV_BIN}" "${UV_TOOL_ARGS[@]}"; then
    INSTALL_OK=1
  fi
fi

if [ "${INSTALL_OK}" = "0" ]; then
  # Retry once with a freshly upgraded uv: a pre-existing uv may be old or in
  # a broken state (e.g. failing managed-Python resolution on Windows).
  log_warn "首次安装失败，正在升级 uv 并重试一次..."
  RETRY_PY=""
  if command -v python3 >/dev/null 2>&1; then RETRY_PY="python3"
  elif command -v python >/dev/null 2>&1; then RETRY_PY="python"
  fi
  if [ -n "${RETRY_PY}" ]; then
    "${RETRY_PY}" -m pip install --user -q -U uv -i "${INDEX}" 2>/dev/null || \
      "${RETRY_PY}" -m pip install --user -q -U --break-system-packages uv -i "${INDEX}" 2>/dev/null || true
  fi
  [ -x "${HOME}/.local/bin/uv" ] && UV_BIN="${HOME}/.local/bin/uv"
  if "${UV_BIN}" "${UV_TOOL_ARGS[@]}"; then
    INSTALL_OK=1
  fi
fi

if [ "${INSTALL_OK}" = "0" ]; then
  # The tool may exist in a broken/stale state (e.g. its Python interpreter
  # no longer matches). Remove it and retry the install from scratch once.
  log_warn "安装仍失败，正在清理已有 msagent 工具并再试一次..."
  "${UV_BIN}" tool uninstall mindstudio-agent >/dev/null 2>&1 || true
  if "${UV_BIN}" "${UV_TOOL_ARGS[@]}"; then
    INSTALL_OK=1
  fi
fi

if [ "${INSTALL_OK}" = "0" ]; then
  log_error "uv 工具安装失败，请查看上面的错误信息。"
  if [ "${MSAGENT_NO_FALLBACK:-0}" = "1" ]; then
    log_error "已设置 MSAGENT_NO_FALLBACK，跳过 venv 兜底安装。"
    log_error "请先解决问题，再用推荐方式重试："
    log_error "  ${UV_BIN} tool install -U --python \"${MSAGENT_PYTHON}\" --default-index \"${INDEX}\" \"${INSTALL_SPEC}\""
    log_error "如需手动兜底安装："
    log_error "  python3 -m venv ~/.msagent-venv && ~/.msagent-venv/bin/pip install -U -i ${INDEX} ${INSTALL_SPEC}"
    exit 1
  fi
  if prompt_yn "Try the isolated venv fallback instead?"; then
    if ! install_venv_fallback; then
      log_error "venv 兜底安装也失败了。请稍后重试，或在以下地址反馈问题："
      log_error "  https://gitcode.com/Ascend/msagent/issues"
      exit 1
    fi
  else
    log_error "安装已中止，可稍后用以下命令重试："
    log_error "  curl -LsSf https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.sh | bash"
    exit 1
  fi
else
  # -------------------------------------------------------------------------
  # PATH: make the tool bin directory available in new shells.
  # -------------------------------------------------------------------------
  phase "配置 PATH 环境变量"
  setup_path
  SUMMARY_TOOL_BIN="${TOOL_BIN_DIR}"
fi

# Ensure msprof-mcp / msprof-analyze are reachable from PATH (best effort).
expose_tool_executables

# Optional: provision Node and pre-install ascend-doc-mcp so the docs MCP works
# from the first run (domestic npm/node mirrors; best-effort unless required).
phase "准备昇腾文档查询服务（Node.js + ascend-doc-mcp）"
if prepare_ascend_doc_mcp; then
  SUMMARY_NODE="${NODE_VERSION_RESOLVED}"
  if [ -n "${NPM_REGISTRY_USED}" ]; then
    SUMMARY_DOC_MCP="v${MCP_PACKAGE_VERSION_RESOLVED:-latest} at ${MSAGENT_ASCEND_DOC_MCP_PREFIX:-${HOME}/.msagent/ascend-doc-mcp}"
  fi
else
  if [ "${MSAGENT_ASCEND_DOC_MCP_REQUIRE:-0}" = "1" ]; then
    log_error "文档查询服务准备失败，且已设置 MSAGENT_ASCEND_DOC_MCP_REQUIRE=1。"
    exit 1
  fi
  log_warn "文档查询服务未就绪（不影响 msagent 本体使用，仅 ascend-knowledge 的资料查询不可用）。"
  ascend_doc_mcp_guidance
  SUMMARY_DOC_MCP="不可用（可选功能）"
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
phase "验证安装结果"
MSAGENT_BIN=""
if [ -x "${TOOL_BIN_DIR}/msagent" ]; then
  MSAGENT_BIN="${TOOL_BIN_DIR}/msagent"
elif command -v msagent >/dev/null 2>&1; then
  MSAGENT_BIN="msagent"
fi

VERSION_OUTPUT=""
VERSION_TOKEN=""
if [ -n "${MSAGENT_BIN}" ]; then
  VERSION_OUTPUT="$("${MSAGENT_BIN}" --version 2>&1)" || {
    log_error "msagent 已安装，但 '--version' 执行失败："
    printf '%s\n' "${VERSION_OUTPUT}" >&2
    log_error "可尝试 source ~/.local/bin/env，或重开终端后重试。"
    exit 1
  }
  VERSION_TOKEN="$(printf '%s\n' "${VERSION_OUTPUT}" | msagent_version_token || true)"
  log_success "$(bold "验证通过：msagent ${VERSION_TOKEN:-已安装}")"
fi

# 绝对路径能跑通不代表当前 shell 已加载 PATH（例如 `curl ... | bash`），
# 这里只提示一次最简操作。
PATH_MSAGENT="$(command -v msagent 2>/dev/null || true)"
if [ -z "${PATH_MSAGENT}" ]; then
  log_warn "msagent 已安装，但当前终端的 PATH 尚未生效：执行 source ${PROFILE_FILE:-~/.bashrc}（或重开终端）。"
  log_warn "详见下方\"后续步骤\"第 3 步。"
elif [ "${PATH_MSAGENT}" != "${TOOL_BIN_DIR}/msagent" ] && [ "${PATH_MSAGENT}" != "${TOOL_BIN_DIR}/msagent.exe" ]; then
  log_warn "注意：PATH 中的 'msagent' 实际指向 ${PATH_MSAGENT}（可能是旧版 pip 安装）。"
  log_warn "uv 工具安装位置为 ${TOOL_BIN_DIR}/msagent，建议执行：pip uninstall mindstudio-agent"
fi

MSPROF_BIN=""
if [ -x "${TOOL_BIN_DIR}/msprof-mcp" ]; then
  MSPROF_BIN="${TOOL_BIN_DIR}/msprof-mcp"
elif command -v msprof-mcp >/dev/null 2>&1; then
  MSPROF_BIN="msprof-mcp"
fi
if [ -z "${MSPROF_BIN}" ]; then
  log_warn "当前 shell 的 PATH 中还没有 msprof-mcp；若 msagent 报缺失请重开终端。"
  SUMMARY_MSPROF="尚未在 PATH 中"
else
  log_success "msprof-mcp 可执行文件可用。"
  SUMMARY_MSPROF="正常"
fi

log_success "$(bold "msagent 安装完成。")"
if [ -n "${VERSION_TOKEN}" ] && [ -n "${SUMMARY_PREV_VERSION}" ] && [ "${SUMMARY_PREV_VERSION}" = "${VERSION_TOKEN}" ]; then
  SUMMARY_VERSION="${VERSION_TOKEN}（已是最新）"
elif [ -n "${VERSION_TOKEN}" ]; then
  SUMMARY_VERSION="${VERSION_TOKEN}"
elif [ -n "${VERSION_OUTPUT}" ]; then
  SUMMARY_VERSION="已安装（版本号未识别）"
else
  SUMMARY_VERSION="已安装（当前 shell 未加载 PATH）"
fi
SUMMARY_NPM_REGISTRY="${NPM_REGISTRY_USED:-}"
if [ -z "${SUMMARY_TOOL_BIN}" ]; then
  SUMMARY_TOOL_BIN="${TOOL_BIN_DIR}"
fi
print_summary
print_next_steps
