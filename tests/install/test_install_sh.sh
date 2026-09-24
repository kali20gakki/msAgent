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
# Functional tests for scripts/install.sh using a mock uv (fake_uv.sh).
#
# These tests guard the installer's decision logic (index priority, version
# pinning/announcement, PyPI fallback retry, NO_MODIFY_PATH escape hatch,
# static syntax) without performing a real package install. A real-install
# smoke test lives in smoke_install.sh.
#
# Usage:
#   bash tests/install/test_install_sh.sh
# Exit code 0 = all checks passed.

set -euo pipefail

# The installer itself refuses MSYS/MinGW/CYGWIN shells, so this suite must
# run on Linux/macOS/WSL (CI uses ubuntu-latest).
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*)
    echo "SKIP: install.sh refuses MSYS/MinGW/CYGWIN shells; run on Linux/macOS/WSL"
    exit 0
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
INSTALLER="${REPO_ROOT}/scripts/install.sh"
FAKE_UV="${SCRIPT_DIR}/fake_uv.sh"

PASS=0
FAIL=0
SKIP=0
ORIGINAL_PATH="${PATH}"

# PATH without any node installation, so "no Node available" scenarios are
# deterministic even on CI images that ship Node.
node_free_path() {
  local out="" entry
  local IFS=':'
  for entry in ${PATH}; do
    case "${entry}" in
      *node*|*nvm*) continue ;;
    esac
    out="${out:+${out}:}${entry}"
  done
  [ -n "${out}" ] || out="/usr/bin:/bin"
  printf '%s' "${out}"
}

ok() { printf '  [PASS] %s\n' "$*"; PASS=$((PASS + 1)); }
ko() { printf '  [FAIL] %s\n' "$*" >&2; FAIL=$((FAIL + 1)); }
skip() { printf '  [SKIP] %s\n' "$*"; SKIP=$((SKIP + 1)); }

# new_test_env  -- fresh isolated HOME/bin/log for one scenario. Call it before
# dropping fakes into ${TEST_BIN}; run_installer creates it on demand otherwise.
new_test_env() {
  [ -n "${TEST_HOME:-}" ] && rm -rf "${TEST_HOME}"
  TEST_HOME="$(mktemp -d)"
  TEST_BIN="${TEST_HOME}/bin"
  mkdir -p "${TEST_BIN}"
  UV_LOG="$(mktemp)"
}

# run_installer <out-log> [extra env assignments...]  -- runs the installer in
# an isolated HOME with a mock uv; the last stdout line of the fake uv's
# `tool dir --bin` call becomes the tool bin dir. Test-local executables dropped
# into ${TEST_BIN} (e.g. a fake curl/node/npm) shadow the real ones. The
# installer's exit code is left in ${INSTALL_RC}.
run_installer() {
  local out_log="$1"
  shift
  [ -n "${TEST_HOME:-}" ] || new_test_env
  chmod +x "${FAKE_UV}" 2>/dev/null || true
  export HOME="${TEST_HOME}"
  export MSAGENT_YES=1
  export MSAGENT_NO_MODIFY_PATH=1
  export MSAGENT_NO_ASCEND_DOC_MCP=1
  export UV_BIN="${FAKE_UV}"
  export MSAGENT_TEST_UV_LOG="${UV_LOG}"
  export MSAGENT_TEST_TOOL_BIN="${TEST_BIN}"
  local kv
  for kv in "$@"; do export "${kv}"; done
  # Prepend after the caller's overrides so a scenario can pass its own PATH
  # (e.g. a node-free one) and still reach the fakes in ${TEST_BIN}.
  export PATH="${TEST_BIN}:${PATH}"
  set +e
  bash "${INSTALLER}" > "${out_log}" 2>&1
  INSTALL_RC=$?
  set -e
}

# write_fake_tool <name> <body...>  -- installs an executable stub in ${TEST_BIN}.
write_fake_tool() {
  local name="$1"
  shift
  {
    printf '#!/usr/bin/env bash\n'
    printf '%s\n' "$@"
  } > "${TEST_BIN}/${name}"
  chmod +x "${TEST_BIN}/${name}"
}

# Fake curl for the offline intranet scenarios: the public npm/Node hosts are
# blocked, the Huawei Cloud mirror answers. Archive downloads always fail so no
# real bytes are fetched. -o is honored, so probe output never reaches stdout.
#   $1  an extra case pattern (e.g. '*npm.internal.example*') that makes one
#       more host reachable
#   $2  the version the package tag endpoint serves; empty = tag unreadable
write_fake_curl() {
  local extra_pattern="${1:-}" tag_version="${2:-}"
  local -a lines=(
    'set -u'
    'url=""; out=""'
    'while [ "$#" -gt 0 ]; do'
    '  case "$1" in'
    '    -o) out="${2:-}"; shift 2 ;;'
    '    http*) url="$1"; shift ;;'
    '    *) shift ;;'
    '  esac'
    'done'
    'emit() { if [ -n "${out}" ] && [ "${out}" != "/dev/null" ]; then printf "%s" "$1" > "${out}"; else printf "%s" "$1"; fi; }'
    'case "${url}" in'
  )
  if [ -n "${tag_version}" ]; then
    lines+=("  */@opencxd%2Fascend-doc-mcp/latest) emit '{\"name\":\"@opencxd/ascend-doc-mcp\",\"version\":\"${tag_version}\"}'; exit 0 ;;")
  else
    lines+=('  */@opencxd%2Fascend-doc-mcp/latest) exit 7 ;;')
  fi
  if [ -n "${extra_pattern}" ]; then
    lines+=("  ${extra_pattern}) exit 0 ;;")
  fi
  lines+=(
    '  *registry.npmmirror.com*|*registry.npmjs.org*|*nodejs.org*|*aliyun.com*|*tuna.tsinghua.edu.cn*|*astral.sh*) exit 7 ;;'
    '  *mirrors.huaweicloud.com/nodejs/latest-v22.x/*)'
    '    emit "<a href=\"node-v22.99.0-linux-x64.tar.xz\">node-v22.99.0-linux-x64.tar.xz</a>"'
    '    exit 0 ;;'
    '  *mirrors.huaweicloud.com/nodejs/node-v*) exit 7 ;;'
    '  *mirrors.huaweicloud.com*) exit 0 ;;'
    '  *) exit 7 ;;'
    'esac'
  )
  write_fake_tool curl "${lines[@]}"
}

# Fake npm: reports the given registry for `config get registry` and records
# every invocation in ${HOME}/npm-args.log, so a scenario can assert which
# registry the real install would have used.
write_fake_npm() {
  local registry="$1"
  write_fake_tool npm \
    "if [ \"\${1:-}\" = \"config\" ] && [ \"\${2:-}\" = \"get\" ]; then echo \"${registry}\"; exit 0; fi" \
    'printf "%s\n" "$*" >> "${HOME}/npm-args.log"' \
    'prefix=""' \
    'while [ "$#" -gt 0 ]; do case "$1" in --prefix) prefix="$2"; shift 2 ;; *) shift ;; esac; done' \
    '[ -n "${prefix}" ] || exit 1' \
    'mkdir -p "${prefix}/node_modules/@opencxd/ascend-doc-mcp"' \
    ': > "${prefix}/node_modules/@opencxd/ascend-doc-mcp/package.json"' \
    'exit 0'
}

# Fake npm that fails for one host (as an intranet mirror does off-site) and
# succeeds everywhere else. Its own `config get registry` reports the public
# default so the failing host can only come from an explicit MSAGENT_NPM_REGISTRY.
write_fake_npm_failing_host() {
  local host="$1"
  write_fake_tool npm \
    'if [ "${1:-}" = "config" ] && [ "${2:-}" = "get" ]; then echo "https://registry.npmjs.org/"; exit 0; fi' \
    'printf "%s\n" "$*" >> "${HOME}/npm-args.log"' \
    "case \"\$*\" in" \
    "  *${host}*) echo \"npm error network request to ${host} failed\" >&2; exit 1 ;;" \
    'esac' \
    'prefix=""' \
    'while [ "$#" -gt 0 ]; do case "$1" in --prefix) prefix="$2"; shift 2 ;; *) shift ;; esac; done' \
    '[ -n "${prefix}" ] || exit 1' \
    'mkdir -p "${prefix}/node_modules/@opencxd/ascend-doc-mcp"' \
    ': > "${prefix}/node_modules/@opencxd/ascend-doc-mcp/package.json"' \
    'exit 0'
}

# Fake Node 22 runtime, so the stage reuses it instead of downloading.
write_fake_node() {
  local version="${1:-v22.0.0}"
  write_fake_tool node "case \"\${1:-}\" in --version) echo ${version} ;; esac" 'exit 0'
  write_fake_tool npx 'exit 0'
}

# make_fake_node_tarball <dest> <dir> <version>  -- real .tar.gz holding
# <dir>/bin/node, so the installer's `tar -xf` and the bin/node check succeed.
make_fake_node_tarball() {
  local dest="$1" dir="$2" version="$3" staging
  staging="$(mktemp -d)"
  mkdir -p "${staging}/${dir}/bin"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'case "${1:-}" in --version) echo %s ;; esac\n' "${version}"
    printf 'exit 0\n'
  } > "${staging}/${dir}/bin/node"
  chmod +x "${staging}/${dir}/bin/node"
  ( cd "${staging}" && tar -czf "${dest}" "${dir}" )
  rm -rf "${staging}"
}

cleanup_env() {
  rm -rf "${TEST_HOME:-}" "${UV_LOG:-}"
  unset TEST_HOME TEST_BIN UV_LOG INSTALL_RC 2>/dev/null || true
  export PATH="${ORIGINAL_PATH}"
  unset MSAGENT_VERSION MSAGENT_INDEX MSAGENT_TEST_UV_FAIL MSAGENT_TEST_UV_FAIL_PYPI_ONLY \
    MSAGENT_NODE_MIRROR MSAGENT_NPM_REGISTRY MSAGENT_NPM_REGISTRY_ONLY MSAGENT_NODE_HOME \
    MSAGENT_NPM_REGISTRY_FALLBACKS MSAGENT_NPM_MIRRORS_FILE MSAGENT_NO_ASCEND_DOC_MCP 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# 0. Helper unit checks (functions extracted from the installer)
# ---------------------------------------------------------------------------
printf '== Helper checks ==\n'
VT_FILE="$(mktemp)"
sed -n '/^msagent_version_token()/,/^}/p' "${INSTALLER}" > "${VT_FILE}"
TOKEN="$(printf '\033[1;36mmsagent\033[0m 26.1.2 (unknown)\n' | bash -c ". '${VT_FILE}'; msagent_version_token")"
[ "${TOKEN}" = "26.1.2" ] && ok "version token survives ANSI colour" || ko "version token survives ANSI colour (got '${TOKEN}')"
TOKEN="$(printf 'msagent 26.1.2 (unknown)\n' | bash -c ". '${VT_FILE}'; msagent_version_token")"
[ "${TOKEN}" = "26.1.2" ] && ok "version token from a plain banner" || ko "version token from a plain banner (got '${TOKEN}')"
TOKEN="$(printf '  Repo : https://gitcode.com/Ascend/msagent\n' | bash -c ". '${VT_FILE}'; msagent_version_token")"
[ -z "${TOKEN}" ] && ok "ignores the repo URL line" || ko "ignores the repo URL line (got '${TOKEN}')"
rm -f "${VT_FILE}"
VG_FILE="$(mktemp)"
sed -n '/^version_gt()/,/^}/p' "${INSTALLER}" > "${VG_FILE}"
if bash -c ". '${VG_FILE}'; version_gt 26.1.3 26.1.2"; then
  ok "version_gt detects a newer version"
else
  ko "version_gt detects a newer version"
fi
if bash -c ". '${VG_FILE}'; version_gt 26.1.2 26.1.2"; then
  ko "version_gt treats equal versions as not newer"
else
  ok "version_gt treats equal versions as not newer"
fi
if bash -c ". '${VG_FILE}'; version_gt 26.1.2 26.1.10"; then
  ko "version_gt is numeric, not lexical"
else
  ok "version_gt is numeric, not lexical"
fi
rm -f "${VG_FILE}"

# Version ranking must be numeric (not lexicographic) and portable: macOS/BSD
# sort has no -V, and a mirror directory lists several patch releases at once,
# so "last text line" is not "newest version".
VSD_FILE="$(mktemp)"
sed -n '/^version_sort_desc()/,/^}/p' "${INSTALLER}" > "${VSD_FILE}"
AV_FILE="$(mktemp)"
sed -n '/^archive_version()/,/^}/p' "${INSTALLER}" > "${AV_FILE}"
SORTED="$(printf '22.9.0\tnode-v22.9.0-linux-x64.tar.gz\n22.20.0\tnode-v22.20.0-linux-x64.tar.xz\n22.0.0\tnode-v22.0.0-linux-x64.tar.gz\n22.10.0\tnode-v22.10.0-linux-x64.tar.xz\n' \
  | bash -c ". '${VSD_FILE}'; version_sort_desc")"
EXPECTED_SORTED="node-v22.20.0-linux-x64.tar.xz
node-v22.10.0-linux-x64.tar.xz
node-v22.9.0-linux-x64.tar.gz
node-v22.0.0-linux-x64.tar.gz"
[ "${SORTED}" = "${EXPECTED_SORTED}" ] \
  && ok "ranks Node archives numerically, newest first" \
  || ko "ranks Node archives numerically, newest first (got '${SORTED//$'\n'/, }')"
RANKED_FIRST="$(printf '26.1.0a1\t26.1.0a1\n26.1.0\t26.1.0\n26.1.0a2\t26.1.0a2\n' \
  | bash -c ". '${VSD_FILE}'; version_sort_desc" | head -n 1)"
[ "${RANKED_FIRST}" = "26.1.0" ] \
  && ok "ranks a final release above its own pre-releases" \
  || ko "ranks a final release above its own pre-releases (got '${RANKED_FIRST}')"
RANKED_ALPHA="$(printf '26.1.0a1\t26.1.0a1\n26.1.0a2\t26.1.0a2\n26.1.0a10\t26.1.0a10\n' \
  | bash -c ". '${VSD_FILE}'; version_sort_desc" | head -n 1)"
[ "${RANKED_ALPHA}" = "26.1.0a10" ] \
  && ok "ranks pre-release counters numerically" \
  || ko "ranks pre-release counters numerically (got '${RANKED_ALPHA}')"
ARCHIVE_VER="$(bash -c ". '${AV_FILE}'; archive_version node-v22.23.2-linux-x64.tar.xz")"
[ "${ARCHIVE_VER}" = "22.23.2" ] \
  && ok "extracts the version from a Node archive name" \
  || ko "extracts the version from a Node archive name (got '${ARCHIVE_VER}')"
rm -f "${VSD_FILE}" "${AV_FILE}"
if grep -qE '\|[[:space:]]*sort -V' "${INSTALLER}"; then
  ko "no pipeline depends on GNU sort -V"
else
  ok "no pipeline depends on GNU sort -V"
fi

# ---------------------------------------------------------------------------
# 1. Static checks
# ---------------------------------------------------------------------------
printf '== Static checks ==\n'
if bash -n "${INSTALLER}"; then ok "bash -n syntax"; else ko "bash -n syntax"; fi
if command -v shellcheck >/dev/null 2>&1; then
  if shellcheck -S warning "${INSTALLER}"; then ok "shellcheck (warning+)"; else ko "shellcheck (warning+)"; fi
else
  skip "shellcheck not installed"
fi

# ---------------------------------------------------------------------------
# 2. Deterministic logic: MSAGENT_VERSION pin + MSAGENT_INDEX passthrough
# ---------------------------------------------------------------------------
printf '\n== Scenario: MSAGENT_VERSION + MSAGENT_INDEX passthrough ==\n'
OUT="$(mktemp)"
run_installer "${OUT}" MSAGENT_VERSION=1.2.3 MSAGENT_INDEX=https://example.invalid/simple

grep -q "即将安装 msagent 1.2.3（由 MSAGENT_VERSION 指定）" "${OUT}" && ok "announces pinned version at start" || ko "announces pinned version at start"
grep -q -- "--default-index https://example.invalid/simple" "${UV_LOG}" && ok "passes MSAGENT_INDEX through" || ko "passes MSAGENT_INDEX through"
grep -q "mindstudio-agent==1.2.3" "${UV_LOG}" && ok "uses pinned spec" || ko "uses pinned spec"
grep -q -- "--python >=3.11" "${UV_LOG}" && ok "uses >=3.11 python request" || ko "uses >=3.11 python request"
grep -q "已设置 MSAGENT_NO_MODIFY_PATH" "${OUT}" && ok "honors MSAGENT_NO_MODIFY_PATH" || ko "honors MSAGENT_NO_MODIFY_PATH"
grep -q "已跳过文档查询服务准备" "${OUT}" && ok "honors MSAGENT_NO_ASCEND_DOC_MCP" || ko "honors MSAGENT_NO_ASCEND_DOC_MCP"
[ ! -f "${TEST_HOME}/.bashrc" ] && ok "no shell profile modified" || ko "no shell profile modified"
cleanup_env

# ---------------------------------------------------------------------------
# 3. PyPI fallback retry (auto index chain; requires huaweicloud reachable)
# ---------------------------------------------------------------------------
printf '\n== Scenario: PyPI fallback retry when mirror install fails ==\n'
if curl -fsS -o /dev/null --connect-timeout 6 --max-time 12 -I "https://mirrors.huaweicloud.com/repository/pypi/simple/pip/" 2>/dev/null; then
  OUT="$(mktemp)"
  run_installer "${OUT}" MSAGENT_TEST_UV_FAIL_PYPI_ONLY=1
  grep -q "改用 PyPI 官方源重试一次" "${OUT}" && ok "fallback retry triggered" || ko "fallback retry triggered"
  grep -q -- "https://pypi.org/simple" "${UV_LOG}" && ok "retry uses official PyPI" || ko "retry uses official PyPI"
  grep -q "即将安装 msagent" "${OUT}" && ok "announces target version" || ko "announces target version"
  grep -q "已选择 PyPI 源" "${OUT}" && ok "selected an index" || ko "selected an index"
  cleanup_env
else
  skip "huaweicloud mirror unreachable in this environment"
fi

# ---------------------------------------------------------------------------
# 4. NO_MODIFY_PATH + failing uv (venv fallback path is not exercised here)
# ---------------------------------------------------------------------------
printf '\n== Scenario: failing uv is reported and installer exits non-zero ==\n'
OUT="$(mktemp)"
run_installer "${OUT}" MSAGENT_INDEX=https://example.invalid/simple MSAGENT_TEST_UV_FAIL=1 MSAGENT_NO_FALLBACK=1
grep -q "已设置 MSAGENT_NO_FALLBACK" "${OUT}" && ok "reports failure with actionable message" || ko "reports failure with actionable message"
cleanup_env

# ---------------------------------------------------------------------------
# 5. ascend-doc-mcp stage, offline: intranet-friendly mirror probing
# ---------------------------------------------------------------------------
printf '\n== Scenario: Node provisioning prefers the Huawei Cloud mirror ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple \
  "PATH=$(node_free_path)"
grep -q "正在从 https://mirrors.huaweicloud.com/nodejs 下载用户级 Node 22 LTS" "${OUT}" \
  && ok "selects the Huawei Cloud Node mirror first" || ko "selects the Huawei Cloud Node mirror first"
grep -q "正在从 https://mirrors.huaweicloud.com/nodejs 下载 node-v22.99.0-linux-x64.tar.xz" "${OUT}" \
  && ok "parses the mirror index and picks the newest v22 archive" || ko "parses the mirror index and picks the newest v22 archive"
grep -q "无法从 https://mirrors.huaweicloud.com/nodejs 获取 Node.js" "${OUT}" \
  && ok "reports the failing mirror" || ko "reports the failing mirror"
grep -q "文档查询服务未就绪" "${OUT}" \
  && ok "stays non-fatal for msagent itself" || ko "stays non-fatal for msagent itself"
grep -q "export MSAGENT_NPM_REGISTRY=https://mirrors.huaweicloud.com/repository/npm" "${OUT}" \
  && ok "hints the intranet npm registry" || ko "hints the intranet npm registry"
grep -q "MSAGENT_NO_ASCEND_DOC_MCP=1" "${OUT}" \
  && ok "hints the opt-out" || ko "hints the opt-out"
grep -q "本功能需要 >= 22 才能使用" "${OUT}" \
  && ko "drops the unconditional Node >= 22 warning" || ok "drops the unconditional Node >= 22 warning"
[ "${INSTALL_RC}" = "0" ] && ok "installer still exits 0" || ko "installer still exits 0"
cleanup_env

printf '\n== Scenario: Node mirror falls back to npmmirror on the public internet ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_tool curl \
  'set -u' \
  'url=""; out=""' \
  'while [ "$#" -gt 0 ]; do' \
  '  case "$1" in' \
  '    -o) out="${2:-}"; shift 2 ;;' \
  '    http*) url="$1"; shift ;;' \
  '    *) shift ;;' \
  '  esac' \
  'done' \
  'emit() { if [ -n "${out}" ] && [ "${out}" != "/dev/null" ]; then printf "%s" "$1" > "${out}"; else printf "%s" "$1"; fi; }' \
  'case "${url}" in' \
  '  *mirrors.huaweicloud.com*) exit 7 ;;' \
  '  *registry.npmmirror.com/-/binary/node/latest-v22.x/*)' \
  '    emit "{\"name\":\"node-v22.99.0-linux-x64.tar.xz\"}"' \
  '    exit 0 ;;' \
  '  *registry.npmmirror.com*) exit 7 ;;' \
  '  *) exit 7 ;;' \
  'esac'
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple \
  "PATH=$(node_free_path)"
grep -q "Node 镜像不可达：https://mirrors.huaweicloud.com/nodejs" "${OUT}" \
  && ok "skips an unreachable candidate" || ko "skips an unreachable candidate"
grep -q "正在从 https://registry.npmmirror.com/-/binary/node 下载用户级 Node 22 LTS" "${OUT}" \
  && ok "falls back to the npmmirror binary mirror" || ko "falls back to the npmmirror binary mirror"
grep -q "正在从 https://registry.npmmirror.com/-/binary/node 下载 node-v22.99.0-linux-x64.tar.xz" "${OUT}" \
  && ok "parses the JSON mirror index" || ko "parses the JSON mirror index"
cleanup_env

printf '\n== Scenario: MSAGENT_NODE_MIRROR overrides the candidate chain ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_tool curl 'set -u' 'exit 7'
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple \
  "PATH=$(node_free_path)" MSAGENT_NODE_MIRROR=https://npm.internal.example/node
grep -q "正在从 https://npm.internal.example/node 下载用户级 Node 22 LTS" "${OUT}" \
  && ok "honors MSAGENT_NODE_MIRROR" || ko "honors MSAGENT_NODE_MIRROR"
cleanup_env

printf '\n== Scenario: a missing Node archive falls back to the next version ==\n'
OUT="$(mktemp)"
new_test_env
NODE_TARBALL="$(mktemp).tar.gz"
make_fake_node_tarball "${NODE_TARBALL}" node-v22.22.2-linux-x64 v22.22.2
# The listing mixes versions and formats and does NOT put the newest first, and
# the newest archive 404s: the installer must pick by version, not by text
# position, then step down to the next-newest instead of failing the stage.
write_fake_tool curl \
  'set -u' \
  'url=""; out=""' \
  'while [ "$#" -gt 0 ]; do' \
  '  case "$1" in' \
  '    -o) out="${2:-}"; shift 2 ;;' \
  '    http*) url="$1"; shift ;;' \
  '    *) shift ;;' \
  '  esac' \
  'done' \
  'emit() { if [ -n "${out}" ] && [ "${out}" != "/dev/null" ]; then printf "%s" "$1" > "${out}"; else printf "%s" "$1"; fi; }' \
  'case "${url}" in' \
  '  *node-v22.23.2-linux-x64.tar.xz) exit 7 ;;' \
  "  *node-v22.22.2-linux-x64.tar.gz) cp '${NODE_TARBALL}' \"\${out}\"; exit 0 ;;" \
  '  *node-v22.9.0-linux-x64.tar.gz) exit 7 ;;' \
  '  *mirrors.huaweicloud.com/nodejs/latest-v22.x/*)' \
  '    emit "<a href=\"node-v22.9.0-linux-x64.tar.gz\">node-v22.9.0-linux-x64.tar.gz</a>"' \
  '    emit "{\"name\":\"node-v22.22.2-linux-x64.tar.gz\"}"' \
  '    emit "<a href=\"node-v22.23.2-linux-x64.tar.xz\">node-v22.23.2-linux-x64.tar.xz</a>"' \
  '    exit 0 ;;' \
  '  *mirrors.huaweicloud.com*) exit 0 ;;' \
  '  *) exit 7 ;;' \
  'esac'
write_fake_npm 'https://mirrors.huaweicloud.com/repository/npm/'
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple \
  "PATH=$(node_free_path)"
FIRST_DOWNLOAD="$(grep -m 1 '下载 node-v22' "${OUT}" || true)"
case "${FIRST_DOWNLOAD}" in
  *node-v22.23.2-linux-x64.tar.xz*) ok "downloads the newest archive, not the first listed one" ;;
  *) ko "downloads the newest archive, not the first listed one (got '${FIRST_DOWNLOAD}')" ;;
esac
grep -q "下载失败：node-v22.23.2-linux-x64.tar.xz" "${OUT}" \
  && ok "reports the unusable archive" || ko "reports the unusable archive"
SECOND_DOWNLOAD="$(grep '下载 node-v22' "${OUT}" | sed -n 2p)"
case "${SECOND_DOWNLOAD}" in
  *node-v22.22.2-linux-x64.tar.gz*) ok "steps down to the next-newest archive" ;;
  *) ko "steps down to the next-newest archive (got '${SECOND_DOWNLOAD}')" ;;
esac
grep -q "无法从 https://mirrors.huaweicloud.com/nodejs 获取 Node.js" "${OUT}" \
  && ko "the fallback keeps the stage alive" || ok "the fallback keeps the stage alive"
[ -x "${TEST_HOME}/.msagent/node/bin/node" ] \
  && ok "the fallback installed a usable Node" || ko "the fallback installed a usable Node"
rm -f "${NODE_TARBALL}"
cleanup_env

# ---------------------------------------------------------------------------
# 6. ascend-doc-mcp stage, offline: npm registry fallback + npm diagnostics
# ---------------------------------------------------------------------------
printf '\n== Scenario: npm registry fallback and npm failure are reported ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl
write_fake_tool node 'case "${1:-}" in --version) echo v22.0.0 ;; esac' 'exit 0'
write_fake_tool npm 'echo "npm ERR! code EAI_AGAIN" >&2' 'echo "npm ERR! network request failed" >&2' 'exit 1'
write_fake_tool npx 'exit 0'
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "使用已有 Node v22.0.0" "${OUT}" \
  && ok "reuses a usable system Node" || ko "reuses a usable system Node"
grep -q "源：https://mirrors.huaweicloud.com/repository/npm" "${OUT}" \
  && ok "falls back to the Huawei Cloud npm registry" || ko "falls back to the Huawei Cloud npm registry"
grep -q "所有 npm 源都没有响应元数据请求，改为按顺序逐个尝试安装" "${OUT}" \
  && ok "falls back to trying every source when none answers metadata" \
  || ko "falls back to trying every source when none answers metadata"
grep -q "在 https://mirrors.huaweicloud.com/repository/npm 上预安装失败" "${OUT}" \
  && ok "reports the failing registry" || ko "reports the failing registry"
grep -q "npm ERR! code EAI_AGAIN" "${OUT}" \
  && ok "surfaces the npm error tail" || ko "surfaces the npm error tail"
[ "${INSTALL_RC}" = "0" ] && ok "installer still exits 0" || ko "installer still exits 0"
cleanup_env

printf '\n== Scenario: an outdated system Node triggers provisioning ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl
write_fake_node v18.19.0
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "检测到 Node v18.19.0，但本功能需要 >= 22，将安装用户级 Node。" "${OUT}" \
  && ok "detects an outdated Node" || ko "detects an outdated Node"
grep -q "正在从 https://mirrors.huaweicloud.com/nodejs 下载用户级 Node 22 LTS" "${OUT}" \
  && ok "provisions instead of using it" || ko "provisions instead of using it"
cleanup_env

# ---------------------------------------------------------------------------
# 7. ascend-doc-mcp stage, offline: automatic intranet/internet registry choice
# ---------------------------------------------------------------------------
printf '\n== Scenario: the machine-configured npm registry wins (intranet) ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl '*npm.internal.example*' '9.9.9'
write_fake_npm 'http://npm.internal.example/'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "使用本机 npm 已配置的源：http://npm.internal.example/" "${OUT}" \
  && ok "uses the registry npm is already configured with" || ko "uses the registry npm is already configured with"
grep -q "正在预安装 @opencxd/ascend-doc-mcp@9.9.9（源：http://npm.internal.example/）" "${OUT}" \
  && ok "installs the version reported by the probed source" || ko "installs the version reported by the probed source"
grep -q -- "--registry http://npm.internal.example/" "${TEST_HOME}/npm-args.log" \
  && ok "the install itself used the intranet registry" || ko "the install itself used the intranet registry"
grep -q "npm 源不可达" "${OUT}" \
  && ko "does not probe the public chain when the intranet registry works" \
  || ok "does not probe the public chain when the intranet registry works"
grep -q "文档查询服务已预安装到" "${OUT}" \
  && ok "pre-install succeeds without any environment variable" || ko "pre-install succeeds without any environment variable"
grep -q "\[1/6\] 检查环境并选择下载源" "${OUT}" \
  && ok "numbers the install phases" || ko "numbers the install phases"
grep -q "\[6/6\] 验证安装结果" "${OUT}" \
  && ok "reaches the verify phase" || ko "reaches the verify phase"
grep -q "安装结果" "${OUT}" \
  && ok "prints a final summary block" || ko "prints a final summary block"
grep -q "未能从 https://example.invalid/simple 解析出版本号" "${OUT}" \
  && ok "says so explicitly when no version can be resolved" \
  || ko "says so explicitly when no version can be resolved"
grep -q "PyPI 源" "${OUT}" && grep -q "文档查询服务" "${OUT}" \
  && ok "summary reports the selected source and MCP status" || ko "summary reports the selected source and MCP status"
cleanup_env

printf '\n== Scenario: the published package version is pinned from the registry tag ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl '*registry.npmmirror.com*' '9.9.9'
write_fake_npm 'https://registry.npmmirror.com/'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "正在预安装 @opencxd/ascend-doc-mcp@9.9.9（源：https://registry.npmmirror.com）" "${OUT}" \
  && ok "resolves the version over plain HTTP (no npm needed)" || ko "resolves the version over plain HTTP (no npm needed)"
grep -q -- "--no-package-lock @opencxd/ascend-doc-mcp@9.9.9" "${TEST_HOME}/npm-args.log" \
  && ok "installs the resolved version" || ko "installs the resolved version"
grep -q "文档查询服务 *v9.9.9" "${OUT}" \
  && ok "summary reports the installed package version" || ko "summary reports the installed package version"
cleanup_env

printf '\n== Scenario: a public npm default is ignored, the probe chain is used ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl '*registry.npmmirror.com*'
write_fake_npm 'https://registry.npmmirror.com/'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "使用本机 npm 已配置的源" "${OUT}" \
  && ko "ignores a public default" || ok "ignores a public default"
grep -q -- "--registry https://registry.npmmirror.com " "${TEST_HOME}/npm-args.log" \
  && ok "public internet still uses the domestic mirror chain" || ko "public internet still uses the domestic mirror chain"
cleanup_env

printf '\n== Scenario: an intranet registry that fails is left for the public chain ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl '*npm.internal.example*'
write_fake_tool npm \
  'if [ "${1:-}" = "config" ] && [ "${2:-}" = "get" ]; then echo "http://npm.internal.example/"; exit 0; fi' \
  'printf "%s\n" "$*" >> "${HOME}/npm-args.log"' \
  'case "$*" in' \
  '  *npm.internal.example*) echo "npm error network request to http://npm.internal.example/ failed" >&2; exit 1 ;;' \
  'esac' \
  'prefix=""' \
  'while [ "$#" -gt 0 ]; do case "$1" in --prefix) prefix="$2"; shift 2 ;; *) shift ;; esac; done' \
  '[ -n "${prefix}" ] || exit 1' \
  'mkdir -p "${prefix}/node_modules/@opencxd/ascend-doc-mcp"' \
  ': > "${prefix}/node_modules/@opencxd/ascend-doc-mcp/package.json"' \
  'exit 0'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "正在预安装 @opencxd/ascend-doc-mcp@latest（源：http://npm.internal.example/）" "${OUT}" \
  && ok "tries the configured intranet registry first" || ko "tries the configured intranet registry first"
grep -q "在 http://npm.internal.example/ 上预安装失败" "${OUT}" \
  && ok "reports the intranet failure" || ko "reports the intranet failure"
grep -q "正在预安装 @opencxd/ascend-doc-mcp@latest（源：https://registry.npmmirror.com）" "${OUT}" \
  && ok "falls through to the next registry" || ko "falls through to the next registry"
grep -q "文档查询服务已预安装到" "${OUT}" \
  && ok "pre-install still succeeds" || ko "pre-install still succeeds"
grep -q "npm 源 *https://registry.npmmirror.com" "${OUT}" \
  && ok "summary reports the registry that worked" || ko "summary reports the registry that worked"
[ "${INSTALL_RC}" = "0" ] && ok "installer still exits 0" || ko "installer still exits 0"
cleanup_env

printf '\n== Scenario: an unreachable configured registry is a warned last resort ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_tool curl 'set -u' 'exit 7'
write_fake_npm 'http://stale.internal.example/'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "所有 npm 源都没有响应元数据请求" "${OUT}" \
  && ok "warns when no registry answers metadata" || ko "warns when no registry answers metadata"
grep -q -- "--registry http://stale.internal.example/" "${TEST_HOME}/npm-args.log" \
  && ok "still tries it when nothing else is reachable" || ko "still tries it when nothing else is reachable"
[ "${INSTALL_RC}" = "0" ] && ok "installer still exits 0" || ko "installer still exits 0"
cleanup_env

# ---------------------------------------------------------------------------
# 8. ascend-doc-mcp stage, offline: explicit registry preference vs pinning
# ---------------------------------------------------------------------------
printf '\n== Scenario: an explicit MSAGENT_NPM_REGISTRY still falls back ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl '*npm.internal.example*'
write_fake_npm_failing_host 'npm.internal.example'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple \
  MSAGENT_NPM_REGISTRY=http://npm.internal.example/
grep -q "正在预安装 @opencxd/ascend-doc-mcp@latest（源：http://npm.internal.example/）" "${OUT}" \
  && ok "tries the explicit registry first" || ko "tries the explicit registry first"
grep -q "在 http://npm.internal.example/ 上预安装失败" "${OUT}" \
  && ok "reports the explicit registry failure" || ko "reports the explicit registry failure"
grep -q "正在预安装 @opencxd/ascend-doc-mcp@latest（源：https://registry.npmmirror.com）" "${OUT}" \
  && ok "falls back to a public mirror" || ko "falls back to a public mirror"
grep -q "文档查询服务已预安装到" "${OUT}" \
  && ok "pre-install still succeeds" || ko "pre-install still succeeds"
[ "${INSTALL_RC}" = "0" ] && ok "installer still exits 0" || ko "installer still exits 0"
cleanup_env

printf '\n== Scenario: MSAGENT_NPM_REGISTRY_ONLY pins the source ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl '*npm.internal.example*'
write_fake_npm_failing_host 'npm.internal.example'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple \
  MSAGENT_NPM_REGISTRY=http://npm.internal.example/ MSAGENT_NPM_REGISTRY_ONLY=1
grep -q "所有候选源均无法预安装文档查询服务" "${OUT}" \
  && ok "does not fall back when pinned" || ko "does not fall back when pinned"
grep -q "内网环境请把 npm 源指向内网镜像" "${OUT}" && grep -q "npm config set registry" "${OUT}" \
  && ok "hints the intranet mirror when every source fails" || ko "hints the intranet mirror when every source fails"
grep -q "registry.npmmirror.com" "${TEST_HOME}/npm-args.log" \
  && ko "never tries the public chain when pinned" || ok "never tries the public chain when pinned"
[ "${INSTALL_RC}" = "0" ] && ok "installer still exits 0" || ko "installer still exits 0"
cleanup_env

# ---------------------------------------------------------------------------
# 9. ascend-doc-mcp stage, offline: unwritable npm cache
# ---------------------------------------------------------------------------
printf '\n== Scenario: an unwritable npm cache falls back to a usable dir ==\n'
if [ "$(id -u)" -eq 0 ]; then
  skip "running as root: an unwritable directory cannot be simulated"
else
  OUT="$(mktemp)"
  new_test_env
  mkdir -p "${TEST_HOME}/.cache/msagent/npm-cache"
  chmod 500 "${TEST_HOME}/.cache/msagent/npm-cache"
  write_fake_curl '*registry.npmmirror.com*'
  write_fake_npm 'https://registry.npmjs.org/'
  write_fake_node
  run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
  grep -q "npm 缓存目录不可写" "${OUT}" \
    && ok "detects the unwritable cache" || ko "detects the unwritable cache"
  grep -q -- "--cache ${TEST_HOME}/.msagent/npm-cache" "${TEST_HOME}/npm-args.log" \
    && ok "falls back to a writable cache" || ko "falls back to a writable cache"
  grep -q "文档查询服务已预安装到" "${OUT}" \
    && ok "pre-install still succeeds" || ko "pre-install still succeeds"
  chmod 700 "${TEST_HOME}/.cache/msagent/npm-cache" 2>/dev/null || true
  cleanup_env
fi

# ---------------------------------------------------------------------------
# 10. install.sh: the announced version comes from the selected index
# ---------------------------------------------------------------------------
printf '\n== Scenario: the announced version comes from the selected index ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_tool curl \
  'set -u' \
  'url=""; out=""' \
  'while [ "$#" -gt 0 ]; do' \
  '  case "$1" in' \
  '    -o) out="${2:-}"; shift 2 ;;' \
  '    http*) url="$1"; shift ;;' \
  '    *) shift ;;' \
  '  esac' \
  'done' \
  'emit() { if [ -n "${out}" ] && [ "${out}" != "/dev/null" ]; then printf "%s" "$1" > "${out}"; else printf "%s" "$1"; fi; }' \
  'case "${url}" in' \
  '  *mirrors.internal.example/simple/mindstudio-agent/*)' \
  '    emit "<a href=\"../../packages/mindstudio_agent-26.9.9-py3-none-any.whl\">mindstudio_agent-26.9.9-py3-none-any.whl</a>"' \
  '    exit 0 ;;' \
  '  *) exit 7 ;;' \
  'esac'
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://mirrors.internal.example/simple
grep -q "即将安装 msagent 26.9.9（最新版，源：https://mirrors.internal.example/simple）" "${OUT}" \
  && ok "announces the concrete version from the index" || ko "announces the concrete version from the index"
grep -q "mindstudio-agent==26.9.9" "${UV_LOG}" \
  && ok "installs exactly the announced version" || ko "installs exactly the announced version"
cleanup_env

# ---------------------------------------------------------------------------
# 11. install.sh: a newer local install is not silently downgraded
# ---------------------------------------------------------------------------
printf '\n== Scenario: a newer local install is not downgraded ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_tool msagent 'case "${1:-}" in --version) printf "msagent 99.9.9 (unknown)\n" ;; esac' 'exit 0'
write_fake_tool curl \
  'set -u' \
  'url=""; out=""' \
  'while [ "$#" -gt 0 ]; do' \
  '  case "$1" in' \
  '    -o) out="${2:-}"; shift 2 ;;' \
  '    http*) url="$1"; shift ;;' \
  '    *) shift ;;' \
  '  esac' \
  'done' \
  'emit() { if [ -n "${out}" ] && [ "${out}" != "/dev/null" ]; then printf "%s" "$1" > "${out}"; else printf "%s" "$1"; fi; }' \
  'case "${url}" in' \
  '  *mirrors.internal.example/simple/mindstudio-agent/*)' \
  '    emit "<a href=\"../../packages/mindstudio_agent-26.9.9-py3-none-any.whl\">mindstudio_agent-26.9.9-py3-none-any.whl</a>"' \
  '    exit 0 ;;' \
  '  *) exit 7 ;;' \
  'esac'
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://mirrors.internal.example/simple
grep -q "已安装的 msagent 99.9.9 高于源上的 26.9.9，跳过安装以免降级" "${OUT}" \
  && ok "warns instead of downgrading" || ko "warns instead of downgrading"
grep -q "mindstudio-agent==" "${UV_LOG}" \
  && ko "does not run uv tool install" || ok "does not run uv tool install"
cleanup_env

# ---------------------------------------------------------------------------
# 12. install.sh: site-injected fallback npm sources (no internal host in-repo)
# ---------------------------------------------------------------------------
printf '\n== Scenario: a site-injected fallback source is used last ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl '*mirror.corp.example*'
write_fake_tool npm \
  'if [ "${1:-}" = "config" ] && [ "${2:-}" = "get" ]; then echo "https://registry.npmjs.org/"; exit 0; fi' \
  'printf "%s\n" "$*" >> "${HOME}/npm-args.log"' \
  'case "$*" in' \
  '  *mirror.corp.example*) : ;;' \
  '  *) echo "npm error network request failed" >&2; exit 1 ;;' \
  'esac' \
  'prefix=""' \
  'while [ "$#" -gt 0 ]; do case "$1" in --prefix) prefix="$2"; shift 2 ;; *) shift ;; esac; done' \
  '[ -n "${prefix}" ] || exit 1' \
  'mkdir -p "${prefix}/node_modules/@opencxd/ascend-doc-mcp"' \
  ': > "${prefix}/node_modules/@opencxd/ascend-doc-mcp/package.json"' \
  'exit 0'
write_fake_node
mkdir -p "${TEST_HOME}/.msagent"
printf '# internal mirrors\nexport  http://mirror.corp.example/npm\n' > "${TEST_HOME}/.msagent/npm-mirrors"
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple \
  MSAGENT_NPM_REGISTRY_FALLBACKS='https://registry.npmmirror.com, http://mirror.corp.example/npm'
grep -q "正在预安装 .*（源：http://mirror.corp.example/npm）" "${OUT}" \
  && ok "tries the site-injected fallback" || ko "tries the site-injected fallback"
grep -q "文档查询服务已预安装到" "${OUT}" \
  && ok "pre-install succeeds through the fallback" || ko "pre-install succeeds through the fallback"
COUNT="$(grep -c -- "--registry https://registry.npmmirror.com " "${TEST_HOME}/npm-args.log" 2>/dev/null || true)"
[ "${COUNT}" = "1" ] && ok "de-duplicates a fallback repeating a public source" \
  || ko "de-duplicates a fallback repeating a public source (tried ${COUNT}x)"
cleanup_env

printf '\n== Scenario: a public success never touches the fallback ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_curl '*registry.npmmirror.com*'
write_fake_npm 'https://registry.npmjs.org/'
write_fake_node
mkdir -p "${TEST_HOME}/.msagent"
printf 'http://mirror.corp.example/npm\n' > "${TEST_HOME}/.msagent/npm-mirrors"
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "mirror.corp.example" "${TEST_HOME}/npm-args.log" \
  && ko "does not touch the fallback when the public chain works" \
  || ok "does not touch the fallback when the public chain works"
grep -q "cmc-cd-mirror" "${TEST_HOME}/npm-args.log" \
  && ko "does not touch the built-in intranet mirror either" \
  || ok "does not touch the built-in intranet mirror either"
cleanup_env

printf '\n== Scenario: the built-in intranet mirror is the last resort ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_tool curl 'set -u' 'exit 7'
write_fake_tool npm \
  'if [ "${1:-}" = "config" ] && [ "${2:-}" = "get" ]; then echo "https://registry.npmjs.org/"; exit 0; fi' \
  'printf "%s\n" "$*" >> "${HOME}/npm-args.log"' \
  'case "$*" in' \
  '  *cmc-cd-mirror.rnd.huawei.com*) : ;;' \
  '  *) echo "npm error code ETIMEDOUT" >&2; exit 1 ;;' \
  'esac' \
  'prefix=""' \
  'while [ "$#" -gt 0 ]; do case "$1" in --prefix) prefix="$2"; shift 2 ;; *) shift ;; esac; done' \
  '[ -n "${prefix}" ] || exit 1' \
  'mkdir -p "${prefix}/node_modules/@opencxd/ascend-doc-mcp"' \
  ': > "${prefix}/node_modules/@opencxd/ascend-doc-mcp/package.json"' \
  'exit 0'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple
grep -q "正在预安装 .*（源：http://cmc-cd-mirror.rnd.huawei.com/npm）" "${OUT}" \
  && ok "falls back to the built-in intranet mirror" || ko "falls back to the built-in intranet mirror"
grep -q "文档查询服务已预安装到" "${OUT}" \
  && ok "pre-install succeeds through the built-in mirror" || ko "pre-install succeeds through the built-in mirror"
LAST="$(grep -o -- '--registry [^ ]*' "${TEST_HOME}/npm-args.log" | tail -1)"
case "${LAST}" in
  *cmc-cd-mirror.rnd.huawei.com*) ok "the intranet mirror is tried last" ;;
  *) ko "the intranet mirror is tried last (last was '${LAST}')" ;;
esac
cleanup_env

# ---------------------------------------------------------------------------
# 13. install.sh: probe first, install only on the reachable source
# ---------------------------------------------------------------------------
printf '\n== Scenario: only the probed, reachable source is used for install ==\n'
OUT="$(mktemp)"
new_test_env
write_fake_tool curl \
  'set -u' \
  'url=""; out=""' \
  'while [ "$#" -gt 0 ]; do' \
  '  case "$1" in' \
  '    -o) out="${2:-}"; shift 2 ;;' \
  '    http*) url="$1"; shift ;;' \
  '    *) shift ;;' \
  '  esac' \
  'done' \
  'emit() { if [ -n "${out}" ] && [ "${out}" != "/dev/null" ]; then printf "%s" "$1" > "${out}"; else printf "%s" "$1"; fi; }' \
  'case "${url}" in' \
  '  *mirror.reachable.example*/@opencxd%2Fascend-doc-mcp/latest) emit "{\"version\":\"7.7.7\"}"; exit 0 ;;' \
  '  *) exit 7 ;;' \
  'esac'
write_fake_tool npm \
  'if [ "${1:-}" = "config" ] && [ "${2:-}" = "get" ]; then echo "https://registry.npmjs.org/"; exit 0; fi' \
  'printf "%s\n" "$*" >> "${HOME}/npm-args.log"' \
  'case "$*" in' \
  '  *mirror.reachable.example*) : ;;' \
  '  *) echo "npm error network request failed" >&2; exit 1 ;;' \
  'esac' \
  'prefix=""' \
  'while [ "$#" -gt 0 ]; do case "$1" in --prefix) prefix="$2"; shift 2 ;; *) shift ;; esac; done' \
  '[ -n "${prefix}" ] || exit 1' \
  'mkdir -p "${prefix}/node_modules/@opencxd/ascend-doc-mcp"' \
  ': > "${prefix}/node_modules/@opencxd/ascend-doc-mcp/package.json"' \
  'exit 0'
write_fake_node
run_installer "${OUT}" MSAGENT_NO_ASCEND_DOC_MCP= MSAGENT_INDEX=https://example.invalid/simple \
  MSAGENT_NPM_REGISTRY_FALLBACKS=http://mirror.reachable.example/npm
grep -q "正在探测 npm 源" "${OUT}" \
  && ok "probes the sources before installing" || ko "probes the sources before installing"
grep -q "可达的 npm 源：http://mirror.reachable.example/npm" "${OUT}" \
  && ok "reports which sources answered" || ko "reports which sources answered"
ATTEMPTS="$(grep -c -- '--registry ' "${TEST_HOME}/npm-args.log" 2>/dev/null || true)"
[ "${ATTEMPTS}" = "1" ] && ok "installs once, on the reachable source only" \
  || ko "installs once, on the reachable source only (made ${ATTEMPTS} attempts)"
grep -q "正在预安装 @opencxd/ascend-doc-mcp@7.7.7（源：http://mirror.reachable.example/npm）" "${OUT}" \
  && ok "uses the version the probe reported" || ko "uses the version the probe reported"
grep -q "文档查询服务已预安装到" "${OUT}" \
  && ok "pre-install succeeds" || ko "pre-install succeeds"
cleanup_env

# ---------------------------------------------------------------------------
printf '\n== Summary ==\n'
printf 'passed=%d failed=%d skipped=%d\n' "${PASS}" "${FAIL}" "${SKIP}"
rm -f "${OUT:-}" 2>/dev/null || true
[ "${FAIL}" -eq 0 ]
