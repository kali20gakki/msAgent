#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${REPO_ROOT}/dist"

mkdir -p "${DIST_DIR}"
rm -f "${DIST_DIR}"/msagent-*.whl

echo "Building msagent wheel package..."

if command -v uv >/dev/null 2>&1; then
  uv build --wheel --out-dir "${DIST_DIR}" "${REPO_ROOT}"
else
  python -m pip install --upgrade build
  python -m build --wheel --outdir "${DIST_DIR}" "${REPO_ROOT}"
fi

WHEEL_PATH="$(find "${DIST_DIR}" -maxdepth 1 -type f -name "msagent-*.whl" | sort | tail -n 1)"

if [[ -z "${WHEEL_PATH}" ]]; then
  echo "Build failed: no wheel generated in ${DIST_DIR}" >&2
  exit 1
fi

echo "Wheel generated: ${WHEEL_PATH}"
echo "Install with:"
echo "pip install \"${WHEEL_PATH}\""
