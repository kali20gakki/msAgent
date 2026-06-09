#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="myenv"
PYTHON_VERSION=""
USE_EXISTING_ENV=0
SET_PROJECT_ENV=0
USE_HF_MIRROR=0
USE_PROJECT_UV_CACHE=1
PYPI_MIRROR="https://mirrors.ustc.edu.cn/pypi/web/simple"
MSMODELING_REPO_URL="https://gitcode.com/Ascend/msmodeling.git"

usage() {
    cat <<'EOF'
Usage: bash ./.agents/skills/msmodeling-env-installer/scripts/install-current-project-deps.sh [options]

Options:
  --env-name <name>          Virtual environment directory name. Default: myenv
  --python-version <version> Python version passed to "uv venv". Default: detected major.minor
  --use-existing-env         Install into an existing environment instead of creating myenv
  --set-project-env          Set PYTHONPATH for this script process and print the export command
  --use-hf-mirror            Set HF_ENDPOINT for this script process and print the export command
  --no-project-uv-cache      Do not set UV_CACHE_DIR to .uv-cache under the repository root
  -h, --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            ENV_NAME="${2:?Missing value for --env-name}"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="${2:?Missing value for --python-version}"
            shift 2
            ;;
        --use-existing-env)
            USE_EXISTING_ENV=1
            shift
            ;;
        --set-project-env)
            SET_PROJECT_ENV=1
            shift
            ;;
        --use-hf-mirror)
            USE_HF_MIRROR=1
            shift
            ;;
        --no-project-uv-cache)
            USE_PROJECT_UV_CACHE=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

resolve_python_launcher() {
    if command_exists python3; then
        echo "python3"
        return
    fi
    if command_exists python; then
        echo "python"
        return
    fi
    echo "No Python launcher found. Install Python 3.10+ first." >&2
    exit 1
}

python_version_text() {
    "$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")'
}

python_major_minor() {
    "$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
}

assert_python_min_version() {
    "$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
        echo "Detected Python $(python_version_text). Python 3.10+ is required." >&2
        exit 1
    }
}

python_scripts_path() {
    "$PYTHON" -c 'import sysconfig; print(sysconfig.get_path("scripts") or "")'
}

resolve_uv_command() {
    if command_exists uv; then
        command -v uv
        return
    fi

    local scripts_path
    scripts_path="$(python_scripts_path)"
    if [[ -n "$scripts_path" ]]; then
        for file_name in uv uv.exe; do
            if [[ -x "$scripts_path/$file_name" ]]; then
                echo "$scripts_path/$file_name"
                return
            fi
        done
    fi

    echo "uv executable not found after installation. Ensure Python Scripts directory is on PATH or reinstall uv." >&2
    exit 1
}

enable_project_uv_cache() {
    if [[ "$USE_PROJECT_UV_CACHE" -eq 0 ]]; then
        return
    fi

    if [[ -n "${UV_CACHE_DIR:-}" ]]; then
        echo "Using existing UV_CACHE_DIR: $UV_CACHE_DIR"
        return
    fi

    export UV_CACHE_DIR="$REPO_ROOT/.uv-cache"
    mkdir -p "$UV_CACHE_DIR"
    echo "UV_CACHE_DIR set for current session: $UV_CACHE_DIR"
}

ensure_msmodeling_repo_root() {
    if [[ -f "README.md" && -f "requirements.txt" ]]; then
        return
    fi

    if [[ -f "msmodeling/README.md" && -f "msmodeling/requirements.txt" ]]; then
        echo "msmodeling repository found under ./msmodeling. Entering it..."
        cd msmodeling
        return
    fi

    if [[ -e "msmodeling" ]]; then
        echo "Path ./msmodeling exists but does not look like the msmodeling repository root. Move it aside or run this script from a valid msmodeling repository root." >&2
        exit 1
    fi

    echo "msmodeling repository root not found. Cloning from $MSMODELING_REPO_URL ..."
    git clone "$MSMODELING_REPO_URL"
    cd msmodeling

    if [[ ! -f "README.md" || ! -f "requirements.txt" ]]; then
        echo "Clone finished, but README.md or requirements.txt is missing. Check repository contents." >&2
        exit 1
    fi
}

test_python_module_available() {
    local launcher="$1"
    local module_name="$2"
    "$launcher" -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$module_name') else 1)" >/dev/null 2>&1
}

test_python_package_installed() {
    local launcher="$1"
    local package_name="$2"
    "$launcher" -m pip show "$package_name" >/dev/null 2>&1
}

assert_existing_environment_clean() {
    local launcher="$1"
    local blocked_packages=()

    if test_python_module_available "$launcher" "torch_npu"; then
        blocked_packages+=("torch_npu")
    fi

    for package_name in torch-npu torch_npu cudatoolkit; do
        if test_python_package_installed "$launcher" "$package_name"; then
            blocked_packages+=("$package_name")
        fi
    done

    if [[ "${#blocked_packages[@]}" -gt 0 ]]; then
        local package_list
        package_list="$(IFS=", "; echo "${blocked_packages[*]}")"
        echo "Existing environment contains $package_list. README fallback requires an environment without torch_npu or cudatoolkit. Create a fresh environment by rerunning without --use-existing-env." >&2
        exit 1
    fi

    echo "Existing environment check passed: torch_npu and cudatoolkit are absent."
}

ensure_msmodeling_repo_root

REPO_ROOT="$(pwd)"
PYTHON="$(resolve_python_launcher)"
DETECTED_PYTHON="$(python_version_text)"
assert_python_min_version
echo "Detected Python version: $DETECTED_PYTHON"

if ! command_exists uv; then
    echo "uv not found. Installing uv with pip..."
    "$PYTHON" -m pip install uv -i "$PYPI_MIRROR"
fi

UV="$(resolve_uv_command)"
echo "Using uv executable: $UV"
enable_project_uv_cache

if [[ -z "$PYTHON_VERSION" ]]; then
    PYTHON_VERSION="$(python_major_minor)"
    echo "Python version not specified. Using detected Python version for venv: $PYTHON_VERSION"
fi

VENV_PYTHON="$REPO_ROOT/$ENV_NAME/bin/python"
if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]]; then
    VENV_PYTHON="$REPO_ROOT/$ENV_NAME/Scripts/python.exe"
fi

if [[ "$USE_EXISTING_ENV" -eq 0 ]]; then
    if [[ -e "$ENV_NAME" ]]; then
        echo "Environment path already exists: $ENV_NAME. Rerun with --use-existing-env to reuse it, or remove the directory after confirming it can be rebuilt." >&2
        exit 1
    fi

    echo "Creating virtual environment: $ENV_NAME (Python $PYTHON_VERSION)"
    "$UV" venv --python "$PYTHON_VERSION" "$ENV_NAME"

    if [[ ! -x "$VENV_PYTHON" ]]; then
        echo "Virtual environment python not found: $VENV_PYTHON" >&2
        exit 1
    fi

    echo "Installing dependencies with uv pip..."
    "$UV" pip install --python "$VENV_PYTHON" -r requirements.txt -i "$PYPI_MIRROR"
else
    echo "Using existing environment fallback: pip install -r requirements.txt"
    if [[ -x "$VENV_PYTHON" ]]; then
        assert_existing_environment_clean "$VENV_PYTHON"
        "$VENV_PYTHON" -m pip install -r requirements.txt
    else
        assert_existing_environment_clean "$PYTHON"
        "$PYTHON" -m pip install -r requirements.txt
    fi
fi

if [[ -x "$VENV_PYTHON" ]]; then
    "$UV" pip check --python "$VENV_PYTHON"
else
    "$PYTHON" -m pip check
fi

if [[ "$SET_PROJECT_ENV" -eq 1 ]]; then
    export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    echo "PYTHONPATH set for current script process: $PYTHONPATH"
    echo "Run in your shell to keep it after this script exits:"
    echo "export PYTHONPATH=\"$REPO_ROOT\${PYTHONPATH:+:\$PYTHONPATH}\""
fi

if [[ "$USE_HF_MIRROR" -eq 1 ]]; then
    export HF_ENDPOINT="https://hf-mirror.com"
    echo "HF_ENDPOINT set for current script process: $HF_ENDPOINT"
    echo "Run in your shell to keep it after this script exits:"
    echo 'export HF_ENDPOINT="https://hf-mirror.com"'
fi

if [[ "$OSTYPE" == msys* || "$OSTYPE" == cygwin* ]]; then
    echo "Done. Activation command (Git Bash): source $ENV_NAME/Scripts/activate"
else
    echo "Done. Activation command: source $ENV_NAME/bin/activate"
fi
