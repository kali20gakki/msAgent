# Building and Packaging

This document describes the unified build entry point `build.py` of msAgent and the underlying build script `scripts/build_whl.sh` that it invokes.

## Recommended Approach

Follow [msAgent Installation Guide — Source Code Installation](../install_guide/msagent_install_guide.md#33-source-code-installation) to set up the compilation and test environment.

> **Note:** The method for building the environment image and the bundled software versions are maintained by the *MindStudio Unified Image Building Guide*. This repository does not redefine them.

You are advised to use the unified build entry point in the repository root directory:

```bash
python3 build.py
```

After a successful build, the wheel is first generated in `dist/` and then archived to `artifacts/`. The installation command is as follows:

```bash
pip install artifacts/mindstudio_agent-<version>-py3-none-any.whl
```

## Build and Test Commands

| Command | Description |
|---|---|
| python3 build.py | Installs `uv`, builds the wheel, and archives it. |
| python3 build.py local | Builds and archives the wheel using the locally installed `uv`. |
| python3 build.py test | Installs `uv` and runs `tests/ut` and `tests/skills`. |
| python3 build.py test local | Runs `tests/ut` and `tests/skills` using the locally installed `uv`. |

The build process of `build.py` is as follows:

1. Runs `python -m pip install uv` in non-`local` mode.
2. Calls `scripts/build_whl.sh` in build mode to sync the build version, check Python and skills resources, and build the wheel.
3. Copies the whl from `dist/` to `artifacts/`.
4. Calls `scripts/run_ut.sh` in test mode, in which `uv run` automatically prepares the test dependencies.

`scripts/build_whl.sh` can still be used independently, which suits scenarios where you need to directly control its environment variables. It first validates `uv.lock`. If the lock file is inconsistent, it automatically runs `uv lock` in the current workspace to update it, and then uses `uv build`. If `uv` is not installed, it falls back to `pip wheel`.

## Common Build Parameters

| Environment Variable | Default Value | Description |
|---|---|---|
| DIST_DIR | `dist/` | Output directory. |
| SKILLS_PATH | `skills` | Specifies the skills directory to be packaged. By default, the `skills/` directory at the repository root is used. |
| WHL_VERSION | `Version` in `version.info` | Specifies the wheel version number. If it is not set, the `Version` in `version.info` is used. |
| VERIFY_WHEEL_INSTALL | 0 | Whether to perform a wheel installation smoke test in a temporary virtual environment. |
| PYTHON_BIN | Auto-detected | Specifies the Python used for the build. |
| SMOKE_IMPORT_MODULE | Auto-derived | Module imported during the smoke test. |
| SMOKE_RESOURCE_PATH | `resources/configs/default/config.mcp.json` | Resource file checked during the smoke test to confirm that it is packaged into the wheel. |
| SMOKE_SKILL_PATH | `resources/configs/default/skills/README.md` | Skills resource file checked during the smoke test to confirm that it is packaged into the wheel. |

If you want to include the installation smoke test:

```bash
python3 build.py --extra VERIFY_WHEEL_INSTALL=1
```

## Manual Build

If you do not use the script, you can run the equivalent commands manually:

```bash
# Install uv
pip install uv

# Ensure the skills directory exists
test -d skills
# Check that the lock file is up to date and consistent with the current project dependency declarations
uv lock --check
# Build the wheel package of the project
uv build --wheel --out-dir dist .
```

## Installing the Build Result

After the build is complete, `mindstudio_agent-*.whl` is generated in the `dist/` directory and can be installed directly:

```bash
pip install dist/mindstudio_agent-<version>-py3-none-any.whl
```

You can also install the corresponding wheel file directly in Windows PowerShell or CMD. For example:

```powershell
pip install .\dist\mindstudio_agent-<version>-py3-none-any.whl
```

## Related Files

- Unified build entry point: `build.py`
- Underlying build script: `scripts/build_whl.sh`
- Project metadata: `pyproject.toml`
- Default skills directory: `skills/`
