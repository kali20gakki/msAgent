# msAgent Installation Guide

## 1. Installation Notes

This document is intended for users who are new to MindStudio-Agent and helps you complete the msAgent installation.

You are advised to use [one-click installation](#31-one-click-installation). It automatically installs the latest version of `mindstudio-agent` without conflicting with the existing environment.

The following methods are also supported: [pip installation](#32-pip-installation) and [source code installation](#33-source-code-installation).

> One-click installation uses an isolated uv tool environment. **You can only uninstall it with `uv tool uninstall mindstudio-agent`, and `pip uninstall` does not work.** For the upgrade and uninstall commands of each method, see [§5.2](#52-selecting-the-upgrade-and-uninstall-command-by-installation-method).

## 2. Environment Requirements

- **No Python installation is required in advance**: One-click installation automatically downloads the required Python version (`>= 3.11`).

- `glibc >= 2.34`

  This is required to satisfy the dependency of the `trace_processor` binary in `msprof-mcp`. Recommended operating systems are `Ubuntu >= 21.10` and `openEuler >= 21.09`. For other operating systems, check the requirements yourself.

- Before using this tool, install CANN. For details, see [CANN Quick Installation](https://www.hiascend.com/en/cann/download) to install the Ascend NPU driver and CANN software (including the Toolkit and ops packages), and configure the environment variables.

## 3. Installation Methods

### 3.1 One-Click Installation

> **Installation method**: An isolated uv tool environment (the installer runs `uv tool install -U --python 3.11 mindstudio-agent`), **without going through pip**. You can only use uv subcommands for upgrade and uninstall. For details, see [§5.2](#52-selecting-the-upgrade-and-uninstall-command-by-installation-method).

The installer uses a domestic mirror by default for fast installation. If the mirror has not synchronized the latest version, the installer automatically switches to the official PyPI to obtain the latest version.

**Linux/macOS/WSL:**

```shell
curl -LsSf https://raw.atomgit.com/Ascend/msagent/raw/26.2.0/scripts/install.sh | bash
```

**Windows (PowerShell 5.1 or later):**

```powershell
irm https://raw.atomgit.com/Ascend/msagent/raw/26.2.0/scripts/install.ps1 | iex
```

Run the following command. If the msAgent version is displayed, the installation is successful.

```shell
msagent --version
```

> If `msagent: command not found` is prompted, run the command printed at the end of the installer log (usually `source ~/.bashrc`) or reopen the terminal. For details, see [§4.1](#41-what-to-do-if-command-not-found-is-prompted-after-installation).

#### Common Environment Variables (Optional)

| Variable | Description |
| --- | --- |
| MSAGENT_VERSION | Installs a specified version, for example `MSAGENT_VERSION=26.1.0` (the latest version is installed by default). |
| MSAGENT_INDEX | Specifies the PyPI mirror address, for example `MSAGENT_INDEX=https://pypi.org/simple`. |
| MSAGENT_PYTHON | The Python version used by the tool environment (default: 3.11) |
| MSAGENT_NO_MODIFY_PATH | Skips PATH modification when set to `1` (you configure PATH yourself). |
| MSAGENT_YES | Runs automatically without interaction when set to `1` (for CI and unattended scenarios). |

> Note: To force a specific source, set `MSAGENT_INDEX=<index>`.

### 3.2 pip Installation

The uninstall method differs depending on the target location:

- **Method A: Virtual environment + pip**

  ```shell
  python3 -m venv ~/.msagent-venv
  ~/.msagent-venv/bin/pip install mindstudio-agent
  ```

  Upgrade/uninstall: `~/.msagent-venv/bin/pip install -U mindstudio-agent`, `~/.msagent-venv/bin/pip uninstall -y mindstudio-agent`

- **Method B: uv tool installation (recommended over pip)**

  ```shell
  uv tool install -U --python 3.11 mindstudio-agent
  ```

  Upgrade/uninstall is the same as for [one-click installation](#31-one-click-installation). **You cannot uninstall with pip**: `uv tool upgrade mindstudio-agent`, `uv tool uninstall mindstudio-agent`

> `Python >= 3.11` is required. Directly running `pip install` into the system environment easily causes conflicts or triggers the PEP 668 restriction. For details, see item 7 in the [FAQ](../support/faq.md).

### 3.3 Source Code Installation

> **Installation method**: `pip install` into the current Python environment (a mirror container or a virtual environment). For upgrade, run `pip install -U` with a rebuilt wheel. For uninstall, run `pip uninstall mindstudio-agent` in that environment. For details, see [§5.2](#52-selecting-the-upgrade-and-uninstall-command-by-installation-method).

#### 3.3.1 Environment Setup

All source code compilation uses the MindStudio standard build environment.

- For daily development or when using a released image, see the [MindStudio Tool Development Environment Installation Guide](https://atomgit.com/Ascend/msot/blob/26.2.0/docs/en/common/dev_env_setup.md).
- When you need to reproduce the environment from the base operating system or perform source code build verification or unit test verification, you must refer to the [MindStudio Unified Build Image Creation Guide](https://atomgit.com/Ascend/msot/blob/26.2.0/docs/en/common/docker_image_build_guide.md) to build an environment image on site from the openEuler base image.

The source code compilation and unit test commands in the rest of this document are all run in the containers of the specified image or the environment image built on site. The CANN software package version, GCC version, and Python version follow the *Unified Build Image Creation Guide*. This repository does not maintain them separately.

After the image build is complete, you must use the `ctr_in.py` command provided in Chapter 7 of the *Unified Build Image Creation Guide* to start and enter the container in an interactive terminal. Do not use a common `docker run` command to create a container, and do not use `docker exec <container_name> bash -c '<command>'` as a replacement for the interactive environment. Otherwise, the Python, GCC, and CANN environment initialization may be skipped.

#### 3.3.2 Compiling and Installing

1. After entering the interactive container shell opened by `ctr_in.py`, run the following command to clone this repository:

   ```bash
   cd ~
   git clone https://atomgit.com/Ascend/msagent.git -b 26.2.0
   ```

2. Stay in the same interactive container shell opened by `ctr_in.py`, and run the following command in the repository root directory to automatically download dependencies and build the project:

   ```bash
   cd ~/msagent
   python3 build.py
   ```

3. After a successful build, the installation package is generated in the `artifacts/` directory. Run the following command to install it:

   ```shell
   pip install artifacts/mindstudio_agent-{version}-py3-none-any.whl
   ```

   After the installation is complete, if the following information is displayed, the software is installed successfully.

   ```ColdFusion
   Successfully installed mindstudio-agent-{version}
   ```

   The default build installs `uv`. If `uv` is already installed locally, you can run `python3 build.py local` to skip the installation.

#### 3.3.3 Running Unit Tests (Optional)

This step is not required for installation. To verify the basic functions of the code, run the following command in the repository root directory:

```shell
python3 build.py test
```

This command installs `uv` and runs the test cases in `tests/ut` and `tests/skills`. The `uv run` in `scripts/run_ut.sh` automatically prepares the test dependencies. If `uv` is already installed locally, you can run `python3 build.py test local` to skip the installation. If the command returns 0 and no test case fails, the unit tests pass.

## 4. Verifying the Installation

After the installation is complete, run the following commands to verify that the tool is available:

```shell
msagent --version
msagent --help
```

If the commands output the version information and help information, the installation is successful.

### 4.1 What to Do If `command not found` Is Prompted After Installation

The installer only writes the uv tool directory to the shell startup file and does not modify the currently open terminal. Therefore, if you run `msagent` immediately after the installation, `command not found` is prompted.

1. Run the command printed at the end of the installer log (usually `source ~/.bashrc`, but use the startup file actually printed in the log), or reopen the terminal.
2. If the log prompts "add \<tool directory\> to PATH yourself" (for example, when `MSAGENT_NO_MODIFY_PATH=1` is set, the startup file is not writable, or the Windows user-level PATH does not take effect), set it manually:

   ```shell
   export PATH="$(uv tool dir --bin):${PATH}"                       # Takes effect in the current session
   echo "export PATH=\"$(uv tool dir --bin):\$PATH\"" >> ~/.bashrc  # Persists the setting
   ```

   Windows: In PowerShell, use `$env:PATH = "$(uv tool dir --bin);$env:PATH"`. In cmd, use `set PATH=<tool directory>;%PATH%`.
3. Run `msagent --version` again to verify.

> Unless you want to manage PATH yourself, do not set `MSAGENT_NO_MODIFY_PATH=1` (see §3.1 Common Environment Variables).

## 5. Upgrade and Uninstall

`msagent` saves the user configuration and running state in `MSAGENT_HOME` (default: `~/.msagent/`) and does not generate a new `.msagent/` in the working directory.

Automatic migration of the `.msagent` directory of an old project is not provided at the current stage. When you upgrade or switch versions, retain the old directory. Do not delete the configuration directory as an upgrade step.

When you upgrade the version, pay attention to version compatibility. For details, see the [Release Notes](https://atomgit.com/Ascend/release-management/blob/master/MindStudio/26.2.0/release_notes.md).

### 5.1 Checking the Current Installation Method

The upgrade and uninstall commands vary by installation method. Check your installation first:

```shell
uv tool list                # Contains mindstudio-agent → uv tool method (one-click installation belongs to this type)
command -v msagent          # Points to ~/.local/bin → uv tool method. Points to */venv/bin → virtual environment pip
pip show mindstudio-agent   # Has output → this pip environment also has one installed (the default location of source code installation)
```

One-click installation does not leave `mindstudio-agent` in the pip environment. Running `pip uninstall mindstudio-agent` for it only prompts `Skipping mindstudio-agent as it is not installed`, and the uninstall does not take effect.

### 5.2 Selecting the Upgrade and Uninstall Command by Installation Method

| Installation Method | Upgrade | Uninstall |
| --- | --- | --- |
| One-click installation/`uv tool install` (uv tool method) | Rerun the one-click installation command, or run `uv tool upgrade mindstudio-agent`. | uv tool uninstall mindstudio-agent |
| Virtual environment + pip (for example, `~/.msagent-venv`) | `~/.msagent-venv/bin/pip install -U mindstudio-agent` | ~/.msagent-venv/bin/pip uninstall -y mindstudio-agent |
| pip in the current Python environment (source code installation) | `pip install -U mindstudio-agent` after rebuilding the wheel | pip uninstall mindstudio-agent |

The upgrade command for one-click installation is as follows:

```shell
curl -LsSf https://raw.atomgit.com/Ascend/msagent/raw/26.2.0/scripts/install.sh | bash
```

> The uv tool method and the pip method are separate. If both are installed, only one takes effect in PATH (the installer prompts about the shadowing). Confirm with `command -v msagent` as described in section 5.1 before you uninstall.
