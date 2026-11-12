# msAgent安装指南

## 1. 安装说明

本文面向第一次使用 MindStudio-Agent 的用户，帮助您完成 msAgent 安装。

推荐使用[一键安装](#31-一键安装)：自动安装 `mindstudio-agent` 最新版本，不与现有环境冲突。

同时支持[源码安装](#32-源码安装)方式。

> 一键安装默认使用 uv 隔离工具环境；若 uv 安装失败并选择虚拟环境回退，则安装到独立 venv。请按实际安装方式卸载，命令见[按安装方式选择升级与卸载命令](#52-按安装方式选择升级与卸载命令)。

## 2. 环境要求

- **无需预先安装 Python**：一键安装会自动下载所需的 Python 版本（`>= 3.11`）。

- `glibc >= 2.34`

  用于满足 `msprof-mcp` 中 `trace_processor` 二进制依赖（建议操作系统：`Ubuntu >= 21.10`、`openEuler >= 21.09`，其他操作系统请自行查询）

- 在昇腾 NPU 环境中使用本工具前，需要安装昇腾 NPU 驱动和 CANN 软件（包含 Toolkit 和 ops 包），并配置环境变量。具体操作请参见《[CANN 快速安装](https://www.hiascend.com/zh/cann/download)》。
- 在 Windows 上，安装并启动本工具无需预先安装昇腾 NPU 驱动或 CANN。

## 3. 安装方式

### 3.1 一键安装

> **默认安装方式**：uv 隔离工具环境（安装器执行 `uv tool install`），不经过 pip。若 uv 安装失败，安装器通常会询问是否使用虚拟环境回退；设置 `MSAGENT_YES=1` 时会自动选择回退，设置 `MSAGENT_NO_FALLBACK=1` 时则跳过回退。回退会安装到独立 venv，默认路径为 `~\.msagent-venv`（Linux/macOS 为 `~/.msagent-venv`），也可通过 `MSAGENT_FALLBACK_VENV` 指定。升级、卸载命令见[按安装方式选择升级与卸载命令](#52-按安装方式选择升级与卸载命令)。

安装器默认使用国内镜像（安装快）；若镜像尚未同步最新版本，会自动改用官方 PyPI 获取最新版。

**Linux / macOS / WSL：**

```shell
curl -LsSf https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.sh | bash
```

**Windows（PowerShell 5.1 及以上）：**

```powershell
irm https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.ps1 | iex
```

执行如下命令提示msAgent版本即安装成功。

```shell
msagent --version
```

> 若提示 `msagent: command not found`，按安装器日志末尾打印的命令执行（通常是 `source ~/.bashrc`）或重开终端，见[安装后提示 command not found 怎么办](#41-安装后提示-command-not-found-怎么办)。

#### 常用环境变量（可选）

| 变量 | 说明 |
| --- | --- |
| `MSAGENT_VERSION` | 安装指定版本，如 `MSAGENT_VERSION=26.1.0`（默认安装最新版） |
| `MSAGENT_INDEX` | 指定 PyPI 镜像地址，如 `MSAGENT_INDEX=https://pypi.org/simple` |
| `MSAGENT_PYTHON` | 工具环境使用的 Python 版本（默认 `3.11`） |
| `MSAGENT_NO_MODIFY_PATH` | 设为 `1` 时跳过 PATH 修改（由用户自行配置 PATH） |
| `MSAGENT_NO_FALLBACK` | 设为 `1` 时，uv 安装失败后不询问是否回退到虚拟环境 |
| `MSAGENT_FALLBACK_VENV` | 指定虚拟环境回退的安装路径（默认 Linux/macOS 为 `~/.msagent-venv`，Windows 为 `%USERPROFILE%\.msagent-venv`） |
| `MSAGENT_YES` | 设为 `1` 时免交互自动执行（CI/无人值守场景） |

> 提示：如需强制指定某个源，可设置 `MSAGENT_INDEX=<index>`。

### 3.2 源码安装

> **安装方式**：`pip install` 到当前 Python 环境（镜像容器或虚拟环境）；升级用重新构建的 wheel 执行 `pip install -U`，卸载用该环境的 `pip uninstall mindstudio-agent`，见[按安装方式选择升级与卸载命令](#52-按安装方式选择升级与卸载命令)。

#### 3.2.1 环境准备

源码编译统一使用 MindStudio 标准构建环境。

- 日常开发或使用已发布镜像，请参考《[MindStudio工具开发环境安装指导](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/dev_env_setup.md)》。
- 需要从基础操作系统复现环境、执行源码构建验证或单元测试验证时，必须参考《[MindStudio统一构建镜像制作指南](https://gitcode.com/Ascend/msot/blob/master/docs/zh/common/docker_image_build_guide.md)》，从openEuler基础镜像现场构建环境镜像。

本文档后续的源码编译和单元测试命令，均在上述指定镜像容器或现场构建的环境镜像的容器中执行，CANN 软件包版本、GCC 版本和 Python 版本以统一镜像制作指南为准，本仓库不重复维护。

镜像构建完成后，必须使用统一镜像制作指南第 7 章给出的 `ctr_in.py` 命令，在交互式终端中启动并进入容器。不得使用普通 `docker run` 创建容器，也不得使用 `docker exec <容器名> bash -c '<命令>'` 替代交互式环境；否则可能跳过 Python、GCC 和 CANN 环境初始化。

#### 3.2.2 编译并安装

1. 进入 `ctr_in.py` 打开的交互式容器 Shell 后，执行如下命令克隆本仓库：

   ```bash
   cd ~
   git clone https://gitcode.com/Ascend/msagent.git -b master
   ```

2. 保持在 `ctr_in.py` 打开的同一个交互式容器 Shell 中，在仓库根目录执行以下命令，自动完成依赖下载与构建：

   ```bash
   cd ~/msagent
   python3 build.py
   ```

3. 构建成功后，安装包将生成在 `artifacts/` 目录下。使用如下命令进行安装：

   ```shell
   pip install artifacts/mindstudio_agent-{version}-py3-none-any.whl
   ```

   安装完成后，若显示如下信息，则说明软件安装成功。

   ```ColdFusion
   Successfully installed mindstudio-agent-{version}
   ```

   默认构建会安装 `uv`。如本地已安装 `uv`，可使用 `python3 build.py local` 跳过安装。

#### 3.2.3 执行单元测试（可选）

此步骤非安装必需。如需验证代码基本功能，可在仓库根目录执行：

```shell
python3 build.py test
```

该命令会安装 `uv` 并运行 `tests/ut` 和 `tests/skills` 下的用例，`scripts/run_ut.sh` 中的 `uv run` 会自动准备测试依赖。如本地已安装 `uv`，可执行 `python3 build.py test local` 跳过安装。命令返回码为 0，且测试用例均无失败，表示单元测试通过。

## 4. 验证安装

安装完成后，执行以下命令验证工具是否可用：

```shell
msagent --version
msagent --help
```

命令能够输出版本信息和帮助信息，表示安装成功。

### 4.1 安装后提示 command not found 怎么办

安装器只把 uv 工具目录写入 shell 启动文件，不会改动当前已打开的终端，因此装上后立刻执行 `msagent` 会提示 `command not found`。

1. 执行安装器日志末尾打印的命令（通常是 `source ~/.bashrc`，以日志实际打印的启动文件为准），或重开终端。
2. 日志提示"请自行把 \<工具目录\> 加入 PATH"时（设置了 `MSAGENT_NO_MODIFY_PATH=1`、启动文件不可写、Windows 用户级 PATH 未生效等），手动设置：

   ```shell
   export PATH="$(uv tool dir --bin):${PATH}"                       # 当前会话生效
   echo "export PATH=\"$(uv tool dir --bin):\$PATH\"" >> ~/.bashrc  # 持久化
   ```

   Windows：PowerShell 用 `$env:PATH = "$(uv tool dir --bin);$env:PATH"`，cmd 用 `set PATH=<工具目录>;%PATH%`。
3. `msagent --version` 复验。

> 除非要自行管理 PATH，否则不要设置 `MSAGENT_NO_MODIFY_PATH=1`（见 §3.1 常用环境变量）。

### 4.2 Windows 一键安装失败后如何清理

安装失败后，可在 PowerShell 中执行以下命令移除 msagent 的 uv 工具环境（如果尚未安装成功，提示未找到工具可忽略）：

```powershell
uv tool uninstall mindstudio-agent
```

只有在 uv 安装失败后选择了虚拟环境回退时，才需要卸载该 venv 中的版本。默认路径如下；若设置过 `MSAGENT_FALLBACK_VENV`，请将命令中的路径替换为当时指定的实际路径：

```powershell
& "$env:USERPROFILE\.msagent-venv\Scripts\pip.exe" uninstall -y mindstudio-agent
```

以上操作不会卸载 uv 或 Python，也不会删除 `~\.msagent` 中的用户配置和运行数据。除非确认该虚拟环境只用于 msagent，否则不要直接删除整个目录。

## 5. 升级与卸载

`msagent` 将用户配置和运行状态保存在 `MSAGENT_HOME`（默认 `~/.msagent/`），不会在工作目录生成新的 `.msagent/`。

当前阶段尚未提供旧项目 `.msagent` 的自动迁移。升级或切换版本时请保留旧目录，不要将删除配置目录作为升级步骤。

升级版本时需要关注版本配套关系，请参见《[版本说明](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/master/release_notes.md)》。

### 5.1 先确认当前是哪种安装方式

升级、卸载命令随安装方式不同而不同，先自检：

```shell
uv tool list                # 含 mindstudio-agent → uv 工具方式（一键安装默认使用此方式）
command -v msagent          # 指向 ~/.local/bin → uv 工具方式；指向 */venv/bin → 虚拟环境 pip
pip show mindstudio-agent   # 有输出 → 该 pip 环境还装有一份（源码安装的默认落点）
```

一键安装默认使用 uv；只有 uv 安装失败且用户选择虚拟环境回退时，才会通过 pip 将其安装到 venv。对 uv 工具环境执行 `pip uninstall mindstudio-agent` 不会卸载该版本。

### 5.2 按安装方式选择升级与卸载命令

| 安装方式 | 升级 | 卸载 |
| --- | --- | --- |
| 一键安装默认方式 / `uv tool install`（uv 工具方式） | 重跑一键安装命令，或 `uv tool upgrade mindstudio-agent` | `uv tool uninstall mindstudio-agent` |
| 一键安装的虚拟环境回退 | 使用该 venv 的 pip 执行 `install -U mindstudio-agent` | 使用该 venv 的 pip 执行 `uninstall -y mindstudio-agent` |
| 虚拟环境 + pip（如 `~/.msagent-venv`） | `~/.msagent-venv/bin/pip install -U mindstudio-agent` | `~/.msagent-venv/bin/pip uninstall -y mindstudio-agent` |
| 当前 Python 环境 pip（源码安装） | 重新构建 wheel 后 `pip install -U mindstudio-agent` | `pip uninstall mindstudio-agent` |

一键安装的升级命令：

```shell
curl -LsSf https://raw.gitcode.com/Ascend/msagent/raw/master/scripts/install.sh | bash
```

> uv 工具方式与 pip 方式互不相通，两种都装过时 PATH 上只有一个生效（安装器会提示遮蔽），按 5.1 的 `command -v msagent` 确认后再卸载。
