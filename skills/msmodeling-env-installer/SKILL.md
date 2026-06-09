---
name: msmodeling-env-installer
description: Install and verify the msmodeling development environment. Use when the user explicitly asks to install msmodeling dependencies, set up this repository, create `myenv` with `uv`, install this repository's `requirements.txt`, set project `PYTHONPATH`, or configure `HF_ENDPOINT`; if the user only says to install an environment, ask whether they mean msmodeling dependencies before proceeding.
version: 0.2.2
source: local-session-analysis
---

# msModeling 环境安装器

## 适用场景

直接使用本 skill 的场景：

- 用户明确要求安装 msmodeling 环境依赖、初始化 msmodeling 开发环境或按 msmodeling README 配置环境。
- 用户明确要求在当前 msmodeling 仓库中创建 `myenv`。
- 用户明确要求安装当前仓库的 `requirements.txt`。
- 用户明确要求检查当前 msmodeling Python 依赖。
- 用户明确要求为当前 msmodeling 会话配置 `PYTHONPATH`。
- 用户明确要求配置 Hugging Face 镜像 `HF_ENDPOINT=https://hf-mirror.com`。
- 用户已有 Python 环境，并明确希望为 msmodeling 执行 fallback 依赖安装。

需要先确认的场景：

- 用户只说“安装环境”“安装依赖”“配置环境”“初始化环境”，但没有明确说明是 msmodeling、本仓库、`requirements.txt`、`myenv` 或 `uv`。
- 当前仓库后续可能存在其他环境安装工具时，不能默认选择本 skill。

遇到上述模糊请求时，先向用户确认：

```text
你是要安装 msmodeling 当前仓库的环境依赖吗？确认后我会使用 msmodeling-env-installer 执行。
```

只有用户确认后，才继续执行本 skill 的安装流程。

## 默认策略

优先使用 README 推荐的 `uv` 流程创建独立虚拟环境，并在安装后执行依赖一致性检查。

默认值和约束：

- 默认虚拟环境名为 `myenv`。
- Python 最低版本为 `3.10`。
- 默认使用当前机器检测到的 Python 主次版本创建虚拟环境，避免 `uv` 额外下载 Python；只有用户显式要求或本机不可用时才使用 README 示例中的 `3.13`。
- 默认使用中科大 PyPI 镜像：`https://mirrors.ustc.edu.cn/pypi/web/simple`。
- 默认不覆盖已有环境；如果 `myenv` 已存在，需说明复用或重建的影响。
- 默认不持久化系统环境变量，只设置当前 shell 会话或给出可执行命令。
- 涉及网络安装时，先展示将执行的命令，并按当前工具权限请求用户授权。

## 环境选择策略

根据当前运行环境选择自动化脚本：

| 当前环境 | 优先命令 |
|:---|:---|
| Windows PowerShell | `.\\skills\\msmodeling-env-installer\\scripts\\install-current-project-deps.ps1` |
| Linux/macOS Bash | `bash ./skills/msmodeling-env-installer/scripts/install-current-project-deps.sh` |
| WSL/Git Bash | `bash ./skills/msmodeling-env-installer/scripts/install-current-project-deps.sh` |

如果当前 shell 与操作系统不匹配，优先选择当前 shell 可直接执行的脚本。例如在 Windows 的 Git Bash 中使用 `.sh`，在 Windows PowerShell 中使用 `.ps1`。

## 工作流程

1. 确认用户意图明确指向 msmodeling 环境依赖；如果只是泛化的“安装环境”，先询问是否安装 msmodeling 当前仓库的环境依赖。
2. 检查当前目录是否为 msmodeling 仓库根目录，至少包含 `README.md` 和 `requirements.txt`。
   - 若当前目录不是 msmodeling 仓库根目录，先检查当前目录下是否已有 `msmodeling/README.md` 和 `msmodeling/requirements.txt`；若存在，进入 `msmodeling` 后继续。
   - 若当前目录及其下级 `msmodeling/` 都不存在仓库，说明将创建 `msmodeling` 目录并从官方仓库克隆；获得授权后直接执行：
     ```bash
     git clone https://gitcode.com/Ascend/msmodeling.git
     cd msmodeling
     ```
   - 克隆后必须重新确认 `README.md` 和 `requirements.txt` 存在，再继续环境安装。
3. 检测 `python`、`python3` 或 Windows `py -3`，确认 Python 版本为 `3.10+`。
4. 检查 `uv` 是否可用；缺失时用 `python -m pip install uv -i https://mirrors.ustc.edu.cn/pypi/web/simple` 安装。
5. 安装或调用 `uv` 后，解析真实 `uv` 可执行路径，不能假设当前 shell 的 `PATH` 已刷新。
6. 选择安装路径：
   - 新建 `myenv`：使用 `uv venv --python <detected-python-version> myenv`。
   - 已有环境 fallback：先检查当前环境不包含 `torch_npu`、`torch-npu`、`cudatoolkit`，再执行 `pip install -r requirements.txt`。
7. 安装依赖：优先执行 `uv pip install --python <venv-python> -r requirements.txt -i <mirror>`。
8. 按需设置当前会话环境变量：
   - `PYTHONPATH` 指向 msmodeling 仓库根目录。
   - `HF_ENDPOINT` 设置为 `https://hf-mirror.com`。
9. 执行依赖检查：
   - `uv pip check --python <venv-python>`。
   - 已有环境 fallback 使用 `python -m pip check`。
10. 向用户报告激活命令、安装结果、验证结果和后续建议。

## 命令模板

### 新建环境

```bash
pip install uv -i https://mirrors.ustc.edu.cn/pypi/web/simple
uv venv --python <detected-python-version> myenv
uv pip install --python <venv-python> -r requirements.txt -i https://mirrors.ustc.edu.cn/pypi/web/simple
```

激活命令：

| 操作系统 | 命令 |
|:---|:---|
| Linux/macOS/WSL/Git Bash | `source myenv/bin/activate` |
| Windows PowerShell | `myenv\Scripts\Activate.ps1` |
| Windows cmd | `myenv\Scripts\activate.bat` |

### 已有环境 fallback

使用该路径前必须先检查当前环境：

```bash
python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch_npu') else 1)"
python -m pip show torch-npu
python -m pip show torch_npu
python -m pip show cudatoolkit
```

如果任一检查显示包存在，不要默认继续 fallback。建议用户新建 `myenv`，或让用户明确确认继续使用该环境。

```bash
python -m pip install -r requirements.txt
python -m pip check
```

### 环境变量

Linux/macOS/WSL/Git Bash：

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export HF_ENDPOINT="https://hf-mirror.com"
```

Windows PowerShell：

```powershell
$env:PYTHONPATH = "$(Get-Location);$env:PYTHONPATH"
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

## 自动化脚本

仓库内脚本：

| 脚本 | 适用环境 |
|:---|:---|
| `scripts/install-current-project-deps.ps1` | Windows PowerShell |
| `scripts/install-current-project-deps.sh` | Linux/macOS/WSL/Git Bash |

脚本会先确认当前目录是否为 msmodeling 仓库根目录；如果当前目录下已有 `msmodeling/` 仓库则自动进入；如果两者都不存在，则按 `optix-deploy` 的方式从 `https://gitcode.com/Ascend/msmodeling.git` 克隆后再继续安装。

PowerShell 参数：

| 参数 | 说明 |
|:---|:---|
| `-EnvName` | 虚拟环境目录名，默认 `myenv` |
| `-PythonVersion` | `uv venv` 使用的 Python 版本，默认使用检测到的本机 Python 主次版本 |
| `-UseExistingEnv` | 跳过新建 venv，使用已有环境安装依赖 |
| `-SetProjectEnv` | 为当前 PowerShell 会话设置 `PYTHONPATH` |
| `-UseHFMirror` | 为当前 PowerShell 会话设置 `HF_ENDPOINT` |
| `-UseProjectUvCache` | 默认启用，将 `UV_CACHE_DIR` 指向仓库内 `.uv-cache` |

Bash 参数：

| 参数 | 说明 |
|:---|:---|
| `--env-name <name>` | 虚拟环境目录名，默认 `myenv` |
| `--python-version <version>` | `uv venv` 使用的 Python 版本，默认使用检测到的本机 Python 主次版本 |
| `--use-existing-env` | 跳过新建 venv，使用已有环境安装依赖 |
| `--set-project-env` | 输出并为当前脚本进程设置 `PYTHONPATH` |
| `--use-hf-mirror` | 输出并为当前脚本进程设置 `HF_ENDPOINT` |
| `--no-project-uv-cache` | 不设置仓库内 `.uv-cache` 作为 `UV_CACHE_DIR` |

示例：

```powershell
.\skills\msmodeling-env-installer\scripts\install-current-project-deps.ps1
.\skills\msmodeling-env-installer\scripts\install-current-project-deps.ps1 -SetProjectEnv -UseHFMirror
.\skills\msmodeling-env-installer\scripts\install-current-project-deps.ps1 -UseExistingEnv
```

```bash
bash ./skills/msmodeling-env-installer/scripts/install-current-project-deps.sh
bash ./skills/msmodeling-env-installer/scripts/install-current-project-deps.sh --set-project-env --use-hf-mirror
bash ./skills/msmodeling-env-installer/scripts/install-current-project-deps.sh --use-existing-env
```

## 安全规则

- 不修改 `requirements.txt`、README 或项目源码。
- 不在未经确认时执行网络安装、删除环境或覆盖已有虚拟环境。
- 用户请求不明确时，不默认执行 msmodeling 环境安装，必须先确认。
- 不默认持久化系统级环境变量。
- fallback 安装前必须检查 `torch_npu`、`torch-npu` 和 `cudatoolkit`。
- Windows 下如遇 PyTorch 兼容问题，提醒用户 README 中的 Windows PyTorch 版本风险；未安装 PyTorch 时优先建议 `2.8` 或更早版本。
- 失败时保留完整命令、关键错误、失败阶段和最小修复建议。

## 完成标准

- 用户意图已确认指向 msmodeling 环境依赖。
- 仓库根目录、Python 版本、`uv` 可用性检查完成。
- 依赖安装流程完成，或失败原因已明确分类。
- `pip check` 或 `uv pip check --python <venv-python>` 已执行并报告结果。
- 输出用户后续可直接执行的激活命令。
- 明确说明当前会话是否设置了 `PYTHONPATH` 或 `HF_ENDPOINT`。
