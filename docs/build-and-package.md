# 编译与打包

本文档基于当前仓库中的 `scripts/build_whl.sh` 与 `pyproject.toml` 说明 `msAgent` 的 wheel 构建方式。

## 推荐方式

推荐优先使用仓库自带脚本：

```bash
bash scripts/build_whl.sh
```

适用场景：

- Linux / macOS
- Windows + Git Bash
- Windows + WSL

## 构建脚本当前行为

`scripts/build_whl.sh` 会按当前实现执行以下步骤：

1. 解析 `pyproject.toml` 中的项目名、Python 最低版本和入口模块
2. 检查本机 Python 版本是否满足要求
3. 识别 `resources/configs/default/skills` 子模块
4. 默认同步 Skills 子模块到上游最新提交
5. 如果存在 `uv.lock`，先执行 `uv lock --check`
6. 优先使用 `uv build --wheel --out-dir dist` 构建 wheel
7. 如果本机没有 `uv`，回退到 `python -m build`

## 常用构建参数

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DIST_DIR` | `dist/` | 输出目录。 |
| `SYNC_SKILLS_REMOTE` | `1` | 是否在打包前把 Skills 子模块同步到上游最新提交。 |
| `VERIFY_WHEEL_INSTALL` | `0` | 是否在临时虚拟环境中做 wheel 安装冒烟验证。 |
| `PYTHON_BIN` | 自动探测 | 指定构建使用的 Python。 |
| `SMOKE_IMPORT_MODULE` | 自动推导 | 冒烟验证时导入的模块。 |
| `SMOKE_RESOURCE_PATH` | `resources/configs/default/config.mcp.json` | 冒烟验证时检查是否被打进 wheel 的资源文件。 |

如果你希望按主仓库当前锁定的 Skills 版本构建，而不是同步上游最新提交，可以这样执行：

```bash
SYNC_SKILLS_REMOTE=0 bash scripts/build_whl.sh
```

如果你想附带安装冒烟验证：

```bash
VERIFY_WHEEL_INSTALL=1 bash scripts/build_whl.sh
```

## 手动构建

如果你不使用脚本，也可以手动执行等价命令：

```bash
git submodule sync --recursive
git submodule update --init --recursive --remote --force --depth 1 resources/configs/default/skills
uv lock --check
uv build --wheel --out-dir dist .
```

如果希望按仓库里固定的 Skills 提交构建，可去掉 `--remote`。

## 安装构建结果

构建完成后，`dist/` 目录会生成 `mindstudio_agent-*.whl`，可直接安装：

```bash
pip install dist/mindstudio_agent-<version>-py3-none-any.whl
```

Windows PowerShell / CMD 也可以直接安装对应的 wheel 文件，例如：

```powershell
pip install .\dist\mindstudio_agent-<version>-py3-none-any.whl
```

## 相关文件

- 构建脚本：`scripts/build_whl.sh`
- 项目元数据：`pyproject.toml`
- 默认 Skills 子模块：`resources/configs/default/skills/`
