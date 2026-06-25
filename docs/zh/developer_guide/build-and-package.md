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

1. 根据 `WHL_VERSION` 或 `version.info` 同步本次构建使用的版本号，并刷新 `pyproject.toml` 与 `uv.lock` 的根包版本
2. 解析 `pyproject.toml` 中的项目名、Python 最低版本和入口模块
3. 检查本机 Python 版本是否满足要求
4. 识别仓库根目录下的 `skills/` 资源目录
5. 校验 Skills 资源目录存在且非空
6. 如果存在 `uv.lock`，先执行 `uv lock --check`
7. 优先使用 `uv build --wheel --out-dir dist` 构建 wheel
8. 如果本机没有 `uv`，回退到 `python -m build`

## 常用构建参数

| 环境变量 | 默认值                                          | 说明 |
|---|----------------------------------------------|---|
| `DIST_DIR` | `dist/`                                      | 输出目录。 |
| `SKILLS_PATH` | `skills`                                     | 指定要打包的 Skills 目录，默认使用仓库根目录下的 `skills/`。 |
| `WHL_VERSION` | `version.info` 中的 `Version` | 指定 wheel 版本号；如果不设置，则使用 `version.info` 中的 `Version`。 |
| `VERIFY_WHEEL_INSTALL` | `0`                                          | 是否在临时虚拟环境中做 wheel 安装冒烟验证。 |
| `PYTHON_BIN` | 自动探测                                         | 指定构建使用的 Python。 |
| `SMOKE_IMPORT_MODULE` | 自动推导                                         | 冒烟验证时导入的模块。 |
| `SMOKE_RESOURCE_PATH` | `resources/configs/default/config.mcp.json`  | 冒烟验证时检查是否被打进 wheel 的资源文件。 |
| `SMOKE_SKILL_PATH` | `resources/configs/default/skills/README.md` | 冒烟验证时检查是否被打进 wheel 的 Skills 资源文件。 |

如果你希望按主仓库当前锁定的 Skills 版本构建，而不是同步上游最新提交，可以这样执行：

如果你想附带安装冒烟验证：

```bash
VERIFY_WHEEL_INSTALL=1 bash scripts/build_whl.sh
```

## 手动构建

如果你不使用脚本，也可以手动执行等价命令：

```bash
# 安裝 uv
pip install uv

# 确认 skills 目录存在
test -d skills
# 检查锁文件是否是最新的、是否和当前项目依赖声明一致
uv lock --check
# 构建项目的 wheel 安装包。
uv build --wheel --out-dir dist .
```

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
- 默认 Skills 目录：`skills/`
