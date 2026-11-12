# 编译与打包

本文档说明 `msAgent` 的统一构建入口 `build.py`，以及其调用的底层构建脚本 `scripts/build_whl.sh`。

## 推荐方式

按照《[msAgent 安装指南 > 源码安装](../install_guide/msagent_install_guide.md#32-源码安装)》章节完成编译和测试环境的搭建。

> **说明：** 环境镜像的构建方法及配套软件版本由 MindStudio 统一镜像制作指南维护，本仓库不重复定义。

推荐优先使用仓库根目录的统一构建入口：

```bash
python3 build.py
```

构建成功后，wheel 会先生成在 `dist/`，再归档到 `artifacts/`。安装命令如下：

```bash
pip install artifacts/mindstudio_agent-<version>-py3-none-any.whl
```

## 构建与测试命令

| 命令 | 作用 |
|---|---|
| `python3 build.py` | 安装 `uv`，构建并归档 wheel。 |
| `python3 build.py local` | 使用本地已安装的 `uv` 构建并归档 wheel。 |
| `python3 build.py test` | 安装 `uv`，运行 `tests/ut` 和 `tests/skills`。 |
| `python3 build.py test local` | 使用本地已安装的 `uv` 运行 `tests/ut` 和 `tests/skills`。 |

`build.py` 的构建流程如下：

1. 非 `local` 模式执行 `python -m pip install uv`
2. 构建模式调用 `scripts/build_whl.sh` 同步构建版本、检查 Python 与 Skills 资源，并构建 wheel
3. 将 `dist/` 中的 whl 复制到 `artifacts/`
4. 测试模式调用 `scripts/run_ut.sh`，由其中的 `uv run` 自动准备测试依赖

`scripts/build_whl.sh` 仍可单独使用，适合需要直接控制其环境变量的场景。它会先校验 `uv.lock`，不一致时在当前工作区自动执行 `uv lock` 更新，再使用 `uv build`；在没有 `uv` 时回退到 `pip wheel`。

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

如果你想附带安装冒烟验证：

```bash
python3 build.py --extra VERIFY_WHEEL_INSTALL=1
```

## 手动构建

如果你不使用脚本，也可以手动执行等价命令：

```bash
# 安装 uv
pip install uv

# 确认 skills 目录存在
test -d skills
# 检查锁文件是否是最新的、是否和当前项目依赖声明一致
uv lock --check
# 构建项目的 wheel 安装包
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

- 统一构建入口：`build.py`
- 底层构建脚本：`scripts/build_whl.sh`
- 项目元数据：`pyproject.toml`
- 默认 Skills 目录：`skills/`
