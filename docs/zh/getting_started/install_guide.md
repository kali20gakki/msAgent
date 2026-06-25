# msAgent安装指南

## 1. 安装说明

本文面向第一次使用 MindStudio-Agent 的用户，帮助您完成 msAgent 安装。

您可通过以下三种方式进行安装：[在线安装](#31-在线安装)、[离线安装](#32-离线安装)、[源码安装](#33-源码安装)。

## 2. 环境要求

- `Python >= 3.11`

- `glibc >= 2.34`

  用于满足 `msprof-mcp` 中 `trace_processor` 二进制依赖（建议操作系统：`Ubuntu >= 21.10`、`openEuler >= 21.09`，其他操作系统请自行查询）

- 使用本工具前需要安装CANN，具体操作请参见《[CANN 快速安装](https://www.hiascend.com/cann/download)》安装昇腾NPU驱动和CANN软件（包含Toolkit和ops包），并配置环境变量。

## 3. 安装方式

### 3.1 在线安装

若您的设备具备互联网访问能力，可通过一条命令自动完成工具的下载与安装。支持如下方式：

- 方式一：

  使用 PyPI 安装，普通用户建议优先使用 PyPI 安装稳定发布版本。

  ```shell
  pip install -U mindstudio-agent
  ```

  执行如下命令提示msAgent版本即安装成功。

  ```shell
  msagent --version
  ```

- 方式二：

  请参见昇腾社区MindStudio[下载](https://www.hiascend.com/developer/software/mindstudio/download)页面，选择对应的CANN版本，并在安装方式中选择“在线安装”，系统将引导您完成后续操作。

### 3.2 离线安装

对处于企业内网等无外网环境的设备，请先在可联网的机器上下载完整的离线安装包，再将其传输至目标设备进行安装。请参见昇腾社区MindStudio[下载](https://www.hiascend.com/developer/software/mindstudio/download)页面，选择对应的CANN版本，并在安装方式中选择“离线安装”，获取对应的安装包及操作指引。

### 3.3 源码安装

源码安装支持脚本构建安装和手动构建安装方式。脚本构建安装为工具提供的源码构建脚本可一键执行生成安装包；手动构建安装则需要安装uv依赖，通过命令行构建。

#### 3.3.1 方式一：脚本构建安装

1. 克隆本仓库。

   ```shell
   git clone https://gitcode.com/Ascend/msagent.git
   ```

2. 执行编译打包。

   ```shell
   cd msagent
   bash scripts/build_whl.sh
   ```

   适用场景：

   - Linux / macOS
   - Windows + Git Bash
   - Windows + WSL

   编译完成后在`dist`目录下生成 whl 包，名称格式为`mindstudio_agent-{version}-py3-none-any`。其中`version`为版本号。

3. 安装whl包。

   ```shell
   pip install dist/mindstudio_agent-{version}-py3-none-any.whl
   ```

   安装完成后，若显示如下信息，则说明软件安装成功。

   ```ColdFusion
   Successfully installed mindstudio-agent-{version}
   ```

#### 3.3.2 方式二：手动构建安装

1. 克隆本仓库。

   ```shell
   git clone https://gitcode.com/Ascend/msagent.git
   ```

2. 安装依赖。

   ```shell
   pip install uv
   ```

3. 执行编译打包。

   ```shell
   cd msagent
   test -d skills
   uv lock --check
   uv build --wheel --out-dir dist .
   ```

   编译完成后在`dist`目录下生成whl包，名称格式为`mindstudio_agent-{version}-py3-none-any`。其中`version`为版本号。

4. 安装whl包

   ```shell
   pip install dist/mindstudio_agent-{version}-py3-none-any.whl
   ```

   安装完成后，若显示如下信息，则说明软件安装成功。

   ```ColdFusion
   Successfully installed mindstudio-agent-{version}
   ```

## 4. 升级与卸载

`msagent` 会在当前工作目录下生成 `.msagent/` 本地目录，用于保存缓存、会话历史、日志和运行时配置等内容。

- 升级前，先删除当前工作目录下的 `.msagent/` 文件夹，避免旧缓存影响新版本行为。
- 卸载时，如果后续不再使用 `msagent`，也建议一并删除 `.msagent/` 文件夹。

常见操作示例：

- 升级：

```shell
rm -rf .msagent
pip install mindstudio-agent
```

- 卸载：

```shell
rm -rf .msagent
pip uninstall mindstudio-agent
```
