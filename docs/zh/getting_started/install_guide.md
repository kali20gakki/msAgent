# 入门安装指南

本文面向第一次使用 MindStudio-Agent 的用户，帮助你完成基础环境准备、安装和最小启动验证。

## 环境要求

- Python `3.11+`
- 推荐使用 `uv` 管理源码运行环境
- 至少准备一个可用的 LLM API Key
- glibc `>= 2.34`，用于满足 `msprof-mcp` 中 `trace_processor` 二进制依赖

## PyPI 安装

普通用户建议优先使用 PyPI 安装稳定发布版本：

```bash
pip install -U mindstudio-agent
msagent --version # 提示版本即安装成功
msagent
```

## 源码运行

如果你需要跟踪最新源码、参与开发，或同步最新内置 Skills，可以使用源码运行方式：

```bash
git clone https://gitcode.com/Ascend/msagent.git
cd msagent
uv sync
uv run msagent --version
uv run msagent
```

源码运行时，后续示例中的 `msagent` 可以替换为 `uv run msagent`。

## 源码安装

如果你需要本地构建并发布软件包，请参考[《构建打包指南》](../developer_guide/build-and-package.md)
