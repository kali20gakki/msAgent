# Release Guide

本文档说明如何通过推送 Git tag，自动触发 GitHub Actions 构建 `mindstudio-agent`，并发布到 GitHub Release 和 PyPI。

当前仓库相关文件：

- `.github/workflows/publish-release.yml`
- `.github/workflows/build-wheel.yml`
- `pyproject.toml`
- `scripts/build_whl.sh`

## 当前发布流程

当你向远端推送符合 `v*` 规则的 tag，例如 `v0.1.1` 时，会自动触发 `Publish Release` workflow。

这个 workflow 会按顺序执行：

1. 校验 tag 是否与 `pyproject.toml` 中的 `project.version` 一致。
2. 执行单元测试。
3. 构建 wheel。
4. 在 Ubuntu、macOS、Windows 上做安装 smoke test。
5. 构建 sdist。
6. 发布 GitHub Release。
7. 发布到 PyPI。

## 1. 前置条件

### 1.1 GitHub Release

仓库需要开启 GitHub Actions，并且你有权限向远端推送 tag。

### 1.2 PyPI 发布

当前仓库的 tag 发布流程默认会执行 PyPI 发布，不是可选步骤。

`publish-pypi` job 使用的是 PyPI Trusted Publishing，所以需要提前在 PyPI 项目后台配置 Trusted Publisher。

当前仓库建议填写：

- Owner: `kali20gakki`
- Repository name: `msAgent`
- Workflow name: `publish-release.yml`
- Environment name: `pypi`

如果这里没有提前配置好，tag 推送后 `publish-pypi` 会失败。

## 2. 修改版本号

发版前，先修改 `pyproject.toml` 中的版本号：

```toml
[project]
version = "0.1.0"
```

例如准备发布 `0.1.1`，就改为：

```toml
[project]
version = "0.1.1"
```

注意：

- tag 必须与版本号严格对应
- 如果版本号是 `0.1.1`，tag 就必须是 `v0.1.1`

## 3. 本地自检

建议在打 tag 前，先本地执行一次最小自检：

```bash
uv lock --check
uv sync --dev
uv run pytest -q
bash scripts/build_whl.sh
python -m build --sdist --outdir dist
```

如果你在 Windows 且没有 `bash` 环境，可以至少确认：

- 测试可以通过
- wheel 能正常构建
- `resources/configs/default/skills` 子模块内容已拉取完整

## 4. 提交版本变更

版本号修改完成后，先提交并推送代码：

```bash
git add pyproject.toml
git commit -m "chore: release v0.1.1"
git push origin main
```

建议先推代码，再打 tag，这样发布点会更清晰。

## 5. 创建并推送 tag

确认 `main` 已经是你要发布的 commit 后，创建并推送 tag：

```bash
git tag -a v0.1.1 -m "Release v0.1.1"
git push origin v0.1.1
```

推送完成后，GitHub 会自动触发 `.github/workflows/publish-release.yml`。

## 6. 自动发布流程说明

### 6.1 `validate-tag`

读取 `pyproject.toml` 中的版本号，并校验：

```text
GITHUB_REF_NAME == "v" + project.version
```

如果不一致，workflow 会直接失败。

### 6.2 `build-wheel`

复用 `.github/workflows/build-wheel.yml`，主要包括：

- `uv sync --dev`
- `uv run pytest -q`
- `bash scripts/build_whl.sh`
- 上传 wheel artifact
- 三个平台安装 smoke test

### 6.3 `build-sdist`

构建源码包，并校验关键文件和 skills 资源是否被正确打包。

### 6.4 `publish-github-release`

下载 wheel 和 sdist，自动创建或更新 GitHub Release，并上传构建产物。

如果 tag 名里包含连字符，例如 `v0.1.1-rc1`，GitHub Release 会被标记为 prerelease。

### 6.5 `publish-pypi`

下载 wheel 和 sdist，通过 `pypa/gh-action-pypi-publish@release/v1` 发布到 PyPI。

## 7. 手动触发 wheel 构建

如果你只是想先验证构建，而不是正式发版，可以手动运行 `Build Wheel` workflow。

这个 workflow 支持 `workflow_dispatch`，会执行：

- 单元测试
- wheel 构建
- 跨平台安装 smoke test

但它不会创建 GitHub Release，也不会发布到 PyPI。

## 8. 发布完成后的结果

发布成功后，你应该能看到：

1. GitHub Actions 中 `Publish Release` 全部 job 成功。
2. GitHub Releases 页面出现对应版本，例如 `v0.1.1`。
3. PyPI 上出现对应版本，例如 `mindstudio-agent 0.1.1`。

你也可以本地验证：

```bash
pip install -U mindstudio-agent==0.1.1
msagent --version
```

## 9. 常见失败原因

### 9.1 tag 和版本号不一致

例如：

- `pyproject.toml` 里还是 `0.1.0`
- 但推了 `v0.1.1`

这种情况会在 `validate-tag` 阶段直接失败。

### 9.2 PyPI Trusted Publisher 未配置

这种情况通常会在 `publish-pypi` 阶段失败。

### 9.3 同版本已经发布过

PyPI 不允许覆盖上传同一个版本号，遇到这种情况需要递增版本号后重新发版。

### 9.4 子模块内容缺失

当前打包流程要求 `resources/configs/default/skills` 有有效内容。

如果缺失，可以先执行：

```bash
git submodule sync --recursive
git submodule update --init --recursive resources/configs/default/skills
```
