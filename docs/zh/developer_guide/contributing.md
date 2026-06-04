# 贡献指南

感谢你对 MindStudio-Agent（`msagent`）的关注。无论你是修复 Bug、补充文档、新增 Skill，还是扩展 Agent 与框架能力，都欢迎通过 Issue 或 Pull Request 参与共建。

本文档汇总常见贡献路径、开发流程与相关参考文档。更完整的文档导航见 [中文文档首页](../../index.md)。

## 你可以贡献什么

MindStudio-Agent 是一个面向 Ascend NPU 场景的 Agent 工作台，核心由 CLI、配置系统、MCP 工具、内置 Skills 与领域 Agent 组成。常见贡献类型包括：

| 贡献类型 | 说明 | 推荐阅读 |
|---|---|---|
| **问题反馈** | Bug、功能需求、文档错误或体验问题 | [FAQ](../user_guide/faq.md)、[GitCode Issues](https://gitcode.com/Ascend/msagent/issues) |
| **文档改进** | 安装说明、使用指南、Agent 说明、架构文档 | [入门安装指南](../getting_started/install_guide.md)、[ReadTheDocs 本地验证](readthedocs-local-build.md) |
| **Skills 扩展** | 新增或完善领域诊断 SOP、脚本与触发说明 | [配置与扩展](../user_guide/configuration-and-extension.md)、[skills/README.md](../../../skills/README.md) |
| **Agent / 配置** | 调整 Agent YAML、Prompt、Tool/Skill 过滤规则 | [Agent / Tool / Skill 过滤规则](../user_guide/agent-tool-skill-filter-rules.md) |
| **框架代码** | CLI、中间件、MCP 集成、配置加载等核心能力 | [架构概览](arch_overview.md) |
| **测试与构建** | 单元测试、集成测试、wheel 构建验证 | [编译与打包](build-and-package.md) |

当前内置 Agent 及其领域定位如下，贡献前可先了解各自边界：

- [Hermes](../agent_guide/Hermes.md)：性能调优
- [Accuracy](../agent_guide/Accuracy.md)：精度调优
- [Zephyr](../agent_guide/Zephyr.md)：模型量化
- [Icarus](../agent_guide/Icarus.md)：算子调优
- [Minos](../agent_guide/Minos.md)：文档体验与代码审查

## 开始之前

### 环境要求

贡献代码或做本地验证前，请先确认环境满足要求：

- Python `3.11+`
- 推荐使用 [uv](https://docs.astral.sh/uv/) 管理依赖
- 至少准备一个可用的 LLM API Key（用于交互验证）

详细说明见 [入门安装指南](../getting_started/install_guide.md) 与 [版本与兼容性](version-and-compatibility.md)。

### 克隆与源码运行

```bash
git clone https://gitcode.com/Ascend/msagent.git
cd msagent
uv sync --dev
uv run msagent --version
```

源码运行时，命令行中的 `msagent` 可替换为 `uv run msagent`。首次启动与模型配置方式见 [快速入门指导](../getting_started/quick_start.md)。

### 建议阅读顺序

1. [架构概览](arch_overview.md)：了解 CLI、Agent Factory、Tools、Skills、MCP 与中间件的分层关系
2. [配置与扩展](../user_guide/configuration-and-extension.md)：理解 `.msagent/` 本地配置、MCP 与 Skills 加载顺序
3. [Agent / Tool / Skill 过滤规则](../user_guide/agent-tool-skill-filter-rules.md)：修改 Agent 能力边界前必读
4. 与你改动相关的 Agent 指南或用户指南

## 开发流程

### 1. 创建 Issue 或认领任务

- 提交 Bug 或功能建议：[GitCode Issues](https://gitcode.com/Ascend/msagent/issues)
- 较大改动建议先开 Issue 讨论方案，避免与维护者方向冲突
- 安全问题请遵循 [安全声明](../legal/SECURITY.md)，避免公开披露可利用细节

### 2. 创建分支并开发

从最新主分支切出功能分支，保持提交粒度清晰、说明准确。

开发过程中如需理解运行时行为，可参考：

- 会话命令与快捷键：[使用指南](../user_guide/usemap.md)
- 上下文压缩机制：[上下文压缩使用指南](../user_guide/context-compaction-guide.md)
- 重试与超时配置：[Retry Middleware 使用指南](../user_guide/retry-middleware-guide.md)

### 3. 本地自检

提交 PR 前，建议至少完成以下检查：

```bash
# 安装开发依赖
uv sync --dev

# 运行测试
uv run pytest -q

# 同步 lockfile 校验（修改 pyproject.toml 后必做）
uv lock --check

# 构建 wheel（涉及打包或 Skills 变更时建议执行）
bash scripts/build_whl.sh
```

如需验证 wheel 安装，可使用：

```bash
VERIFY_WHEEL_INSTALL=1 bash scripts/build_whl.sh
```

构建细节见 [编译与打包](build-and-package.md)。

### 4. 代码质量检查

项目使用 pre-commit 进行提交前检查，包括 Ruff、pylint、Bandit、typos 等。首次使用前安装 hook：

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

配置文件位于 `.pre-commit-config.yaml` 与 `pre-commit/` 目录。Python 代码风格与静态检查规则见 `pre-commit/pyproject.toml`。

### 5. 提交 Pull Request

PR 中建议说明：

- 改动动机与影响范围
- 关联 Issue（如有）
- 本地验证方式（测试命令、手动验证步骤）
- 文档是否已同步更新

文档类 PR 可在本地构建 Sphinx 预览，方法见 [ReadTheDocs 本地验证说明](readthedocs-local-build.md)。

## 按贡献类型的具体指引

### 文档贡献

文档统一维护在 `docs/zh/` 下，按快速入门、Agent 指南、用户指南、开发指南组织。修改 README、安装步骤或 Quick Start 时，建议：

1. 对照 [快速入门指导](../getting_started/quick_start.md) 走通最小流程
2. 使用内置 skill [document-ux-review](../user_guide/document-ux-review.md) 做文档上手体验审查
3. 本地构建文档确认无 ERROR：[ReadTheDocs 本地验证说明](readthedocs-local-build.md)

常见问题可先查 [FAQ](../user_guide/faq.md)。

### Skills 贡献

Skills 是项目最重要的扩展面之一。仓库根目录 `skills/` 为内置 Skills 来源，构建 wheel 时会打包到 `resources/configs/default/skills/`。

新增 Skill 的推荐步骤：

1. 在 `skills/` 下创建 kebab-case 目录，并提供 `SKILL.md`（含 `name`、`description` frontmatter）
2. 将稳定、可重复的步骤放入 `scripts/`（如有需要）
3. 在对应 Agent 的 `skills.patterns` 中放开该 Skill
4. 启动 `msagent` 后通过 `/skills` 验证可见性
5. 更新 [skills/README.md](../skills/README.md) 中的技能列表

详细约定见 [配置与扩展 · 添加自定义 Skill](../user_guide/configuration-and-extension.md#custom-skill-guide) 与 [skills/README.md · 新增 Skill 建议流程](../skills/README.md)。

Skill 匹配与过滤规则见 [Agent / Tool / Skill 过滤规则](../user_guide/agent-tool-skill-filter-rules.md)。

### Agent / 配置贡献

Agent 定义位于 `resources/configs/default/agents/`，运行时复制到工作目录 `.msagent/agents/`。修改 Tool 或 Skill 可见范围时：

- `tools.patterns` 使用 `<category>:<module>:<name_pattern>` 三段式
- `skills.patterns` 使用 `<category>:<name_pattern>` 两段式
- 支持 `!` 前缀负规则与 `fnmatch` 通配符

完整语义与 smoke 示例见 [Agent / Tool / Skill 过滤规则](../user_guide/agent-tool-skill-filter-rules.md) 及 `resources/configs/default/agents/msagent-filter-smoke.yml`。

MCP 接入与字段说明见 [配置与扩展 · MCP 配置](../user_guide/configuration-and-extension.md#mcp-配置)。

### 框架代码贡献

框架基于 deepagents 运行时，模块化分为交互层、调度层、核心层与基础设施层。开发前建议阅读 [架构概览](arch_overview.md) 中的：

- 系统架构与核心模块职责
- 启动流程与 Agent 执行流程
- 测试设计（测试目录为 `tests/`）

修改依赖或 Python 版本要求时，请同步更新 [版本与兼容性](version-and-compatibility.md) 并在 PR 中说明兼容性影响。

### 测试贡献

- 测试目录：`tests/`
- 运行方式：`uv run pytest -q`
- 集成测试可使用 `@pytest.mark.integration` 标记

涉及 Agent 行为、配置加载、Tool/Skill 过滤的改动，建议补充或更新对应单元测试。

## 发布相关（维护者）

正式发版流程由 Git tag 触发 CI，包括版本校验、测试、wheel/sdist 构建、GitHub Release 与 PyPI 发布。维护者请参考 [Release Guide](tag-release.md)。

发版前关键检查项：

- `pyproject.toml` 中 `project.version` 与 tag 严格对应（如 `0.1.3` ↔ `v0.1.3`）
- 修改 `pyproject.toml` 后必须同步 `uv.lock`
- 仓库根目录 `skills/` 目录完整且非空

## 许可证与社区

- 许可证：[Mulan PSL v2](http://license.coscl.org.cn/MulanPSL2)，详见 [法律与声明](../legal/index.md)
- 问题反馈：[GitCode Issues](https://gitcode.com/Ascend/msagent/issues)
- 在线文档：[ReadTheDocs](https://mindstudio-agent.readthedocs.io/zh-cn/latest/)
- 昇腾社区：[MindStudio 软件入口](https://www.hiascend.com/cn/developer/software/mindstudio)

再次感谢你的贡献。无论是修复一行文档、补充一个 Skill，还是改进核心框架，都会帮助更多 Ascend 开发者更快完成调试与调优。
