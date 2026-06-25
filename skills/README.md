# Mindstudio Agent Skills

## 1. Skill 规范

本仓库遵循 Agent Skills 的通用约定，目标是保证技能在不同 agent 间可复用、可迁移、可互操作。

### 1.1 目录结构

每个 skill 采用扁平、自包含的目录结构：

```text
skill-name/
├── SKILL.md       # 必需：技能定义与说明
├── references/    # 可选：参考资料
├── scripts/       # 可选：辅助脚本
└── assets/        # 可选：模板、资源
```

### 1.2 `SKILL.md` 规范

每个 skill 必须包含 `SKILL.md`，并使用 YAML frontmatter 加 Markdown 正文：

```md
---
name: skill-name
description: 技能的详细描述，说明它做什么，以及什么时候使用。
---
```

遵循以下原则：

- `name` 使用小写字母、数字和连字符
- `description` 同时写清“做什么”和“何时使用”
- `scripts/` 放可重复执行的确定性逻辑
- `references/` 放较长的背景材料和细节说明
- `assets/` 放模板、图表、数据文件等静态资源


## 2. Skill 索引目录

### 2.1 性能 Skills

| Skill | 作用 | 示例 prompt |
| --- | --- | --- |
| `ascend-profiler-data-validation` | 检查 profiling 数据是否完整、可分析 | `帮我检查这个 profiler 目录能不能进入后续分析` |
| `ascend-profiler-db-explorer` | 分析 Ascend profiler 数据库 | `查一下这个 db 里 TopK 算子和通信耗时` |
| `ascend-cluster-fast-slow-rank-detector` | 做集群快慢卡分析 | `分析这个集群 profiling 目录里的快慢卡原因` |
| `ascend-msprof-analyze-cli` | 做 Ascend 性能综合分析 | `用 msprof-analyze 看下这个性能瓶颈` |
| `ascend-computation-analysis` | 分析计算侧瓶颈 | `看这个 rank 的计算热点和融合机会` |
| `ascend-communication-analysis` | 分析通信侧瓶颈 | `查这个集群里通信耗时和慢卡` |
| `ascend-schedule-analysis` | 分析调度、下发和 Host Bound 问题 | `看看下发延迟和调度卡在哪里` |
| `mindstudio-cpu-binding` | 分析 CPU 绑核、NUMA 和 Host 侧瓶颈 | `排查这个多卡任务的 CPU binding 问题` |

### 2.2 精度 Skills

| Skill | 作用 | 示例 prompt |
| --- | --- | --- |
| `nan-overflow-detection` | 定位 NaN / overflow / gnorm 异常源头 | `帮我找出最早出现溢出的 rank 和算子` |
| `deterministic-calculation-analysis` | 分析确定性计算问题 | `比对这批 msProbe 数据，找首个不一致 API` |
| `rl-consistency-analysis` | 做训练与推理一致性根因分析 | `分析这次训练和推理不一致的根因` |

### 2.3 量化 Skills

| Skill | 作用 | 示例 prompt |
| --- | --- | --- |
| `quantization-accuracy-tuning-orchestrator` | 自动化量化与精度调优 | `帮我跑一轮自动量化和精度调优` |
| `quant-tuning-quantize` | 执行量化 | `按这个 Practice YAML 做量化` |
| `quant-tuning-evaluate` | 执行量化模型评测 | `把量化后的模型跑一遍评测` |
| `tune-practice-cfg` | 生成或修改 Practice YAML | `帮我生成这一轮调优的 YAML` |
| `msmodelslim-quick-quant` | 快速量化入门 | `给我一个最简量化方案` |
| `msmodelslim-layer-wise-quantization` | 逐层量化 | `模型内存不够，改成逐层量化` |
| `msmodelslim-model-dequant` | 反量化接入 | `给这个 FP8 模型补反量化能力` |
| `msmodelslim-model-analysis` | 量化前模型分析 | `先分析这个模型适不适合做适配` |
| `msmodelslim-model-adapt` | 模型适配 | `帮我做这个模型的适配实现` |
| `msmodelslim-adapter-verification` | 适配器验证 | `验证这个适配器是否可用` |
| `gen-evaluation-cfg` | 生成评测配置 | `生成一份评测 YAML` |

### 2.4 算子 Skills

| Skill | 作用 | 示例 prompt |
| --- | --- | --- |
| `ascendc-operator-performance-optim` | 做 AscendC 算子性能调优 | `分析这个算子并给出优化代码` |
| `msot-msopprof-operator-profiler` | 做 msprof op 算子分析 | `输出这个算子的瓶颈和优化建议` |
| `op-mfu-calculator` | 计算算子 MFU | `按这个 shape 和耗时算 MFU` |

### 2.5 文档审查 Skills

| Skill | 作用 | 示例 prompt |
| --- | --- | --- |
| `document-ux-review` | 按 README 真跑并审查文档可用性 | `按这个仓库的文档跑一遍，看新手能否走通` |
| `gitcode-code-reviewer` | 审查 GitCode PR | `review 这个 PR 并指出问题` |
| `github-raw-fetch` | 拉取 GitHub 文件或 docs | `把这个 GitHub 文档页转成可读内容` |

## 3. Skill 使用

当用户在对话中输入任务时，Agent 会根据 prompt 的意图自动匹配并触发相应 skill。
如果已知要用哪个 skill，也可以通过 `/skill` 命令手动加载指定技能。

| 命令                              | 说明                                                                           |
|---------------------------------|------------------------------------------------------------------------------|
| `/skills`                       | 打开交互式 Skill 列表，上下键浏览、回车加载。                                                   |
| `/skills <skill-name>`          | 直接指定 Skill 名称加载，如 `/skills ascend-computation-analysis`。                     |
| `/skills <skill-name> <prompt>` | 加载 Skill 并传入任务执行，如 `/skills ascend-computation-analysis 帮我根据性能数据分析有无计算类的瓶颈`。 |

![skills_browser](../docs/zh/figures/skills_browser.png)

## 4. 自定义添加 Skill
 	 
除了内置 Skill，用户进入`msagent`交互界面后，可通过 `/add-skill` 从本地路径安装自定义 Skill，满足个性化场景需求。支持指定 Skill 目录或 `SKILL.md` 文件，安装后立即生效。

| 命令 | 说明 |
| --- | --- |
| `/add-skill <path>` | 从本地路径安装 Skill 目录。 |
| `/add-skill <path>` | 也可直接指定 `SKILL.md` 文件路径。 |

![add_skill](../docs/zh/figures/add_skill.png)


## 5. 安装 Skill 到其他 Agent

可按目标 Agent 支持的方式安装 Skills：

### 5.1 方式一：使用 `npx skills` 管理工具

适用于已集成 `npx skills` 工作流的 Agent。若目标 Agent 不支持该工具，请使用下方“手动拷贝”方式。

```bash
git clone https://gitcode.com/Ascend/msagent.git
cd msagent/skills

# 查看当前仓库包含哪些 skill
npx skills add . --list

# 安装指定的某个或某几个 Skills
npx skills add . --skill ascend-communication-analysis --skill ascend-computation-analysis

# 将 Skills 安装到特定的 Agent（例如 trae 和 opencode）
npx skills add . -a trae -a opencode

# 安装仓库中的所有 Skills 到全部 Agents
npx skills add . --all

# 安装仓库中的所有 Skills 到指定的多个 Agent
npx skills add . --skill '*' -a trae -a opencode
```

### 5.2 方式二：手动拷贝 Skill 目录

适用于支持扫描本地 skills 目录的 Agent。安装前请先确认目标 Agent 的 skills 目录约定。

```bash
git clone https://gitcode.com/Ascend/msagent.git
cd msagent/

# opencode
cp -r skills/ascend-profiler-db-explorer ~/.config/opencode/skills/

# claude
cp -r skills/ascend-profiler-db-explorer ~/.claude/skills/

# codex
cp -r skills/ascend-profiler-db-explorer ~/.codex/skills/
```
