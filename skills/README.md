# mindstudio-skills

一个面向支持 `SKILL.md` 约定的 AI Agent 的技能仓库，当前重点覆盖 **Ascend / MindStudio Profiling 分析**，并补充文档体验审查、GitCode PR 审查等通用辅助技能。

这个仓库的目标不是堆积提示词，而是把可复用的诊断经验、明确的 SOP、以及必要的脚本工具收敛到同一个技能目录里，让 Agent 在合适场景下稳定触发、稳定执行、稳定输出。
## 快速使用

### 1. 选择一个 skill

每个 skill 都是一个独立目录，至少包含一个 `SKILL.md`。如果你的 Agent 支持本地 skills 目录，通常可以直接把对应目录复制、软链接，或按仓库方式纳入技能搜索路径。

### 2. 确保运行环境可用

部分技能除了说明文档，还会调用本地脚本或外部工具：

- Python 3
- `pandas`
- `sqlite3`（数据库查询类 skill）
- `git`
- `msprof`
- `torch_npu` 或 `mindspore`（仅离线解析相关技能需要）
- `curl` / `curl.exe`（GitHub raw / docs 抓取类场景）
- `GITCODE_TOKEN` 或 `git config --global gitcode.token <your-token>`（仅 `gitcode-code-reviewer` 需要）

### 3. 用任务语言触发技能

示例：

- “帮我检查这个 MindStudio profiler 数据是否完整可分析”
- “分析这个 Ascend 集群 profiling 目录里的快慢卡问题”
- “查一下这个 `msprof_*.db` 里最耗时的 TopK 算子和通信耗时”
- “根据这个 matmul 的 shape 和耗时计算 MFU”
- “按这个仓库 README 真跑一遍，看看新人能不能走通”
- “review 这个 GitCode PR，并把需要修改的问题整理出来”
- “把这个 GitHub 文件页面链接转成 raw content 给我”
- “读取这个仓库里关于 X 的 docs，先按 `agent_router.md` 找真实路径”

对于 `gitcode-code-reviewer`：

- 输入可以是 GitCode PR 链接，例如 `https://gitcode.com/<owner>/<repo>/pull/<number>`
- 技能会先获取 PR 元数据、变更文件和 diff，再结合仓库上下文做“有依据”的审查，而不是只复述 patch
- 如需把结论发布回 PR，需要先配置 GitCode 访问令牌

对于 `github-raw-fetch`：

- 如果目标是普通文本文件，按 `github.com/<owner>/<repo>/blob/<ref>/...` 到 `raw.githubusercontent.com/<owner>/<repo>/<ref>/...` 的规则直接转换并抓取
- 如果目标属于 docs、FAQ、指南、Markdown 文档体系，必须先读取同仓库同 `ref` 的 `agent_router.md`，再根据其中的目录映射、别名或入口规则推导真实路径
- 抓取 raw 内容时优先使用 `curl`；在 PowerShell 环境中优先使用 `curl.exe -L`

对于 `ascendc-operator-performance-optim`：

- 输入可以是 msprof op 性能数据目录（包含 ArithmeticUtilization.csv、L2Cache.csv 等文件）
- 技能会自动诊断性能瓶颈（Vector/Scalar 利用率、L2 Cache、流水线、资源冲突等）
- 结合算子源码给出具体的代码优化建议，并参考 AscendC 高阶 API 文档生成优化后的代码

## 特点

- 聚焦真实问题场景，而不是抽象能力描述
- 每个 skill 都有清晰的触发条件、边界和输出要求
- 复杂流程优先固化为脚本，减少 Agent 临场发挥带来的不确定性
- 保持轻量目录结构，便于直接复制、组合或迁移到其他 skill 仓库

## 当前技能

| Skill | 说明                                                                                                    | 典型输入                                                                                     | 依赖                                                                     |
| --- |-------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`ascend-profiler-db-explorer`](./ascend-profiler-db-explorer/SKILL.md) | 面向 Ascend PyTorch Profiler / msprof 数据库的 SQL 分析技能，把自然语言问题转成安全可执行 SQL，并结合表结构输出诊断结论                     | `ascend_pytorch_profiler*.db` / `msprof_*.db` 路径，或“TopK 算子 / 通信耗时 / 下发分析 / schema 查询”等问题 | `sqlite3`、Python                                                       |
| [`cluster-fast-slow-rank-detector`](./cluster-fast-slow-rank-detector/SKILL.md) | 面向 Ascend 集群 Profiling 的快慢卡诊断，区分 Host 下发慢、计算慢、通信慢等瓶颈类型                                                | 集群 profiling 根目录、慢卡/快卡分析诉求                                                               | `msprof-analyze-advisor`、Python、`pandas`                               |
| [`document-ux-review`](./document-ux-review/SKILL.md) | 像第一次接触项目的人一样按 README / quick start 真跑一遍，判断新人是否能走通并输出文档体验问题报告                                          | 仓库链接或本地仓库路径，以及“按 README 试一下 / 帮我检查文档能不能跑通”这类请求                                           | 目标项目所需运行环境、Shell、包管理工具                                                 |
| [`gitcode-code-reviewer`](./gitcode-code-reviewer/SKILL.md) | 审查 GitCode PR，结合 PR metadata、diff 和仓库上下文给出深度审查结论，并可按需发布逐行评论                                           | GitCode PR 链接，或“review this PR / 检视这个 PR / 检查 PR”这类请求                                    | Python、`git`、`GITCODE_TOKEN` 或 `git config --global gitcode.token ...` |
| [`mindstudio_profiler_data_check`](./mindstudio_profiler_data_check/SKILL.md) | 对 MindStudio profiler / msprof 采集结果做前置体检，判断是否可进入后续分析                                                  | profiler 结果目录                                                                            | `msprof`，离线解析时依赖 `torch_npu` 或 `mindspore`                             |
| [`op-mfu-calculator`](./op-mfu-calculator/SKILL.md) | 计算 matmul / GEMM / FlashAttention 等算子的 MFU，并给出推导过程                                                    | 算子类型、shape、执行时间、峰值算力                                                                     | 无额外脚本依赖                                                                |
| [`msprof-analyze-cli`](./msprof-analyze-cli/SKILL.md) | 面向 Ascend 性能数据的综合分析，涵盖【集群分析】和【专家建议】两大能力。 | 集群 profiling 路径、性能瓶颈/调优建议诉求                                              | `msprof-analyze`，Python                                                   |
| [`github-raw-fetch`](./github-raw-fetch/SKILL.md) | 读取 GitHub 仓库中的源码、配置、README、Markdown 与 docs；普通文件自动将 `blob` 链接转换为 raw，docs 场景会先读取同仓库同 `ref` 的 `agent_router.md` 再定位真实路径 | GitHub `blob` / raw 文件链接，或“读取某仓库某篇 docs”这类请求                                             | 支持网络抓取的 Agent / 工具，推荐 `curl` / `curl.exe`                              |
| [`msmodeling-device-config`](./msmodeling-device-config/SKILL.md) | 将自然语言、表格或截图中的硬件规格整理为 `DeviceProfile` 设备画像建模输入，引导用户补齐必要事实、区分待校准项，并生成可复用的设备画像配置思路 | 新硬件规格说明、官网参数、截图文字、设备画像新增/修订请求                                                   | 主要依赖仓库内 `tensor_cast` 相关代码与文档                                                 |
| [`msmodeling-env-installer`](./msmodeling-env-installer/SKILL.md) | 安装和验证 `msmodeling` 开发环境，覆盖 `uv` 创建 `myenv`、安装 `requirements.txt`、配置 `PYTHONPATH` / `HF_ENDPOINT` 与依赖一致性检查 | msmodeling 环境安装、创建 `myenv`、安装依赖、检查 Python 依赖、配置 Hugging Face 镜像 | Python 3.10+、`uv`、Shell / PowerShell |
| [`msmodeling-text-generate-executor`](./msmodeling-text-generate-executor/SKILL.md) | 交互式补齐 `python -m cli.inference.text_generate` 所需参数，生成单点仿真候选命令，执行前确认并在执行后总结关键性能指标                        | 模型名、device profile、设备数、Prefill/Decode 场景、并行策略、profiling 数据库等                           | Python3、`msmodeling` 仓库运行环境                                                  |
| [`msmodeling-throughput-optimizer-executor`](./msmodeling-throughput-optimizer-executor/SKILL.md) | 交互式补齐 `python -m cli.inference.throughput_optimizer` 所需参数，生成吞吐规划候选命令，执行前确认并总结最佳部署策略与跨硬件对比结果         | 模型名、硬件 profile、设备数、输入输出长度、TTFT/TPOT、聚合/分离/PD 配比规划诉求                               | Python3、`msmodeling` 仓库运行环境                                                  |
| [`ascendc-operator-performance-optim`](./ascendc-operator-performance-optim/SKILL.md) | 基于 msprof op 性能数据诊断 AscendC 算子性能瓶颈，给出代码优化建议，并参考 AscendC 高阶 API 文档生成优化后的代码                             | msprof op 性能数据目录、算子源码路径、AscendC API 文档链接                                                 | Python3                                                                |
| [`msot-msopprof-operator-profiler`](./msot-msopprof-operator-profiler/SKILL.md) | 基于 msprof op 工具深度分析算子性能瓶颈，给出算子信息、关键性能数据、TOP5性能瓶颈、TOP5优化建议与报告总结，提升开发者算子性能优化效率  | 算子可执行程序、算子msprof op 性能数据目录                                                               | Python3                                                                |
| [`nan-overflow-detection`](./nan-overflow-detection/SKILL.md) | 多卡分布式训练 loss/gnorm 精度溢出检测与根因追溯，跨 rank 定位源卡并追溯根因算子                                              | 包含 rank 子目录（`rank0/`, `rank1/`, ...）的 step0 目录                                          | Python3                                                                |
| [`deterministic-calculation-analysis`](./deterministic-calculation-analysis/SKILL.md) | 执行 MindStudio Probe (msProbe) 数据比对并分析比对结果，定位确定性计算问题首个输入一致输出不一致的API       |  msPorbe dump 数据目录（采集统计信息和tensor的CRC-32校验值，task="statistics", summary_mode="md5"）     | Python3, `msProbe`                                                                |

## 目录结构

```text
.
├── README.md
├── ascend-profiler-db-explorer/
│   ├── SKILL.md
│   ├── references/
│   └── scripts/
├── cluster-fast-slow-rank-detector/
│   ├── SKILL.md
│   └── scripts/
│       ├── compare_api_stats.py
│       ├── compare_op_stats.py
│       └── rank_data_finder.py
├── document-ux-review/
│   └── SKILL.md
├── gitcode-code-reviewer/
│   ├── SKILL.md
│   ├── examples/
│   ├── references/
│   └── scripts/
│       ├── fetch_pr_info.py
│       ├── post_pr_comment.py
│       └── setup_token.py
├── github-raw-fetch/
│   └── SKILL.md
├── mindstudio_profiler_data_check/
│   ├── SKILL.md
│   └── scripts/
│       ├── offline_parse_mindspore.py
│       └── offline_parse_pytorch.py
├── nan-overflow-detection/
│   ├── SKILL.md
│   └── scripts/
│       ├── cross_rank_analyzer.py
│       └── single_rank_tracer.py
├── deterministic-calculation-analysis/
│   ├── SKILL.md
│   └── scripts/
│       ├── find_first_diff_api_L1.py
│       ├── find_first_diff_api_mix.py
│       ├── md5_dump_files_checker.py
│       └── msprobe_utils.py
├── op-mfu-calculator/
│   └── SKILL.md
```


## Skill 设计约定

本仓库遵循一种尽量通用的 Agent Skills 组织方式：

- 每个 skill 一个目录
- 必须包含 `SKILL.md`
- `SKILL.md` 使用 YAML frontmatter，至少包含：
  - `name`
  - `description`
- 可选的 `scripts/` 用于承载确定性更强、需要重复执行的辅助工具
- 说明文档优先写清楚触发条件、步骤约束、输出格式，而不是泛泛解释概念

## 新增 Skill 的建议流程

1. 新建一个 kebab-case 目录，例如 `my-new-skill/`
2. 添加 `SKILL.md`，写清楚 `name`、`description`、适用场景和 SOP
3. 如果流程里有稳定可执行的步骤，把它放进 `scripts/`
4. 用一个真实任务做最小验证，确认技能会被正确触发
5. 把新 skill 补充到本 README 的“当前技能”表格中
