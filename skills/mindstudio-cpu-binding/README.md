# mindstudio-cpu-binding

`mindstudio-cpu-binding` 是一个面向 **NPU + PyTorch / LLM Serving Host CPU 亲和性优化** 的通用 Agent Skill。它关注的不是单一框架或单一运行时，而是帮助任何支持读取 Skill 目录的 Agent Runtime，围绕以下问题做证据化分析：

- Host CPU affinity 是否合理
- NUMA locality 是否匹配 NPU
- cgroup / cpuset 是否限制了真实可用 CPU
- PyTorch / serving runtime 线程配置是否与 CPU 资源冲突
- 多 rank / 多 worker / 多实例是否存在 CPU range 冲突

## 角色分工

- **SKILL.md**：给 Agent 读取的指令入口，定义交互、边界和工作流。
- **README.md**：给人看的总览文档，说明价值、安装方式、使用场景、CLI 流程和开发入口。

## 目录约定

建议保持如下可移植布局，便于不同 Agent Runtime 直接挂载和识别：

```text
skills/
├── mindstudio-cpu-binding/
│   ├── SKILL.md
│   ├── README.md
│   ├── scripts/
│   ├── docs/
│   └── templates/
└── tests/
```

其中 `skills/tests/` 是仓库内用于保障脚本正确性的测试目录，不属于用户安装 Skill 时必须携带的运行内容。`skills/mindstudio-cpu-binding/samples/` 可作为本地开发验证数据目录存在，用于离线示例验证；它不是运行真实节点诊断的必需内容。

## 核心特色

- **聚焦 NPU 工作负载**：覆盖 PyTorch 训练、离线/批量推理，以及 vLLM-Ascend、SGLang 等 LLM Serving 场景。
- **证据化诊断**：先采集、再分析；结论必须能追溯到 Snapshot 和运行时证据。
- **只读优先**：默认先做只读发现和采集，不做状态修改；PID / worker / NPU 映射优先由命令生成候选，用户确认即可。
- **自包含报告**：在 Snapshot 信息足够时生成 `report.html`，信息不足时会明确标记缺口。
- **安全执行边界**：任何会改变状态的动作都必须先展示风险、回滚和验证方式，并等待显式确认。
- **Agent 友好**：交互、命令、产物、报告结构都按 Agent 工作流设计；提问一次一个、选项优先。

## 适用场景

当你遇到这些问题时，适合使用 `mindstudio-cpu-binding`：

- PyTorch 训练 step time 抖动、samples/s 低、rank 间不均衡
- PyTorch 离线/批量推理吞吐低、延迟高、CPU 争抢明显
- vLLM-Ascend、SGLang，或能提供 PID / 进程 / runtime 证据的其他 LLM Serving 的 TTFT、TPOT、tokens/s、QPS、p99 异常
- 多 rank / 多 worker / 多实例之间 CPU range 重叠
- Docker / K8s / Slurm / cgroup / cpuset 限制导致实际可用 CPU 与预期不一致
- 需要判断进程、worker、engine、scheduler、tokenizer 是否跨 NUMA 运行或绑核过宽/过窄

## 非目标与安全边界

默认不覆盖以下内容，除非用户明确要求扩展范围：

- ftrace / eBPF / perf 深度诊断
- IRQ affinity 调整
- CPU governor 调整
- kernel boot 参数修改
- K8s 节点级配置修改
- 自动重启服务
- 未经明确确认就执行任何状态变更命令

换句话说：`mindstudio-cpu-binding` 可以给建议，但不能擅自改环境。

## Generic 安装方式

任何**可以读取一个 skill 目录**的 Agent Runtime，都可以注册本目录作为 `mindstudio-cpu-binding`。

### 运行时要求

至少需要满足这些能力：

- 能读取 `SKILL.md`
- 能访问 `scripts/`
- 能运行 Python
- 能只读访问 `/proc`、`/sys`、cgroup、`lscpu`
- 如果节点上有 NPU，还能读取 `npu-smi`

### 注册方式

把整个 `skills/mindstudio-cpu-binding` 目录挂载进去最简单；最小运行内容是 `SKILL.md`、`scripts/`、`templates/` 和所需 `docs/`。`samples/` 仅用于离线示例验证，可选；`skills/tests/` 是仓库测试目录，不随 Skill 安装。

### Claude Code 示例

仅作为示例，`mindstudio-cpu-binding` 不是 Claude-only 方案。

项目级挂载（从仓库根目录执行，目标是 Claude Code 的 skills 发现目录 `.claude/skills/mindstudio-cpu-binding`，不是仓库内的包目录）：

```bash
mkdir -p .claude/skills
ln -s /absolute/path/to/mindstudio-cpu-binding .claude/skills/mindstudio-cpu-binding
```

用户级挂载：

```bash
ln -s /absolute/path/to/mindstudio-cpu-binding ~/.claude/skills/mindstudio-cpu-binding
```

### 通用触发示例

以下提示适用于任何支持 Skill 的 Agent Runtime：

- “帮我分析这个 NPU 训练任务的 Host CPU 绑核问题。”
- “检查这个 vLLM-Ascend 服务是否存在 NUMA locality 冲突。”
- “分析多进程推理的 cgroup/cpuset 和 CPU range 是否互相冲突。”
- “读取已有 snapshot.json，给出保守和进阶优化方案。”

## Practical CLI 流程

`mindstudio-cpu-binding` 提供一组可直接跑的原型 CLI。以下命令默认从 Skill 目录执行：

```bash
cd skills/mindstudio-cpu-binding
```

推荐按下面顺序使用。Agent 应先说明每条命令的意义和安全边界，再让用户选择是否执行；用户通常不需要手工推导 PID / worker / NPU 的完整映射。

### 1. 分析已有 Snapshot

```bash
python scripts/cli.py analyze --snapshot out/snapshot.json --out out
```

输出会基于已有 Snapshot 生成诊断计划和 HTML 报告。

### 2. 发现进程与 NPU 候选映射

建议优先从只读进程发现开始，尤其是用户不确定目标 PID、TP worker、rank、instance 或 NPU 映射时。该命令只读取 `ps` 和 `npu-smi`，用于生成候选列表，不修改系统状态：

```bash
python scripts/process_discovery.py --out out/processes.json
```

Agent 应根据 `processes.json` 汇总候选，例如 API server、scheduler、engine/worker、TP rank、rank/main 进程和 NPU 占用关系，再让用户确认哪些候选进入下一步采集。

### 3. 候选确认后的只读采集

用户确认目标 PID 或候选映射后，再运行 Snapshot 采集。该命令只读访问 `/proc`、`/sys`、cgroup、NPU topology 和 runtime 信息，生成 `snapshot.json`，不修改 affinity、cgroup 或系统配置：

```bash
python scripts/cli.py collect --pid <pid> --scenario training --framework pytorch --device-type npu --optimization-goal stability --sample-seconds 10 --out out/snapshot.json
```

### 4. 多 rank 采集示例

```bash
python scripts/cli.py collect --pid 12345 --pid 12346 --scenario training --framework pytorch --device-type npu --optimization-goal stability --sample-seconds 10 --rank-map rank0=12345:npu0,rank1=12346:npu1 --out out/snapshot.json
python scripts/cli.py analyze --snapshot out/snapshot.json --out out
```

## 输出物说明

一次完整流程通常会产出以下文件：

- `snapshot.json`：只读采集结果，记录 CPU / NUMA / NPU / 进程 / cgroup / runtime 信息
- `plan.json`：诊断计划、建议、风险、回滚要点
- `report.html`：自包含 HTML 报告，便于分享和查看
- `raw/`：采集到的原始中间数据，便于排查解析和证据链

## report.html 包含什么

`report.html` 重点展示的是“结论 + 证据 + 建议”，常见结构包括：

1. 报告摘要
2. 当前 CPU 绑定状态
3. CPU / NPU / NUMA 拓扑关系
4. CPU / NUMA 逻辑 CPU 网格
5. 运行时 CPU 使用与竞争情况
6. 问题发现
7. 推荐绑核方案
8. 推荐 PyTorch / Runtime / Serving 线程配置
9. 验证计划
10. 风险与回滚
11. 信息缺口

其中拓扑关系部分会尽量把 Server -> NUMA -> NPU 的关系画清楚，避免只看一堆文本不容易定位。

## 开发者参考

### 数据流

```text
collect -> diagnose -> planner -> report
```

### 关键文件地图

- `scripts/collect.py`：采集入口，负责组织 Snapshot 数据
- `scripts/diagnose.py`：根据规则做问题诊断
- `scripts/planner.py`：把诊断结果整理成方案和建议
- `scripts/report.py`：生成 HTML 报告
- `scripts/topology_view.py`：渲染拓扑视图和关系展示

### 设计文档索引

建议按下面顺序阅读：

- `docs/agent-workflow.md`
- `docs/question-flow.md`
- `docs/snapshot-schema.md`
- `docs/diagnosis-rules.md`
- `docs/html-report-design.md`
- `docs/binding-rollback-design.md`

## 测试与检查

从仓库根目录执行：

```bash
pytest skills/tests/test_report.py skills/tests/test_topology_view.py -v
pytest skills/tests -v
```

从仓库根目录执行 README 检查：

```bash
pre-commit run --files skills/mindstudio-cpu-binding/README.md
```

## 说明与边界

- 本项目不保证性能一定提升；它提供的是诊断、建议和验证路径。
- 不会自动改写 Docker、K8s、Slurm 或节点级配置。
- 不以 Claude-only 为前提，可用于任何兼容的 Agent Runtime。
- 所有状态变更动作都必须先确认，再执行。
