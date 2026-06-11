---
name: mindstudio-cpu-binding
description: Use when diagnosing NPU + PyTorch or LLM Serving Host CPU affinity, NUMA locality, cgroup/cpuset constraints, CPU range conflicts, PyTorch/runtime threading, DataLoader, tokenizer, scheduler, vLLM-Ascend, SGLang, TTFT, TPOT, tokens/s, QPS, or multi-rank/multi-worker CPU binding issues.
---

# mindstudio-cpu-binding

`mindstudio-cpu-binding` 是面向 NPU + PyTorch / LLM Serving 工作负载的 Host CPU 绑核优化 Advisor Skill，覆盖 PyTorch 训练、PyTorch 离线/批量推理，以及 vLLM-Ascend、SGLang 等推理服务场景。它指导 Claude 用最少必要问题获取诊断上下文，基于 Host CPU Snapshot 做证据化分析，输出报告、保守/进阶建议、风险、回滚方式和优化前后验证计划。

核心原则：**只基于已观测证据下结论；缺失信息必须显式标记；任何改变运行状态的动作都必须先获得用户明确确认。**

## 适用场景

使用本 Skill 当用户需要：

- 分析 NPU + PyTorch 训练、PyTorch 离线/批量推理、vLLM-Ascend 或 SGLang 推理服务的 Host CPU 绑核问题。
- 判断进程、rank、worker、engine 或实例是否未绑核、跨 NUMA、CPU range 过宽或过窄。
- 检查 Docker、K8s、Slurm、cgroup/cpuset 对真实可用 CPU 的限制。
- 分析 PyTorch、OpenMP、MKL/BLAS、DataLoader worker、tokenizer、scheduler、API server 或 engine worker 线程是否超过有效 CPU。
- 排查多 rank、多 worker、多实例之间 CPU range 重叠或 NPU/NUMA locality 不匹配。
- 生成 CPU 绑核诊断报告、保守优化方案、进阶优化建议、风险说明、回滚方式和 before/after 验证计划。

如果用户只询问架构、模板、路线图或能力边界，不进入采集和诊断流程，只回答对应设计问题。

## 非目标

不要把本 Skill 扩展到以下范围，除非用户明确要求重新定义范围：

- ftrace、eBPF、perf 深度 Host Bound 诊断。
- IRQ affinity 调整。
- CPU governor、kernel boot 参数、系统级隔离核配置。
- 自动修改 Docker、K8s、Slurm 或节点级 cgroup 配置。
- 自动重启、kill 或迁移线上服务。
- 无用户确认地执行 `taskset`、`numactl`、内部绑核脚本或任何改变运行状态的命令。

中高风险动作只能输出建议、风险和人工操作步骤，不能自动执行。

## 安全边界

- 采集阶段默认只读。
- 诊断结论必须区分“已观测证据”和“推断结论”。
- 缺少 NPU -> NUMA 映射时，不得猜测 locality。
- 缺少 cgroup/cpuset 信息时，不得声称推荐 CPU range 一定可用。
- 生成推荐 CPU range 时，必须优先考虑 `cpuset_cpus_effective` 等真实可用 CPU 边界。
- 所有可能改变运行状态的命令都必须先展示风险、回滚方式和验证方式，并等待用户明确确认。
- 只读发现和只读采集命令可以直接给出可执行命令并说明意义，让用户选择是否执行；不要把 PID / worker / NPU 映射推导负担转嫁给用户。
- 对线上服务、容器编排、Slurm、K8s 节点级配置，不自动修改。

## 快速流程

```text
用户问题
  -> 判断是否为具体诊断
  -> 场景澄清
  -> 输入完整性检查
  -> 生成只读采集计划或读取已有 Snapshot
  -> Snapshot 质量检查
  -> 执行诊断规则
  -> 输出报告摘要和 HTML/JSON 产物
  -> 给出保守/进阶方案
  -> 等待用户确认后才可执行低风险临时绑核
  -> 保存 rollback-state 并提供回滚
  -> 设计 before/after 验证计划
```

## 场景澄清

只问阻塞采集或影响关键判断的问题。先根据用户描述识别真实问题场景，再提问；不要机械地问“训练还是推理”。能从 Snapshot 得到的信息，不要求用户手填。提问必须一次只问一个问题，优先使用选择题，并为用户提供“不确定 / 先帮我看 / 使用默认值”选项。

### 场景识别

| 用户问题特征 | 进入分支 |
|--------------|----------|
| step time 抖动、samples/s 低、rank 间不均衡、DDP/HCCL 训练波动 | PyTorch training |
| batch inference 吞吐低、离线推理延迟、多进程推理抢 CPU | PyTorch offline/batch inference |
| vLLM-Ascend、SGLang、OpenAI API server、TTFT、TPOT、tokens/s、QPS、p99、timeout | LLM serving inference |

如果用户已经明确说了 vLLM-Ascend 或 SGLang，直接进入 LLM serving 分支，不再泛泛追问是否 PyTorch 推理。

### 单步选择式提问模板

如果用户描述已经能判断场景，直接进入该场景的下一问；不要重复询问场景。如果无法判断场景，只问一个选择题：

```text
你要分析的是哪类任务？

1. PyTorch 训练
2. PyTorch 离线/批量推理
3. vLLM / SGLang / OpenAI API 这类 LLM Serving
4. 不确定，先帮我自动发现
```

PyTorch training 的问题目标选择：

```text
你最想改善哪个问题？

1. step time 抖动或长尾高
2. samples/s 或整体吞吐低
3. rank 之间速度不均衡
4. NPU 利用率不稳定
5. 不确定，先帮我看
```

PyTorch offline/batch inference 的问题目标选择：

```text
你最想改善哪个问题？

1. 批量推理吞吐低
2. 单次请求延迟高
3. 多进程/多实例之间抢 CPU
4. NPU 利用率不稳定
5. 不确定，先帮我看
```

LLM serving 的问题目标选择：

```text
你最想改善哪个问题？

1. 首字响应慢
2. 单个 token 生成慢
3. tokens/s 或 QPS 低
4. 延迟抖动或 p99 高
5. timeout / 请求失败
6. 不确定，先帮我看
```

运行位置选择：

```text
这个任务大概运行在哪里？

1. 直接在服务器上
2. Docker/容器里
3. K8s/容器平台里
4. Slurm/作业调度里
5. 不确定
```

只读发现确认：

```text
我可以先运行只读进程发现命令来找候选 PID 和 NPU 映射。
它只读取 `ps` 和 `npu-smi`，不修改系统状态。

是否执行？

1. 执行
2. 先不执行，我提供已有 snapshot
3. 先不执行，我手动提供 PID
```

候选确认和 Snapshot 采集确认也必须一次只问一个问题。采集后或信息不足时，才补问 DataLoader、OMP/MKL/torch 线程、TP/DP/PP、tokenizer、benchmark 指标等专业信息；补问也应使用单个选择题或明确说明“不提供也可以继续”。

### 提问规则

- 每轮只问一个问题；不要使用“我先确认 3 点 / 4 点 / 5 点”的批量问法。
- 优先用选择题；选择项使用用户语言，并提供“不确定，先帮我看”或“使用默认值”选项。
- 用户已经提供的信息不要重复问。
- 优先确认主要异常指标、部署方式，以及是否允许运行只读发现命令；PID、进程/worker/instance 到 NPU 的映射应优先由只读发现生成候选。
- 如果用户不知道目标 PID 或 worker/NPU 映射，先说明只读进程发现命令的意义并给出可执行命令，让用户选择是否执行；发现后让用户确认候选，而不是要求用户手工猜 PID 或推导完整映射。
- 不询问 CPU/NUMA 拓扑、当前 affinity、cgroup/cpuset、NPU locality、runtime CPU 使用率等采集器能自取的信息。
- 不把 LLM serving 场景套用 DataLoader 问题；不要把 API server、scheduler、engine worker 都统称为 rank。
- 如果用户已有 Snapshot，跳过采集提问，直接进入 Snapshot 质量检查。

详细提问协议见 `docs/question-flow.md`。

## Snapshot 输入与采集

优先使用用户已有的 Snapshot JSON。Snapshot 数据契约见 `docs/snapshot-schema.md`。

如果用户没有 Snapshot，指导用户在目标 Linux NPU 节点运行只读采集。采集命令必须满足：

- 指定目标 PID。
- 输出 JSON Snapshot。
- 只读访问 `/proc`、`/sys`、cgroup、NPU topology、PyTorch/torch_npu 环境信息，以及 LLM serving 相关进程和线程信息。
- 将缺失字段写入 `availability.missing`、`availability.partial` 或 `availability.errors`。
- 不修改 affinity、cgroup、系统配置或进程状态。

当前 Skill 包内 `scripts/` 是辅助原型脚本目录，已包含 collect/analyze/report、拓扑采集和进程发现入口；这些入口用于只读采集与诊断验证，仍按原型 CLI 对待。当前已有 Snapshot 时可用原型分析入口：

```bash
python scripts/cli.py analyze --snapshot <snapshot.json> --out out
```

仓库开发环境可用 `samples/snapshot.multi-rank.json` 作为离线示例输入。

当前拓扑采集解析原型支持两种模式：

```bash
python scripts/cli.py collect-topology --lscpu-file samples/lscpu.sample.txt --npu-smi-topo-file samples/npu-smi-topo.sample.txt --out out/topology.json
python scripts/topology_collect.py --out out/topology.json
```

第一条用于离线解析样本文本，第二条用于真实 Linux NPU 设备上的只读 live 验证。live 原型只执行 `lscpu` 和 `npu-smi info -t topo`，用于验证能否采集支撑拓扑渲染的数据，不修改系统状态。

当前进程发现原型也支持两种模式：

```bash
python scripts/cli.py discover-processes --ps-file samples/ps.sample.txt --npu-smi-info-file samples/npu-smi-info.sample.txt --out out/processes.json
python scripts/process_discovery.py --out out/processes.json
```

第一条用于离线解析样本文本，第二条用于真实 Linux NPU 设备上的只读 live 发现。live 原型只执行 `ps -eo pid,ppid,comm,args` 和 `npu-smi info`，用于发现 API server、scheduler、tokenizer、engine worker、rank/worker、runtime 和 NPU 进程候选项。发现结果是候选列表，必须让用户确认后才能作为诊断目标。

## Snapshot 质量检查

读取 Snapshot 后先检查质量，再诊断：

- `schema_version` 是否支持。
- `target_pids` 是否与 `processes` 对齐。
- CPU / NPU / NUMA topology 是否存在。
- NPU topology 和 NPU -> NUMA locality 是否存在。
- process / thread affinity 是否存在。
- cgroup/cpuset/cpu quota/throttling 信息是否存在。
- PyTorch threading/env、DataLoader、serving runtime、tokenizer、scheduler、API server 或 engine worker 线程信息是否存在。
- runtime sample、Top threads、current CPU、NUMA 分布是否存在。

缺少字段时：

1. 继续执行有证据支持的规则。
2. 把不可判断项列入“信息缺口”。
3. 在报告中明确哪些结论无法下。
4. 不根据缺失字段做确定性结论。

## 诊断规则顺序

按以下顺序分析，规则详情见 `docs/diagnosis-rules.md`：

1. `R010` 信息不足。
2. `R007` cgroup/cpuset 与应用绑核冲突。
3. `R001` 进程未绑定 CPU。
4. `R003` Rank / Worker / Instance / NPU / NUMA 不匹配。
5. `R002` 跨 NUMA 运行。
6. `R004` 绑核范围过宽。
7. `R005` 绑核范围过窄。
8. `R006` Runtime / PyTorch / Serving 线程过载。
9. `R008` 多 rank / 多 worker / 多实例 CPU range 重叠。
10. `R009` SMT 使用策略与目标不匹配。

排序原因：先处理信息缺口和 cgroup 边界，再处理 locality、range、线程数、SMT 和多实例冲突。

## 输出报告

报告必须优先呈现结论、证据和行动建议，不平铺系统信息。报告模板见 `templates/report-template.md`。

最终报告至少包含：

1. 报告摘要。
2. 当前 CPU 绑定状态。
3. CPU / NPU / NUMA 拓扑关系。
4. CPU / NUMA 逻辑 CPU 网格。
5. 运行时 CPU 使用与竞争情况。
6. 问题发现。
7. 推荐绑核方案。
8. 推荐 PyTorch / Runtime / Serving 线程配置。
9. 验证计划。
10. 风险与回滚。
11. 信息缺口。

拓扑关系 section 必须基于 Snapshot 和诊断计划渲染，不在报告生成阶段重新执行 `lscpu`、`npu-smi` 或读取 live `/proc`。它应先用轻量内联 SVG 展示 Server -> NUMA -> NPU 和 NPU interconnect，再用关系卡片展示 NUMA Node、CPU range、本地 NPU、PID/rank/worker/instance、当前 CPU range、cgroup 有效 CPU range、推荐 CPU range 和跨 NUMA 状态。

结论表达必须基于证据。例如：

- 应写：`根据 Snapshot 中 PID 12345 的 Cpus_allowed_list=0-127，且机器存在 NUMA 0/1 两个节点，当前进程未进行有效 CPU 绑定。`
- 不应写：`这肯定导致性能差。`

缺少 locality 时应写：`当前 Snapshot 缺少 NPU -> NUMA 映射，因此不能判断 Rank/NPU/NUMA 是否匹配。`

## 建议分级

每次输出建议时至少区分保守方案和进阶方案。

### 保守方案

- 不修改系统级配置。
- 只针对目标 PID 或下次启动命令。
- 使用 cgroup/cpuset 允许范围内的 CPU。
- 可验证、可回滚。
- 适合先小范围验证。

### 进阶方案

- 可包含启动命令、环境变量、`numactl`、DataLoader、tokenizer、scheduler、API server、engine worker、OpenMP、BLAS、PyTorch 或 serving runtime 线程池配置建议。
- 通常需要重启任务。
- 只作为建议输出，不能自动执行。
- 必须附带风险、回滚方式和验证指标。

### 禁止自动执行

- IRQ affinity。
- CPU governor。
- kernel boot 参数。
- K8s 节点级配置。
- Docker、K8s、Slurm 配置修改。
- 自动重启线上服务。

## 执行与回滚确认

默认只输出 dry-run / preview。

如果用户要求执行低风险临时绑核，必须满足：

1. PID 仍然存在。
2. 目标 CPU range 在 cgroup/cpuset 允许范围内。
3. 已展示当前 affinity、目标 affinity、apply 命令、rollback 命令和风险。
4. 已保存或要求保存 `rollback-state.json`，包含 PID、进程启动时间、原始 `Cpus_allowed_list`、目标 CPU range。
5. 用户明确确认执行。
6. 执行后重新采集或查询 affinity，验证是否生效。

`taskset` 回滚形式：

```bash
taskset -cp <original-cpu-list> <pid>
```

回滚前必须检查 PID 是否仍存在，并尽量通过 `/proc/<pid>/stat` starttime 判断 PID 是否复用。若 PID 已退出或疑似复用，不自动回滚，要求人工确认。

## 验证计划

每次优化建议都必须包含 before/after 验证计划，要求相同 workload、相同时长、相同采集方式。

训练场景指标：

- samples/s。
- step time avg / p50 / p90 / p99。
- NPU utilization avg / min。
- Host CPU utilization by NUMA。
- context switch。
- CPU migration。
- rank 间 step time 方差。

PyTorch 离线/批量推理指标：

- batch throughput / samples/s。
- 单请求或 batch latency p50 / p90 / p99。
- QPS。
- NPU utilization。
- Host CPU utilization by NUMA。
- error rate。
- 多进程/多实例 p99 latency 方差。

LLM serving 指标：

- QPS / requests/s。
- input tokens/s 与 output tokens/s。
- TTFT p50 / p90 / p99。
- TPOT p50 / p90 / p99。
- end-to-end latency p50 / p90 / p99。
- queueing latency。
- timeout / error rate。
- prefill throughput 与 decode throughput。
- API server、scheduler、tokenizer、engine worker CPU utilization。
- NPU utilization。
- Host CPU utilization by NUMA。
- context switch 与 CPU migration。

成功标准应由业务场景确认；没有业务阈值时，可建议以 throughput/tokens/s 提升、TTFT/TPOT/p99 下降、timeout 下降、NPU utilization 稳定性提升、CPU migration/context switch 下降作为观察目标。

## 辅助文件

- `docs/architecture.md`：架构和组件职责。
- `docs/agent-workflow.md`：端到端流程。
- `docs/question-flow.md`：提问协议。
- `docs/snapshot-schema.md`：Snapshot 数据契约。
- `docs/diagnosis-rules.md`：规则 taxonomy。
- `templates/report-template.md`：报告模板。
- `docs/binding-rollback-design.md`：执行后端和回滚机制。
- `scripts/`：当前辅助原型脚本。
- `samples/`：本地示例 Snapshot（仓库开发用，不属于安装 Skill 必需内容）。
- 仓库开发环境中的 `../tests/`：辅助原型测试；独立安装 Skill 时通常不存在，不能作为运行依赖。

## 常见错误

| 错误 | 正确做法 |
|------|----------|
| 直接猜测 NPU locality | 缺少 NPU -> NUMA 映射时标记信息缺口。 |
| 忽略 cgroup/cpuset | 推荐 CPU range 前先看真实可用 CPU 边界。 |
| 一次问太多问题 | 先识别 PyTorch training、PyTorch batch inference 或 LLM serving，再问 PID、映射、异常指标和部署方式。 |
| LLM serving 还追问 DataLoader | vLLM-Ascend/SGLang 场景优先问 API server、scheduler、tokenizer、engine worker、TTFT、TPOT、tokens/s。 |
| 把建议写成确定收益 | 写成待验证假设，并给 before/after 指标。 |
| 未确认就执行 `taskset` | 只输出 preview，等用户明确确认。 |
| 输出 apply 命令但没有 rollback | 每个改变状态的动作都必须有回滚方式。 |
| 扩展到 ftrace/eBPF/perf | 除非用户明确要求，否则保持 CPU affinity MVP 范围。 |
