# Agent Workflow

## 1. 目标

`CPU Affinity Advisor Agent` 负责把用户的性能问题转化为可采集、可诊断、可验证、可低风险执行和可回滚的 CPU 绑核优化流程。它不是简单地输出命令，而是要完成：

```text
澄清目标 -> 确认输入 -> 触发采集 -> 读取 Snapshot -> 运行诊断规则 -> 生成 HTML 报告 -> 给出方案 -> 用户确认后调用执行后端绑核 -> 保存回滚状态 -> 支持安全回滚 -> 设计验证 -> 沉淀规则
```

MVP 阶段 Agent 以“诊断 + HTML 展示 + 优化建议”为主，并允许在用户明确确认后通过可配置执行后端执行低风险临时绑核。执行后端可以是系统 `taskset`，也可以是内部经过验证的成熟绑核脚本。中高风险动作只生成建议和回滚方式，不自动执行。

## 2. 工作流总览

```text
用户描述问题
  │
  ▼
Step 1: 场景澄清
  │  训练/推理、NPU/PyTorch、PID、优化目标、环境类型
  ▼
Step 2: 输入完整性检查
  │  是否有 PID、rank 映射、NPU 映射、性能基线
  ▼
Step 3: 生成采集计划
  │  给出当前原型命令、未来完整 collector 命令或调用 MCP 采集工具
  ▼
Step 4: 获取 Host CPU Snapshot
  │  读取 snapshot.json，检查 availability
  ▼
Step 5: 执行诊断规则
  │  匹配 diagnosis-rules.md 中的规则
  ▼
Step 6: 生成 HTML 诊断报告
  │  使用 report-template.md 和 html-report-design.md 输出结论、证据、建议、风险、验证
  ▼
Step 7: 给出保守/进阶方案
  │  taskset / numactl / PyTorch 线程池 / DataLoader 配置建议
  ▼
Step 8: 用户确认后执行低风险绑核
  │  保存 rollback-state，调用 taskset 或内部绑核脚本，执行后验证 affinity
  ▼
Step 9: 支持安全回滚
  │  检查 PID、进程启动时间和当前 affinity，确认后恢复原始 affinity
  ▼
Step 10: 设计验证计划
  │  before/after 指标、采集时长、成功标准、回滚方式
  ▼
Step 11: 规则和案例沉淀
     将真实 case 的问题、证据、收益反馈给规则库
```

## 3. Agent 输入

Agent 应接受三类输入。

### 3.1 用户显式输入

```text
- 场景：训练 / 推理
- 框架：PyTorch / vLLM-Ascend / SGLang
- 设备：NPU
- 当前问题：吞吐低 / step time 抖动 / TTFT 高 / TPOT 高 / p99 高 / NPU 利用率低
- 运行环境：裸机 / Docker / K8s / Slurm
- 优化目标：throughput / latency / stability / isolation
- 是否允许执行只读发现命令：是 / 否
```

目标 PID、rank、worker、instance 到 NPU 的映射优先由只读发现命令生成候选，用户通常只需要确认候选，而不是手工填写完整映射。

### 3.2 采集器输入

来自 `snapshot-schema.md`：

```text
- CPU / NPU / NUMA topology
- NPU topology and NUMA locality
- process / thread affinity
- cgroup / cpuset / CPU quota
- PyTorch / Runtime / Serving threading and environment
- runtime CPU usage sample
- availability / missing fields
```

### 3.3 用户补充的性能基线

```text
训练：
- samples/s
- step time avg / p50 / p90 / p99
- NPU utilization

推理：
- QPS
- latency p50 / p90 / p99
- NPU utilization
- error rate
```

## 4. Step 1：场景澄清

Agent 首先判断信息是否足以开始采集。提问遵循分层 + 分支协议，完整规范见 `question-flow.md`。

核心要点：

- 每轮只问一个问题，优先用选择题，不批量要求用户回答 3-5 个问题。
- 只问阻塞采集的必问项：工作负载场景、问题目标、运行位置、是否允许只读发现、目标候选确认。
- 强烈建议项缺失时给默认值并标记待确认：优化目标、运行环境、rank/worker/instance -> NPU 映射。
- 可选项缺失不阻塞，进入 `availability.missing`。
- 不询问采集器能自取的信息：CPU/NUMA 拓扑、当前 affinity、cgroup、NPU locality。
- 不把 TTFT、TPOT、QPS、p99、TP rank、NUMA、cgroup、OMP、MKL 等术语作为第一层问题；第一层使用“首字响应慢、单个 token 生成慢、吞吐低、延迟抖动”等用户语言。

### 澄清原则

- 每轮只问一个问题；下一问取决于上一轮回答。
- 每个选择题都应包含“不确定 / 先帮我看 / 使用默认值”兜底选项。
- 能从 Snapshot 得到的信息，不要求用户手动提供。
- 缺少 NPU topology 时，不猜测 NUMA locality。
- 如果用户只想先看报告模板或架构，不进入采集流程。

## 5. Step 2：输入完整性检查

Agent 使用完整性检查表判断能否进入采集，详见 `question-flow.md` 的检查表与答案映射。关键判定：

| 检查项 | 必需级别 | 缺失时动作 |
|--------|----------|------------|
| 场景 PyTorch training / PyTorch offline inference / LLM serving | 必需 | 从用户描述推断；无法推断时询问 |
| 目标 PID | 必需 | 用户不知道时，先说明并提供只读 `process_discovery.py` 命令生成候选，用户确认后进入采集 |
| 优化目标 | 推荐 | 从用户选择的问题目标推断；无法推断时默认 `stability` 并标记待确认 |
| rank / worker / instance -> NPU 映射 | 多卡、多 worker、多实例时必需 | 优先从只读发现候选生成，用户确认；发现不足时再标记缺失或请求补充 |
| NPU -> NUMA 映射 | 诊断 locality 时必需 | 通过采集器获取 |
| PyTorch / Runtime / Serving 线程配置 | 推荐 | 采集 env，缺失则标记 |
| 性能基线 | 验证时必需 | 报告中标记缺失 |

## 6. Step 3：生成采集计划

当前阶段优先直接给出可执行的只读命令，解释命令意义，并让用户选择是否执行；不要要求用户先推导 PID / rank / worker / instance 到 NPU 的完整映射。以下命令默认在 Skill 目录执行。

如果用户不知道 PID 或 NPU 映射，先说明“下面这个命令只会读取 `ps` 和 `npu-smi`，用于列出 API server、scheduler、engine/worker、rank、NPU 候选；用户确认后才进入采集”。

离线示例命令：

```bash
cd skills/mindstudio-cpu-binding
python scripts/cli.py discover-processes \
  --ps-file samples/ps.sample.txt \
  --npu-smi-info-file samples/npu-smi-info.sample.txt \
  --out out/processes.json

python scripts/cli.py collect-topology \
  --lscpu-file samples/lscpu.sample.txt \
  --npu-smi-topo-file samples/npu-smi-topo.sample.txt \
  --out out/topology.json
```

真实 Linux NPU 设备上的只读 live 命令：

```bash
cd skills/mindstudio-cpu-binding
python scripts/process_discovery.py --out out/processes.json
python scripts/topology_collect.py --out out/topology.json
```

当前完整 Snapshot 采集原型命令：

```bash
cd skills/mindstudio-cpu-binding
python scripts/cli.py collect \
  --pid <pid> \
  --scenario <training|inference> \
  --framework pytorch \
  --device-type npu \
  --optimization-goal <throughput|latency|stability|isolation> \
  --sample-seconds 10 \
  --out out/snapshot.json
```

这条命令的意义是：在不修改系统状态的前提下，把指定进程的 CPU / NUMA / NPU / cgroup / runtime 证据收集成 Snapshot。若用户不确定 PID，先执行只读进程发现命令并让用户确认候选，再回到这一步。

多 rank 示例：

```bash
cd skills/mindstudio-cpu-binding
python scripts/cli.py collect \
  --pid 12345 \
  --pid 12346 \
  --scenario training \
  --framework pytorch \
  --device-type npu \
  --optimization-goal throughput \
  --rank-map rank0=12345:npu0,rank1=12346:npu1 \
  --sample-seconds 10 \
  --out out/snapshot.json
```

如果后续改为 MCP，Agent 调用顺序应等价于：

```text
collect_cpu_topology
collect_npu_topology
collect_process_affinity
collect_cgroup_limits
collect_runtime_config
collect_cpu_runtime_sample
```

## 7. Step 4：Snapshot 读取与质量检查

Agent 读取 Snapshot 后，先检查 `availability`，再进入诊断。

### 质量检查

```text
- schema_version 是否支持？
- target_pids 是否与 processes 对齐？
- CPU/NUMA topology 是否存在？
- NPU topology 是否存在？如果不存在，哪些诊断不能做？
- cgroup 是否 partial？
- PyTorch threading 字段是否缺失？
- runtime_sample 是否有 top_threads？
```

### 输出行为

如果缺少关键字段，Agent 不应中断整个流程，而是：

1. 继续执行可执行的规则。
2. 把不可执行的规则列入“信息缺口”。
3. 在报告中明确哪些结论无法下。

示例：

```text
当前 Snapshot 缺少 NPU -> NUMA 映射，因此本次报告不能判断 Rank/NPU/NUMA 是否匹配；但仍可判断进程是否未绑核、是否受 cgroup 限制、PyTorch 线程数是否超过可用 CPU。
```

## 8. Step 5：诊断规则执行顺序

建议按以下顺序执行规则：

1. `R010 信息不足`
2. `R007 cgroup/cpuset 与应用绑核冲突`
3. `R001 进程未绑定 CPU`
4. `R003 Rank / NPU / NUMA 不匹配`
5. `R002 跨 NUMA 运行`
6. `R004 绑核范围过宽`
7. `R005 绑核范围过窄`
8. `R006 PyTorch 线程池过载`
9. `R008 多 rank / 多实例 CPU range 重叠`
10. `R009 SMT 使用策略与目标不匹配`

排序原因：

- 先处理信息缺口，避免过度结论。
- 先处理 cgroup，因为它决定真实可用 CPU 边界。
- 再处理 locality、range、线程数和 SMT 策略。

## 9. Step 6：报告生成策略

Agent 使用 `../templates/report-template.md` 生成报告。

### 报告优先级

报告不应平铺所有系统信息，而应按价值排序：

1. 摘要结论。
2. 最高严重程度问题。
3. 支撑证据。
4. 推荐方案。
5. 风险与回滚。
6. 验证计划。
7. 信息缺口。

### 结论表达规则

应该写：

```text
根据 Snapshot 中 PID 12345 的 Cpus_allowed_list=0-127，且机器存在 NUMA 0/1 两个节点，当前进程未进行有效 CPU 绑定。
```

不应该写：

```text
这肯定导致性能差。
```

应该写：

```text
如果该进程只服务 NPU 0，且 NPU 0 位于 NUMA 0，则建议优先绑定到 NUMA 0 的 CPU；当前 Snapshot 缺少 NPU -> NUMA 映射，因此该建议需要补充确认。
```

不应该写：

```text
直接绑到 NUMA 0。
```

## 10. Step 7：推荐方案生成

每次报告至少给出两个方案。

### 10.1 保守方案

特点：

- 不修改系统级配置。
- 只针对目标进程或下次启动命令。
- 可回滚。
- 适合先验证。

示例：

```text
保守方案：
- 将 Rank 0 绑定到 NPU 0 本地 NUMA 的 CPU 子集。
- 使用 taskset 临时调整，或在下次启动时用 numactl。
- 不修改 IRQ affinity、CPU governor、K8s 节点配置。
```

### 10.2 进阶方案

特点：

- 同时调整 CPU range、PyTorch 线程池、DataLoader worker。
- 通常需要重启业务。
- 必须有回滚方式。

示例：

```text
进阶方案：
- 每个 rank 独占一组本地 NUMA physical cores。
- OMP_NUM_THREADS 设置为该 rank 分配的 physical core 数或略小。
- DataLoader num_workers 根据输入 pipeline 压力调整。
```

## 11. Step 8：验证计划

Agent 必须要求优化前后使用相同 workload、相同时长、相同采集方式。

### 训练验证指标

```text
- samples/s
- step time avg / p50 / p90 / p99
- NPU utilization avg / min
- Host CPU utilization by NUMA
- context switch
- CPU migration
- rank 间 step time 方差
```

### 推理验证指标

```text
- QPS
- latency p50 / p90 / p99
- NPU utilization
- Host CPU utilization by NUMA
- error rate
- 实例间 p99 latency 方差
```

### 成功标准示例

```text
本次优化视为有效需要满足至少一项：
- throughput 提升 >= 3%。
- step time p99 下降 >= 5%。
- 推理 p99 latency 下降 >= 5%。
- NPU utilization 波动降低。
- CPU migration 或 context switch 明显下降。
```

阈值只是建议，最终应由业务场景定义。

## 12. Step 9：回滚与安全确认

Agent 输出任何可能改变运行状态的命令时，必须同时输出回滚方式。

### 低风险执行后端

```text
后端一：taskset
- taskset -cp <cpu-list> <pid>

后端二：internal-script
- 内部经过验证的绑核脚本
- 由 BindingExecutor 适配，不绕过 mindstudio-cpu-binding 的前置校验和回滚状态保存
```

回滚：

```bash
taskset -cp <original-cpu-list> <pid>
```

回滚依赖执行前保存的 `rollback-state.json`。回滚前必须检查 PID 是否仍存在，并尽量校验 `/proc/<pid>/stat` starttime，避免 PID 复用导致误回滚。详细机制见 `binding-rollback-design.md`。

### 中风险变更

```bash
OMP_NUM_THREADS=... numactl --cpunodebind=... --membind=... python train.py ...
```

回滚：

```text
恢复原启动命令和原环境变量，重新启动任务。
```

### MVP 禁止自动执行

```text
- 修改 IRQ affinity
- 修改 CPU governor
- 修改 kernel boot 参数
- 修改 K8s 节点级 CPU Manager
- 自动重启线上服务
```

## 13. Agent 状态机

```text
INIT
  -> NEED_USER_INPUT
  -> READY_TO_COLLECT
  -> SNAPSHOT_AVAILABLE
  -> DIAGNOSIS_READY
  -> REPORT_READY
  -> PLAN_READY
  -> WAITING_FOR_APPLY_CONFIRMATION
  -> APPLIED
  -> ROLLBACK_AVAILABLE
  -> WAITING_FOR_VALIDATION
  -> VALIDATION_REPORTED
  -> CASE_ARCHIVED
```

状态说明：

| 状态 | 说明 |
|------|------|
| `INIT` | 用户刚提出问题。 |
| `NEED_USER_INPUT` | 缺少 PID、场景或目标。 |
| `READY_TO_COLLECT` | 已能生成采集命令。 |
| `SNAPSHOT_AVAILABLE` | 已获得 Snapshot。 |
| `DIAGNOSIS_READY` | 已执行诊断规则。 |
| `REPORT_READY` | 已生成 HTML 报告和文本摘要。 |
| `PLAN_READY` | 已生成保守/进阶方案、应用命令和回滚命令。 |
| `WAITING_FOR_APPLY_CONFIRMATION` | 等待用户确认是否通过选定执行后端执行低风险绑核。 |
| `APPLIED` | 已执行用户确认的临时绑核动作，并验证 affinity。 |
| `ROLLBACK_AVAILABLE` | 已保存 rollback-state，可安全回滚。 |
| `WAITING_FOR_VALIDATION` | 等待用户应用方案并采集优化后指标。 |
| `VALIDATION_REPORTED` | 已完成 before/after 对比。 |
| `CASE_ARCHIVED` | 规则或案例已沉淀。 |

## 14. 典型交互示例

### 14.1 信息不足

用户：

```text
这个 NPU 训练任务 Host 侧好像有 CPU 瓶颈，帮我看看。
```

Agent：

```text
你要分析的是哪类任务？

1. PyTorch 训练
2. PyTorch 离线/批量推理
3. vLLM / SGLang / OpenAI API 这类 LLM Serving
4. 不确定，先帮我自动发现
```

### 14.2 需要先发现候选

用户：

```text
训练，目标是吞吐，但我不确定哪些 PID 对应哪些 NPU。
```

Agent：

```text
可以先运行只读进程发现命令。它只读取 `ps` 和 `npu-smi`，用于列出可能的 rank/main 进程与 NPU 候选映射，不会修改系统状态。

是否执行？

1. 执行
2. 先不执行，我提供已有 snapshot
3. 先不执行，我手动提供 PID

命令：
cd skills/mindstudio-cpu-binding
python scripts/process_discovery.py --out out/processes.json
```

### 14.3 候选确认后可采集

用户：

```text
确认候选：rank0=12345:npu0,rank1=12346:npu1。
```

Agent：

```text
现在可以运行完整 Snapshot 只读采集。它会读取目标进程、线程、CPU/NUMA/NPU、cgroup 和 runtime 信息并生成 snapshot.json，不会修改系统状态。

是否执行？

1. 执行采集
2. 暂不执行
3. 改用已有 snapshot

命令：
cd skills/mindstudio-cpu-binding
python scripts/cli.py collect --pid 12345 --pid 12346 --scenario training --framework pytorch --device-type npu --optimization-goal throughput --rank-map rank0=12345:npu0,rank1=12346:npu1 --sample-seconds 10 --out out/snapshot.json
```

### 14.4 Snapshot 信息缺失

Agent：

```text
Snapshot 缺少 NPU -> NUMA 映射，因此不能判断 rank 是否绑定到 NPU 本地 NUMA；但可以确认 PID 12345 和 12346 都允许运行在 0-127 全部 CPU 上，当前没有有效 CPU 绑定。
```

### 14.5 输出建议

Agent：

```text
保守方案是先按 cgroup 有效 CPU 范围为两个 rank 切分不重叠 CPU range，并保持原 PyTorch 线程配置不变；进阶方案是在切分 CPU range 后同步调整 OMP_NUM_THREADS 和 DataLoader num_workers。建议先跑保守方案并对比 step time p99、NPU utilization 和 CPU migration。
```

## 15. 与后续形态的关系

### Skill 形态

`agent-workflow.md` 可直接转化为 Skill 的工作流说明。

### MCP 形态

Step 3 和 Step 4 可替换为 MCP 工具调用，Snapshot schema 不变。

### Plugin 形态

Plugin 可打包：

```text
- Agent workflow
- Diagnosis rules
- Report template
- Collector / MCP tools
- Example cases
```

## 16. MVP 验收标准

Agent 工作流 MVP 应满足：

1. 能从用户输入判断是否具备采集条件。
2. 能生成只读采集命令。
3. 能读取 Snapshot 并识别信息缺口。
4. 能按诊断规则输出问题、证据、建议、风险和验证方式。
5. 能区分保守方案和进阶方案。
6. 仅在用户确认后执行低风险临时绑核，且执行前保存 rollback-state。
7. 能设计 before/after 验证计划。
