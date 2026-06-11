# 提问流程设计

## 1. 目标

本文定义 `mindstudio-cpu-binding` 在进入采集前如何向用户提问。提问不是固定问卷，而是一套面向真实问题场景的协议：先识别 PyTorch training、PyTorch offline/batch inference 或 LLM serving，再用贴近用户异常指标的问题收集必要上下文，并把答案映射到采集命令参数或诊断规则输入。

设计意图：

- 用最少的问题让流程进入 `READY_TO_COLLECT`。
- 不询问采集器自己能拿到的信息。
- 缺失信息不阻塞流程，而是降级标记到 `availability.missing`。
- 答案能映射到当前原型命令或未来完整 Snapshot collector 参数。

## 2. 五条提问原则

1. **一次只问一个问题**：不要一次列出 3-5 个问题；下一问取决于用户上一轮回答。
2. **优先使用选项**：选择题优先于开放问题；每个选择题都应包含“不确定 / 先帮我看 / 使用默认值”一类兜底选项。
3. **每个问题都必须有用途**：要么映射到采集命令参数，要么是某条诊断规则的输入。不为问而问。
4. **专业术语后置**：不要把 TTFT、TPOT、QPS、p99、TP rank、NUMA、cgroup、OMP、MKL 等术语作为第一层问题；先用用户语言描述，必要时再解释术语。
5. **候选即确认**：PID、rank、worker、instance 到 NPU 的映射优先由只读发现命令生成候选，Agent 说明命令意义并请用户选择是否执行；用户通常只需要确认候选，而不是手工推导完整映射。

## 3. 问题分层

### 3.1 先识别问题场景

不要机械地从“训练还是推理”开始。先从用户描述识别实际问题：

| 用户描述 | 场景 | 关键指标 |
|----------|------|----------|
| step time 抖动、samples/s 低、rank 间不均衡、DDP/HCCL 波动 | PyTorch training | samples/s、step time、rank variance、NPU utilization |
| batch inference 吞吐低、离线推理 latency 抖动、单机多推理进程抢 CPU | PyTorch offline/batch inference | batch throughput、latency、QPS、NPU utilization |
| vLLM-Ascend、SGLang、OpenAI API server、TTFT、TPOT、tokens/s、QPS、p99、timeout | LLM serving inference | TTFT、TPOT、tokens/s、QPS、p99、queueing latency |

用户已明确 vLLM-Ascend 或 SGLang 时，直接进入 LLM serving 分支，不再泛泛追问“是不是 PyTorch 推理”。

### 3.2 阻塞采集（必问）

缺失时必须先问，问到才能继续。

| 问题 | 用途 | 映射 |
|------|------|------|
| 目标进程候选是否已确认？ | 采集对象 | 用户不知道 PID 时，Agent 先说明并提供只读 `process_discovery.py` 命令：它只读取 `ps` 和 `npu-smi`，用于列出 API server、scheduler、engine/worker、rank 和 NPU 进程候选；用户确认候选后映射为 `--pid`（可重复） |
| 进程、rank、worker 或 instance 到 NPU 的候选映射是否已确认？ | R003 locality 判断、推荐 CPU range | 优先由进程发现结果生成候选 `--rank-map`；若发现结果不足，再请用户从启动命令、编排配置或业务日志补充 |
| 当前最想解决的异常指标是什么？ | 决定验证指标和建议策略 | `--optimization-goal` 与报告验证计划 |

### 3.3 单步选择式提问模板

如果用户描述已经能判断场景，直接进入该场景的下一问；不要重复询问场景。如果无法判断场景，只问一个选择题：

```text
你要分析的是哪类任务？

1. PyTorch 训练
2. PyTorch 离线/批量推理
3. vLLM / SGLang / OpenAI API 这类 LLM Serving
4. 不确定，先帮我自动发现
```

#### PyTorch training：问题目标

```text
你最想改善哪个问题？

1. step time 抖动或长尾高
2. samples/s 或整体吞吐低
3. rank 之间速度不均衡
4. NPU 利用率不稳定
5. 不确定，先帮我看
```

#### PyTorch offline/batch inference：问题目标

```text
你最想改善哪个问题？

1. 批量推理吞吐低
2. 单次请求延迟高
3. 多进程/多实例之间抢 CPU
4. NPU 利用率不稳定
5. 不确定，先帮我看
```

#### LLM serving inference：问题目标

```text
你最想改善哪个问题？

1. 首字响应慢
2. 单个 token 生成慢
3. tokens/s 或 QPS 低
4. 延迟抖动或 p99 高
5. timeout / 请求失败
6. 不确定，先帮我看
```

#### 运行位置

```text
这个任务大概运行在哪里？

1. 直接在服务器上
2. Docker/容器里
3. K8s/容器平台里
4. Slurm/作业调度里
5. 不确定
```

#### 只读发现确认

```text
我可以先运行只读进程发现命令来找候选 PID 和 NPU 映射。
它只读取 `ps` 和 `npu-smi`，不修改系统状态。

是否执行？

1. 执行
2. 先不执行，我提供已有 snapshot
3. 先不执行，我手动提供 PID
```

#### 候选确认

发现命令输出后，只问用户选择候选，不要求用户解释底层映射：

```text
发现以下候选服务/进程，你要分析哪一组？

1. vLLM 服务 A：main + engine + TP workers
2. Python/NPU 进程组 B
3. 全部候选都分析
4. 都不是，我手动指定
```

#### Snapshot 采集确认

```text
我将对已确认的 PID 做只读 Snapshot 采集。
它会读取 `/proc`、`/sys`、cgroup、NPU topology 和 runtime 信息，不修改系统状态。

是否执行？

1. 执行采集
2. 暂不执行
3. 改用已有 snapshot
```

### 3.4 推荐信息（缺失给默认值并标记待确认）

这些信息只在需要时单独提问，不作为第一轮批量问题。

| 信息 | 低心智问法 | 用途 | 缺失默认 |
|------|------------|------|----------|
| 运行位置 | “这个任务大概运行在哪里？”并给出服务器/容器/K8s/Slurm/不确定选项 | 影响 cgroup 字段可信度与建议边界 | `unknown` 并标记待确认 |
| 优化目标 | “你最想改善哪个问题？”并给出场景化选项 | 影响 SMT 策略、绑核宽窄、range 重叠判定 | 从异常指标推断；无法推断时默认 `stability` |
| 多实例/多 worker | 先通过只读发现生成候选，再问“要分析哪一组？” | R008 CPU range 重叠、实例隔离判断 | 标记缺失，建议补充 |

### 3.5 可选补充（影响精度，不阻塞）

缺失进入 `availability.missing`，不阻断采集。只有在 Snapshot 采集后仍无法判断时才补问；补问时要说明“不提供也可以继续”。

| 场景 | 可选补充 | 低心智问法 | 缺失影响 |
|------|----------|------------|----------|
| PyTorch training | DataLoader、OpenMP/MKL/torch 线程、samples/s、step time | “是否有训练吞吐或 step time 数据？没有也可以先继续。” | R006 和收益量化偏保守 |
| PyTorch offline/batch inference | batch size、并发进程数、线程设置、latency、QPS | “是否有吞吐或延迟数据？没有也可以先继续。” | 不能精确区分 batch 侧吞吐和 CPU 侧竞争 |
| LLM serving | TP/DP/PP、实例数、tokenizer、TTFT/TPOT/tokens/s/QPS/queueing latency | “是否有压测结果或服务指标？没有也可以先继续。” | 难以区分 API server、scheduler、tokenizer、engine worker 的 CPU 瓶颈 |
| 所有场景 | 是否执行低风险临时绑核 | “是否只生成方案，还是在确认风险和回滚后尝试临时绑核？” | 默认 `dry-run`，只生成不执行 |

### 3.6 不该问（采集器自取）

以下信息由采集器从 `/proc`、`/sys`、cgroup、`npu-smi` 获取，**不应**要求用户手填，否则既增加负担又显得不专业。

```text
- CPU / NPU / NUMA 拓扑
- 当前进程/线程 Cpus_allowed_list、current_cpu
- cgroup cpuset_cpus_effective、cpu quota、throttling
- NPU -> NUMA locality（npu-smi 可取时）
- 运行期 CPU 使用率、Top 线程
```

## 4. 分支逻辑

提问应按用户已暴露的症状动态调整；凡是 PID、worker、rank、instance 到 NPU 的映射，都优先进入“只读发现 -> 候选确认”，不要直接要求用户手工给完整映射：

```text
workload?
├── PyTorch training
│   ├── step time 抖动或长尾高? -> 问运行位置，然后进入只读发现
│   ├── samples/s 或整体吞吐低? -> 问运行位置，然后进入只读发现
│   └── rank 速度不均衡? -> 问运行位置，然后进入只读发现生成 rank/PID/NPU 候选
├── PyTorch offline/batch inference
│   ├── 批量吞吐低? -> 问运行位置，然后进入只读发现
│   ├── 单次请求延迟高? -> 问运行位置，然后进入只读发现
│   └── 多进程/多实例抢 CPU? -> 问运行位置，然后进入只读发现生成实例/PID/NPU 候选
└── LLM serving inference
    ├── 首字响应慢? -> 关注 scheduler、prefill、queueing、tokenizer CPU
    ├── 单个 token 生成慢? -> 关注 decode worker、engine CPU、NPU locality
    ├── tokens/s 或 QPS 低? -> 关注多 worker/多实例 CPU range 和 cpuset
    └── 延迟抖动/请求失败? -> 关注 CPU 迁移、context switch、cgroup throttling

deployment?
├── 直接在服务器上 -> 记录为 baremetal
├── Docker/容器里 -> 记录为 container，cgroup/cpuset 以采集结果为准
├── K8s/容器平台里 -> 记录为 kubernetes，cgroup/cpuset 以采集结果为准
├── Slurm/作业调度里 -> 记录为 slurm，cgroup/cpuset 以采集结果为准
└── 不确定 -> 记录为 unknown，继续只读采集并在报告中标记待确认
```

## 5. 提问节奏

- 每轮只问一个问题；不要使用“我先确认 3 点 / 4 点 / 5 点”的批量问法。
- 优先用选择题；选择项使用用户语言，并提供“不确定，先帮我看”或“使用默认值”选项。
- 先识别场景；如果用户描述已足够判断场景，直接进入该场景的问题目标选择，不重复问场景。
- 再问问题目标、运行位置、是否执行只读发现、候选确认、是否执行只读采集；每一步都等用户回答后再进入下一步。
- 用户已经提供的信息不要重复问。
- 用户不知道 PID 或 NPU 映射时，不要求用户手工猜测；Agent 先解释 `process_discovery.py` 只读取 `ps` 和 `npu-smi`、用于生成候选且不修改系统状态，再让用户选择是否执行；发现后基于候选 API server、scheduler、tokenizer、engine worker、rank/worker、NPU 进程让用户确认目标。
- 能从 Snapshot 得到的信息，不在提问阶段索取。
- 缺少 NPU topology 时不猜测 NUMA locality，标记缺失。
- LLM serving 场景不要默认追问 DataLoader；第一层问题用“首字响应慢、单个 token 生成慢、吞吐低、延迟抖动、请求失败”等用户语言表达，必要时再在报告中映射到 TTFT、TPOT、tokens/s、p99、timeout。
- 如果用户只想看报告模板或架构，不进入提问与采集流程。

## 6. 选择结果到采集命令的映射

当前阶段先使用已存在的只读原型命令辅助确认采集输入；Agent 应说明命令意义和安全边界，让用户选择是否执行。以下带 `samples/*.txt` 的命令仅用于仓库离线示例验证，真实 Linux NPU 节点请使用后面的 live 只读命令或完整 collect 命令。以下命令默认在 Skill 目录执行：

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

真实 Linux NPU 设备上可用独立原型做只读 live 验证：

```bash
cd skills/mindstudio-cpu-binding
python scripts/process_discovery.py --out out/processes.json
python scripts/topology_collect.py --out out/topology.json
```

当前完整 Snapshot 采集原型命令：

```bash
cd skills/mindstudio-cpu-binding
python scripts/cli.py collect \
  --pid 12345 \
  --scenario training \
  --framework pytorch \
  --device-type npu \
  --optimization-goal throughput \
  --sample-seconds 10 \
  --out out/snapshot.json
```

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

参数对应关系见 `collector-design.md` 的 CLI 设计。

## 7. 完整性检查表

进入采集前用此表判断是否就绪。

| 检查项 | 必需级别 | 缺失时动作 |
|--------|----------|------------|
| 工作负载场景：PyTorch training / PyTorch offline inference / LLM serving | 必需 | 从用户描述推断；无法推断时询问 |
| 目标 PID | 必需 | 用户不知道时，解释并运行只读 `process_discovery.py` 生成候选，用户确认后进入采集 |
| rank / worker / instance -> NPU 映射 | 多卡、多 worker、多实例时必需 | 优先从进程发现和 `npu-smi` 候选生成，用户确认；发现不足时再标记缺失或请求补充 |
| 主要异常指标 | 推荐 | 从用户描述推断；无法推断时默认 `stability` 并标记待确认 |
| 运行环境 | 推荐 | 默认 `unknown` 并标记待确认 |
| NPU -> NUMA 映射 | 诊断 locality 时必需 | 由采集器获取，不向用户索取 |
| PyTorch / runtime / serving 线程配置 | 推荐 | 采集 environ 和线程信息，缺失则标记 |
| 性能基线 | 验证时必需 | 报告中标记缺失 |
| 执行后端与确认意愿 | 仅在拟执行绑核时必需 | 默认 `dry-run` |

## 8. 典型对话示例

### 8.1 PyTorch 训练信息不足

用户：

```text
这个 NPU 训练任务 step time p99 抖得很厉害，帮我看看是不是 Host CPU 绑核问题。
```

Agent：

```text
这是 PyTorch training 的 step time 抖动场景。先确认运行位置：

这个任务大概运行在哪里？

1. 直接在服务器上
2. Docker/容器里
3. K8s/容器平台里
4. Slurm/作业调度里
5. 不确定
```

### 8.2 LLM serving 信息不足

用户：

```text
vLLM-Ascend 服务最近 p99 和 TTFT 都很高，tokens/s 也不稳定。
```

Agent：

```text
这是 LLM serving 场景，我不会按普通 PyTorch batch inference 来问。先确认你最想改善的问题：

1. 首字响应慢
2. 单个 token 生成慢
3. tokens/s 或 QPS 低
4. 延迟抖动或 p99 高
5. timeout / 请求失败
6. 不确定，先帮我看
```

### 8.3 具备采集条件

用户：

```text
训练，PID 12345 和 12346，rank0 用 npu0，rank1 用 npu1，目标是吞吐。
```

Agent：

```text
可以运行只读采集，不会修改系统状态。采集命令应包含两个 PID、training 场景、PyTorch/NPU 框架信息、rank-map 和输出 snapshot.json。生成命令前我会优先使用本 Skill 包内或项目约定的只读采集脚本路径。
```

### 8.4 缺失项降级

Agent：

```text
你没提供 NPU -> NUMA 映射和性能基线。我会标记为信息缺口：本次不能判断 worker/NPU/NUMA locality 是否匹配，也不能量化优化收益；但仍可判断进程是否未绑核、是否受 cgroup/cpuset 限制、是否跨 NUMA 运行以及线程是否过载。
```

### 8.5 LLM serving 不应追问 DataLoader

用户：

```text
SGLang 服务 TPOT 高，decode tokens/s 不稳定。
```

Agent 应优先问一个选择题：

```text
你最想改善哪个问题？

1. 首字响应慢
2. 单个 token 生成慢
3. tokens/s 或 QPS 低
4. 延迟抖动或 p99 高
5. timeout / 请求失败
6. 不确定，先帮我看
```

Agent 不应默认问：

```text
DataLoader num_workers 是多少？
```
