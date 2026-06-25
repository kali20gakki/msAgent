---
name: ascend-msprof-analyze-cli
description: MindStudio Profiler Analyze（msprof-analyze）是面向 AI 训练与推理场景的性能分析工具，基于采集得到的 profiling 数据进行统计、比对和诊断，帮助定位计算、通信、调度及集群场景下的性能瓶颈。
---

# Ascend 性能数据综合分析

目前提供两个功能，根据用户需求自动选择对应工具：

| 能力                     | 工具命令 | 侧重点                                                |
|------------------------| --- |----------------------------------------------------|
| 集群分析 (cluster analyse) | `msprof-analyze -m <mode>` | 涵盖集群多维信息汇总、拆解对比、通信瓶颈定位和下发问题分析等专题。 |
| 专家建议 (advisor)         | `msprof-analyze advisor <subcommand>` | 基于性能数据自动识别计算、调度、通信等潜在问题，并输出优化建议。                   |

---

## 功能选择指南

> ⚠️ **重要**：在开始分析前，必须先根据以下规则确定使用哪个功能。

### 选择规则

| 条件                        | 优先使用的分析能力 | 原因                      |
|---------------------------| --- |-------------------------|
| 卡数 ≥ 64 或数据量很大（db大于1G）    | 集群分析 | 集群分析专门优化了大数据量场景，能快速汇总数据 |
| 数据格式为 **db**              | 集群分析 | 集群分析仅支持 db 格式           |
| 数据格式为 **text** (json/csv) | 专家建议 | 专家建议支持 text 和 db 格式     |
| 需要分析慢卡/慢链路原因              | 集群分析后，再用专家建议 | 先定位慢卡，再用专家建议深入诊断        |
| 需要算子优化建议                  | 专家建议 | 专家建议输出详细的调优建议           |
| 单卡数据一般以'ascend_pt'或者'ascend_ms'结尾  | 专家建议 | 直接使用专家建议进行诊断            |

### 典型决策流程

```
用户提出分析需求
        │
        ▼
┌─────────────────────────────┐
│ 数据量评估：卡数、数据规模    │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ 数据格式判断：db 还是 text？  │
└─────────────────────────────┘
        │
        ├──────────────────────────────┐
        ▼                              ▼
   【db格式且为多卡数据】                    【text格式或者单卡数据】
        │                              │
        ▼                              ▼
  优先集群分析                  只能专家建议
        │                        (跳过第3步)
        ▼
┌─────────────────────────────┐
│ 集群分析是否找到慢卡？        │
└─────────────────────────────┘
        │
        ├─────────────┬─────────────┐
        ▼             ▼             ▼
     是慢卡        不是慢卡        无瓶颈
        │             │             │
        ▼             │             ▼
  对该慢卡用      分析其他问题     结论输出
  专家建议深入
```

### 场景与能力匹配表

| 用户需求场景 | 推荐能力 | 理由 |
| --- | --- | --- |
| "分析这个 16 卡集群的慢卡问题" | 集群分析 → 专家建议 | 先定位慢卡，再深入诊断 |
| "这个单卡的数据有算子执行慢" | 专家建议 | 单卡场景专家建议更全面 |
| "帮我看看这个 profiling 数据" | 先判断数据格式 | 格式决定能力选择 |
| "通信带宽低是什么原因" | 集群分析 | 通信矩阵分析是集群分析强项 |
| "这个算子为什么慢" | 专家建议 | 算子问题诊断是专家建议强项 |
| "数据格式是 csv 的" | 专家建议 | 集群分析仅支持 db |

---

## 能力一：集群综合分析 (Cluster)

### 1.1 技能目标
在 Ascend 多卡/集群场景下，利用 msprof-analyze 的 `-m` 参数指定分析能力，对集群训练数据进行综合分析。用户只需说明要分析什么，系统自动选择对应的 `-m` 参数执行分析。

### 1.2 分析模式选择 (-m 参数)

| 分析能力 | 介绍 |
| --- | --- |
| cluster_time_summary | 提供集群训练过程中迭代耗时的拆解，帮助用户找到性能瓶颈。 |
| compute_op_sum | device侧运行的计算类算子汇总。 |
| freq_analysis | 识别aicore是否存在空闲（频率为800MHz）、异常（频率不为1800MHz或800MHz）的情况并给出分析结果。 |
| ep_load_balance | moe负载信息汇总分析。 |
| communication_time_sum | 集群场景通信时间和带宽汇总分析。 |
| hccl_sum | 通信类算子信息汇总。 |
| slow_rank | 根据当前的快慢卡统计算法，展示各个rank得出的快慢卡影响次数，识别慢卡出现的原因。 |
| cann_api_sum | CANN层API的汇总。 |
| free_analysis | 提供对Device侧大块空闲时间的自动分析能力，能够识别空闲时间产生的原因，帮助用户定位性能问题。 |
| all | 同时解析通信矩阵communication_matrix和通信耗时数据communication_time。 |

### 1.3 集群分析命令模板

```bash
# 【集群综合分析】
msprof-analyze -m <mode> -d <cluster_data_path> [-o <output_path>] [--force] --agent
```

命令示例：
```bash
msprof-analyze -m cluster_time_summary -d ./cluster_data -o ./output --agent
msprof-analyze -m free_analysis -d ./cluster_data -o ./output --agent
msprof-analyze -d ./cluster_data -o ./output --agent # 默认为all
```

### 1.4. 诊断先验知识库 (Expert Rules)

> ⚠️ 集群分析 (Cluster) 专用诊断规则
禁止仅凭单项指标字面意思下结论，必须严格遵守以下华为官方诊断逻辑：

* **【Host 下发瓶颈 (伪快卡)】**
    * **现象**：某卡（Rank X）的 `Free Time` 极长（占比 > 10% 或远超均值），且 `Compute` 和 `Communication` 时间异常偏短。
    * **定性**：**Rank X 绝非快卡，而是导致集群阻塞的"慢卡"。** CPU 下发慢导致其 NPU 饿死（产生巨大 Free Time）。当它终于发起通信时，其他卡已等待多时（其他卡 Wait 长），故其通信瞬间完成。
    * **验证**：使用 `free_analysis` 分析空闲时间原因，使用 `freq_analysis` 检查频率是否异常低。
* **【纯计算快慢卡】**
    * **现象**：各卡 `Free Time` 普遍较短且均匀，但某卡 `Compute Time` 显著大于均值。
    * **定性**：计算型慢卡。
* **【慢链路定位】**
    * **现象**：某条链路的 `Bandwidth(GB/s)` 显著低于同类型链路的均值。
    * **定性**：根据 Transport Type 判断链路类型，LOCAL（片内拷贝，速度最高）> HCCS/PCIE（节点内片间拷贝）> RDMA（节点间拷贝，速度最低）。同类型链路带宽差异过大表示存在慢链路。
    * **验证**：使用 `communication_matrix_sum` 查看各链路带宽。
* **【频率异常】**
    * **现象**：NPU 频率为 800MHz（空闲状态）或非 1800MHz/800MHz。
    * **定性**：频率异常可能表示算子等待或调度问题。
    * **验证**：使用 `freq_analysis` 检测频率异常。
* **【通信瓶颈】**
    * **现象**：通信耗时占比高，快慢卡明显。
    * **定性**：使用 `communication_bottleneck` 分析是 Host 侧下发慢还是 Device 侧计算慢导致通信等待。

### 1.5. 硬性约束 (MUST DO)

1. **必须使用 msprof-analyze 工具**：使用 `msprof-analyze` 命令进行集群分析。
2. **分析能力选择**：根据提示词选择分析模式，如果根据提示词无法确定使用哪个分析能力，请让用户明确使用哪一个分析能力，不能自己尝试。
3. **禁止孤立分析单卡**：在集群场景下，严禁仅分析单个 Rank 数据而不进行多卡对比，最好只基于最后输出的文件进行分析。命令行失败后不能继续读取各个rank的数据。
4. **必须分析时间占比**：输出报告必须包含各 Rank 的计算、通信、内存拷贝、空闲时间占比分析（适用于 cluster_time_summary 模式）。
5. **时间单位统一规范**：所有原始数据（单位为微秒），报告中必须自动换算为毫秒（ms）展示，并明确标注单位。

### 1.6. 标准操作流程 (SOP)

> ⚠️ 集群分析 (Cluster) 专用流程

1. **确认分析模式**：
   - 根据用户需求确定 `-m` 参数值（参见第2节映射表）。
   - 若用户要求"完整分析"或"全部"或没有明确要求分析哪一方面，默认使用 `all` 模式。

2. **执行分析命令**：
   - 执行命令：`msprof-analyze -m <mode> -d <cluster_data> [-o <output_path>] [--force] --agent`
   - 默认情况下，不需要使用`-o`参数，除非用户明确指定输出路径。

3. **读取输出结果**：
   - 根据命令运行返回的结果，读取生成的文件。
   - 如果返回信息有 error，或者 JSON 里 message 为空，不用往下运行。提示让用户自己执行命令（去掉 --agent 参数），查看日志定位原因。
   - 后续的分析可以结合json中的'suggestion'字段。

4. **数据解读与瓶颈定位**：
   - 如果根据表头无法明确数据的含义，可以参考[references/recipe_output_format_introduct.md](references/recipe_output_format_introduct.md)。
   - 对照【先验知识库】综合判断瓶颈类型
   - 如果从数据中无法做出明确结论，不要给出不准确的结论。

5. **输出报告**：按以下结构输出最终回复：
   - **分析概要**：使用的分析模式、执行的完整命令以及输出结果路径等信息。
   - **详细数据**：表格形式展示分析结果
   - **瓶颈定位**：瓶颈类型、关键证据（引用具体数值和占比）
   - **优化建议**：针对性建议

### 1.7 集群分析故障

* **【时间单位混淆】**
  - **现象**：输出的时间字段单位不明确。
  - **规避**：所有时间相关字段统一使用毫秒（ms），在报告中明确说明。
* **【profiler_level 设置过低】**
  - **现象**：无法获取通信带宽和通信矩阵信息。
  - **规避**：profiler_level 建议设置为 Level1 或更高。
* **【db 类型数据缺失】**
  - **现象**：Recipe 分析能力无法使用，报错缺少 db 文件。
  - **规避**：确认使用 db 类型数据，Ascend PyTorch Profiler 需指定 `export_type=["db"]`。
* **【Rank/Step ID 无效】**
  - **现象**：指定 --rank_list 或 --step_id 后无输出或报错。
  - **规避**：确认配置的 ID 在实际数据中存在。

---

## 能力二：专家建议分析 (Advisor)

### 2.1 技能目标
基于 Ascend PyTorch Profiler 或 MindSpore Profiler 采集的性能数据，使用 msprof-analyze advisor 功能进行自动分析，并输出性能调优建议。

### 2.2 分析模式

使用 `msprof-analyze advisor <subcommand>` 命令格式，支持三种分析模式，默认使用`all`：

| 子命令           | 说明 | 包含功能 |
|---------------| --- | --- |
| `all`         | 总体性能瓶颈 | 全部功能（计算 + 通信 + 调度） |
| `computation` | 计算瓶颈 | computation + Kernel compare |
| `schedule`    | 调度瓶颈 | schedule + API compare |

### 2.3 功能详情表

| 维度 | 功能 | 说明 |
| --- | --- | --- |
| overall | Overall Summary | 计算、通信、空闲等维度拆解 |
| overall | Environment Variable Issues | 环境变量设置推荐 |
| overall | slow rank | 慢卡识别 |
| overall | slow link | 慢链路识别 |
| computation | AICPU Issues | AI CPU 调优 |
| computation | Operator Dynamic Shape Issues | 识别动态Shape算子 |
| computation | AI Core Performance Analysis | MatMul、FlashAttentionScore 等算子分析 |
| computation | Block Dim Issues | Block Dim 算子调优 |
| computation | Operator No Bound Issues | 算子瓶颈分析 |
| computation | Fusion Issues | 融合算子图调优 |
| computation | AI Core Frequency Issues | AI Core 算子降频分析 |
| communication | Packet Analysis | 通信小包检测 |
| communication | Bandwidth Contention Analysis | 通信计算带宽抢占检测 |
| communication | Communication Retransmission Analysis | 通信重传检测 |
| communication | Byte Alignment Analysis | 通信算子字节对齐检测 |
| schedule | Affinity API Issues | 亲和API替换调优 |
| schedule | Operator Dispatch Issues | 识别算子下发问题(路径3/路径5) |
| schedule | SyncBatchNorm Issues | BatchNorm同步检测 |
| schedule | Synchronize Stream Issues | 流同步检测 |
| schedule | GC Analysis | 垃圾回收事件检测 |
| schedule | Fusible Operator Analysis | 检测 Host/MTE 瓶颈算子序列 |
| dataloader | Slow Dataloader Issues | 异常 dataloader 检测 |
| memory | Memory Operator Issues | 识别异常的内存申请释放操作 |
| comparison | Kernel compare | 标杆性能数据 Kernel 对比 |
| comparison | API compare | 标杆性能数据 API 对比 |

### 2.4 Advisor 命令模板

```bash
# 【总体性能瓶颈】
msprof-analyze advisor <subcommand> -d <profiling_path> [-o <output_path>] [-bp <benchmark_path>] [--force] --agent
```

命令示例：
```bash
# 单卡场景）
msprof-analyze advisor all -d ./profiling_data -o ./output --force --agent
# 集群场景
msprof-analyze advisor all -d ./cluster_profiling -o ./output --force --agent
```

### 2.5 数据路径要求

| 场景 | 路径要求 |
| --- | --- |
| 单卡 | 指定到 `*_ascend_pt` 或 `*_ascend_ms` 目录 |
| 多卡/集群 | 指定到 `*_ascend_pt` 或 `*_ascend_ms` 的父目录 |
