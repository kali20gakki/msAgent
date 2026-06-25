---
name: ascend-cluster-fast-slow-rank-detector
description: 专门用于 Ascend 集群 Profiling 性能数据的“快慢卡”诊断专家技能。当用户提供【集群性能数据目录/路径】并要求分析【快慢卡】、【慢节点】、【负载不均衡】或【集群瓶颈】时，必须触发此技能。该技能会自动接收集群路径，调度相关工具输出快慢卡的宏观定性与微观根因（如 Host 下发瓶颈、算子计算劣化）。
---

# 集群快慢卡诊断

## 1. 技能目标
在 Ascend 多卡/集群场景下，利用msprof-analyze命令工具结合专家规则，自动识别因计算、通信或 Host 下发导致的性能瓶颈卡（慢卡），并下钻定位微观根因。

## 2. 诊断先验知识库 (Expert Rules)
禁止仅凭单项指标字面意思下结论，必须严格遵守以下华为官方诊断逻辑：

* **【Host 下发瓶颈 (伪快卡)】**
    * **现象**：某卡（Rank X）的 `Free Time` 极长（占比 > 10% 或远超均值），且 `Compute` 和 `Communication` 时间异常偏短。
    * **定性**：**Rank X 绝非快卡，而是导致集群阻塞的“慢卡”。** CPU 下发慢导致其 NPU 饿死（产生巨大 Free Time）。当它终于发起通信时，其他卡已等待多时（其他卡 Wait 长），故其通信瞬间完成。
    * **动作**：调用 `scripts/compare_api_stats.py`，重点观察 `launch`、`aclrtSynchronizeDevice` 等下发/同步 API 的耗时与间隙差异。
* **【纯计算快慢卡】**
    * **现象**：各卡 `Free Time` 普遍较短且均匀，但某卡 `Compute Time` 显著大于均值。
    * **定性**：计算型慢卡。若单算子调用次数 (`count`) 不同，为负载切分不均；若次数相同但平均耗时 (`avg_time`) 激增，为算子硬件劣化或动态 Shape 导致。
    * **动作**：调用 `scripts/compare_op_stats.py` 对比算子执行差异。
* **【通信/慢链路瓶颈】**
    * **现象**：各卡通信带宽远低于理论值（如 SDMA < 2GB/s）。
    * **定性**：通常为小包通信（ZeRO3 切分过细）、SDMA 地址未对齐或硬件问题。

## 3. 标准操作流程 (SOP)
请严格按照以下流程执行：

1. 输入数据类型判断

先判断用户提供的路径或文件属于哪一类，并把后续分析所需证据统一整理为“集群级宏观证据”和“Rank 级明细证据”两类：

（1）明确当前输入数据的类型
* 路径下若存在Profiling数据：
  * 共识别到多少个 Rank；
  * Profilng数据的类型是DB or Text，若路径下存在ascend_pytorch_profiler_{rank_id}.db文件，则为 DB 类型数据，否则为 Text；
  * 各 Rank 的 profiling 文件夹是否齐全。
* 路径下是否已存在 `msprof-analyze` 的分析结果目录`cluster_analysis_output`：若存在，输出已包含哪些内容
* 告知用户，当前数据类型、Rank数量等信息

（2）路由后续操作
* 用户给的输入数据是 text 格式：请参考 `skills/ascend-msprof-analyze-cli` SKILL 的能力二：专家建议分析 (Advisor) 部分进行分析，然后直接进入流程 3 基于证据做快慢卡判定
* 用户给的输入数据是 db 格式，判断是否已经存在`cluster_analysis_output` 结果文件夹：
  * 若存在，检查是否包含 `cluster_time_summary`、`compute_op_sum`、`hccl_sum`、`slow_rank`、`slow_link`、`cann_api_sum` 等结果。记录缺失项，直接进入流程 2 调用 `msprof-analyze` 集群分析能力补齐；如果结果已完整，直接进入流程 3 基于证据做快慢卡判定
  * 不存在，直接进入流程 2 调用 `msprof-analyze` 集群分析能力
* 用户输入只存在 `cluster_analysis_output` 结果文件夹：直接进入流程 3 基于证据做快慢卡判定

2. 调用 `msprof-analyze` 集群分析能力
流程 2 只负责生成分析结果，不负责直接下结论。  
必须调用 `msprof-analyze` 执行集群分析。请参考 `skills/ascend-msprof-analyze-cli` SKILL 中 “集群综合分析 (Cluster)” 的部分
**要求不是只跑 `all`，而是将 README 中与集群分析相关的 `-m` 能力逐项跑全**，至少覆盖下列能力：

| 分析能力 | 介绍 |
|--|--------|
| all | 组合能力，可作为补充，但**不能替代逐项执行**。 |
| cluster_time_summary | 提供集群训练过程中迭代耗时拆解，作为判断 Free/Compute/Communication/Wait 异常的宏观入口。 |
| compute_op_sum | 汇总 device 侧计算类算子，用于定位计算型慢卡和负载切分不均。 |
| hccl_sum | 汇总通信类算子，用于定位通信型慢卡、慢链路和通信带宽异常。 |
| slow_rank | 展示各 Rank 的快慢卡影响次数和慢卡候选原因。 |
| slow_link | 分析集群异常耗时链路，用于辅助识别慢链路或通信瓶颈。 |
| cann_api_sum | 汇总 CANN 层 API，用于定位 Host 下发、同步点和 launch 间隙问题。 |

命令示例：
```bash
msprof-analyze -m cluster_time_summary -d ./cluster_data -o ./output/cluster_time_summary
```

如环境允许，优先为每个 `-m` 能力使用独立输出目录，避免不同分析结果互相覆盖。若某项能力执行失败，必须记录失败命令、错误摘要和缺失影响，后续结论中不得把该项当成已验证证据。

3. 基于证据做快慢卡判定

流程 3 只在证据集可用后执行。agent 不能只引用单项指标直接给结论，必须综合流程 1/2 得到的宏观证据和必要的 Rank 级明细证据，明确回答以下问题：

（1）**是否存在快慢卡现象**；  
（2）**真正的慢卡 Rank ID 是谁**，以及候选快卡 Rank ID 是谁；  
（3）**问题属于哪一类**：
   * Host 下发慢 / 调度瓶颈；
   * 计算型慢卡；
   * 通信型慢卡 / 慢链路；
   * 负载不均衡；
   * 多种问题叠加；
   * 证据不足，暂不能判定。

（4）**判定依据是什么**：至少引用 `cluster_time_summary`、`slow_rank` 或 `slow_link` 中的一类宏观证据；若涉及计算型或 Host 下发瓶颈，还应继续调用流程 4 中的对比脚本，用 `compare_op_stats.py` 或 `compare_api_stats.py` 给出微观证据。


4. 快卡vs慢卡比对

对比脚本统一存放在本技能目录的 `scripts/` 文件夹中，支持自动发现集群目录或手动指定文件（优先 CSV，次选 DB）。

**核心命令模板：**
```bash
# 【计算类瓶颈】调用算子对比脚本（将 <本技能目录> 替换为 get_skill 返回中的路径）
python <本技能目录>/scripts/compare_op_stats.py <集群数据根目录> <慢卡RankID> <快卡RankID> [--top N]

# 【下发类瓶颈】调用 API 对比脚本
python <本技能目录>/scripts/compare_api_stats.py <集群数据根目录> <慢卡RankID> <快卡RankID> [--top N]
```
参数说明：
* cluster_path: 集群数据根目录（包含 profiler_info_{rank}.json）。 
* slow_rank / fast_rank: 慢卡与快卡（基准）的 Rank ID。 
* --top N: （可选）输出差异最大的前 N 条，默认 20。 
* --slow-path / --fast-path: （可选）当集群自动发现机制报错时，用于手动指定慢/快卡的 *.csv 或 *.db 绝对路径。 
* --json: （可选）以 JSON 格式结构化输出。

5. 对慢卡目录执行 advisor，并给出调优建议（可选）

当流程3已锁定慢卡 Rank ID 后，必须进入该慢卡对应的 profiling 文件夹，再执行：

```bash
msprof-analyze advisor all -d <slow-rank-profiling-dir>
```

然后基于 advisor 输出，给出该慢卡的**针对性调优建议**。建议内容必须与瓶颈类型对应，例如：

* Host 下发慢：排查下发线程绑核、同步点、launch 间隙、CPU 饥饿、数据准备阻塞；
* 计算型慢卡：排查算子 count 不一致、动态 shape、算子劣化、融合缺失、AICore 利用率问题；
* 通信型慢卡：排查小包通信、并行切分策略、链路带宽、通信域配置、SDMA/HCCL 异常；
* 负载不均衡：排查数据切分、专家负载、pipeline stage 分配、rank 侧 workload 偏斜。

若 advisor 输出与流程3的集群判断不一致，必须显式指出这一点，并说明以哪类证据为主、为什么。
