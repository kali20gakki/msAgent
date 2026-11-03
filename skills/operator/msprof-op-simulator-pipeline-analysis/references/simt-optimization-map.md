# SIMT 与混合编程：从流水位置到线程级原因

## 先判断数据走哪条路径

从源码与实际启动接口确认 SIMT Kernel 或 `__simt_vf__`，并记录线程块数、每块线程数、编译期最大线程数及索引公式。外层 SIMD 的核数、SIMT VF 的线程数和独立 SIMT Kernel 的 grid/block 不是可互换的参数。

- **直接访问 GM**：SIMT 线程的 GM load/store 在 VF 内发生，MTE2/MTE3 可以没有有效搬运。先查线程地址合并、缓存与指令效率，不能将“没有 MTE 重叠”判为单缓冲问题。
- **UB 中转**：外层 MTE2 → SIMT VF(UB) → MTE3，分别检查外层数据块依赖和内部 SIMT 访存；前者适用[共同参考](pipeline-analysis.md)，后者按本表分析。
- **SIMD/SIMT 混合**：按 VF 片段分别归因；出现 PUSHQ、VECTOR、RVEC 不能独自证明是 SIMT。AIC/AIV MIX 的矩阵同步是另一类协作，不直接对应 Warp 发散。

## SIMT 技巧与流水证据

| 现象 / 位置 | 定因前的补证 | 官方技巧与适用改动 | 最小对照应验证什么 |
| --- | --- | --- | --- |
| 少数核持续工作，或大量短任务反复调度 | 全核覆盖、线程块数、物理核数、每线程工作量；仅一个核或采样核不能定因 | [线程块配置][grid]：结合规模调整 gridDim，必要时用 grid-stride 循环摊薄调度 | 对相同逻辑元素比较核间尾部、调度次数与完整耗时；不能将“块越多越快”或固定64核作为通用结论 |
| VF 内出现大量栈 load/store，调线程数后耗时反常 | 目标版本编译器资源报告、最大线程数、实际线程数、反汇编栈访问；区分栈用途 | [线程数与寄存器溢出][register]：调整最大线程数、寄存器需求和局部变量活跃范围 | 栈流量/指令与相关 stall 减少，同时观察并行度、每线程循环代价；实际线程数不得超过编译期上限，不照搬其他架构寄存器档位 |
| GM 访问段长，算术少；MTE 可以不活跃 | 展开同一 Warp 同一指令上各 lane 的地址、stride、有效字节与目标 Cache Line；有计数再查实际流量 | [访存合并][coalescing]：重新设计线程到连续数据的映射，必要时 UB 中转转置 | 更少的访存事务或无效字节、较短的访存阶段；逻辑数据量不变。MTE 的512B建议不是 Warp 合并访问的替代条件 |
| GM 已连续，但 UB 访问或转置阶段仍拖长 | 对照目标型号的 bank/group/subbank 低位映射，计算 Warp 的并发地址；相应证据支持后再归因 | [避免 SIMT UB subbank 冲突][bank]：padding、布局或线程映射 | 冲突相关代价和 UB 阶段减少；记录 padding 占用，不将 SIMD Bank 指标直接等同于 SIMT subbank 冲突 |
| 重复访问少量表数据时 VF 访存成本高 | 热点/流式数据的复用范围、工作集、当前 DCache 路径与支持的缓存策略；只有时序则是候选 | [DCache 策略][dcache]、哈希探测场景的 [SoA 布局][soa] | 热点有效复用增加、无用字段或重载流量减少；不能由 VF 时长计算 Cache 命中率，也不能强制所有数据驻留/旁路 |
| 大量窄 load/store 或标量元素访问支撑很少有效数据 | 指令位宽、结构体/指针对齐、lane 地址、奇数尾块和现有运算形式 | [短向量类型][short]、[类型对齐][align]：语义允许时用 half2 等或调整类型对齐 | 访存/发射指令减少、有效字节比例改善；保持数值与越界约束，宽类型并不自动保证访存合并或总吞吐提高 |
| 控制流热点，VF 忙但有效 lane 少；Warp 结束时间分散 | 同一 Warp 内真实输入对应的分支条件、控制流/掩码、反汇编或可用采样；Warp 寿命差不是发散证明 | [减少 Warp 内分支分歧][branch]、简单分支的 [predicate 优化][predicate] | 分支切换、串行分支或无效 lane 成本减少；比较实际指令，编译器可能已谓词化。不能将有副作用/越界分支改为无条件执行 |
| 整数除法、取模或循环寻址是主要指令热点 | 除数是否固定、类型/符号、输入范围、源码与真实指令；排除只是等待访存 | [快除预计算][division]、提取循环不变量及简化可证明冗余寻址 | 目标指令或依赖链缩短、有效搬运工作可保持不变；验证除零、符号和舍入边界，不能把无符号定除数算法直接套到一般除法 |
| 共享地址原子更新集中，线程进度被更新操作拖住 | 实际 atomic 指令、目标地址分布、返回值是否被使用、数据范围与必要的跨线程语义 | [原子优化][atomic]：寄存器/UB 局部聚合后 GM 汇总，合法时减少未用返回值或选择足够位宽 | GM 原子次数/竞争与相应执行代价下降；记录局部聚合和同步成本，验证溢出、浮点累加顺序和结果，不凭图删除必要原子 |

没有支持的线程级事件或补充计数时，仍可报告外层 VF 时序和源码推导的候选，但不能伪造 Warp stall、lane 效率或 occupancy。资源报告中的 Stack size 与动态 spill 代价也不是同一个量。

PCSampling、Warp Stall、全核 Occupancy 等数据可作为补证，其获取方式和适用范围先从 msopprof 仓库读取；采样占比不是精确周期分解，独立 Warp 重放不能与仿真对齐成同一条时间轴。

## SIMD/SIMT 混合的衔接

| 外层与内部现象 | 有条件的改动 | 观察目标与限制 |
| --- | --- | --- |
| SIMT 直接 GM 的代价高，而连续块可装入 UB | [UB 中转][ub]，再评估可独立复用的 Ping/Pong 与输入/输出同步 | 新出现 MTE2/MTE3 与 VF 的跨块重叠，内部 GM 访问转为 UB；同时核算 DCache、系统保留及 VF 栈，不能直接复用纯 SIMD 的全部可用 UB 数值 |
| 外层频繁调用极短 SIMT VF，启动/调度成本压过实际计算 | [线程块与 VF 调用配置][hybrid-grid]，在容量和依赖允许时合并工作粒度、减少调用 | VF 调用/外层准备次数减少，内部指令与每线程工作增加可能抵消收益；不通过增大外层核数解决所有问题 |
| SIMD 标量串行处理大量分支判断，成为下一阶段生产者 | [SIMT 并行分支判断][hybrid-branch]，仅将合适片段交给 SIMT | 外层 SCALAR 代价减少，但新增 VF 调用、访存和汇合同样计入；保持同一 Warp 的分支分析及程序语义 |

引入 UB 或重分块会改变物理搬运、调用与调度数量。比较相同逻辑输入和精度，分别验证“外层重叠”和“内部执行效率”；外层气泡消失并不证明内部访存或分支已经最优。

## 来源与版本

依据官方 asc-devkit 提交 `fba3e6925684e20cae884aa86300814e9cc64d05` 的 [SIMT 优化概述][overview]和[混合编程概述][hybrid]及下列专题整理。流水映射与验证顺序是本 skill 的分析方法，具体 API、Cache Line、Bank 与资源上限按目标产品文档确认。

[overview]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/overview.md
[hybrid]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_simt_hybrid_optimization/overview.md
[grid]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/execution_config/thread_block_config.md
[register]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/execution_config/thread_count_register_overflow.md
[coalescing]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/memory_access/memory_access_merge.md
[bank]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/memory_access/avoid_ub_bank_conflict_simt.md
[dcache]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/memory_access/dcache_access.md
[soa]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/memory_access/soa_layout_efficiency.md
[short]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/memory_access/short_vector_efficiency.md
[align]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/memory_access/type_alignment_transfer.md
[branch]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/control_flow/avoid_warp_branch.md
[predicate]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/control_flow/branch_handing.md
[division]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_simt_hybrid_optimization/instruction_optimization/fast_integer_division.md
[atomic]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simt_operator_optimization/instruction_optimization/atomic_instruction_optimization.md
[ub]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_simt_hybrid_optimization/memory_access/ub_memory_access.md
[hybrid-grid]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_simt_hybrid_optimization/execution_control/thread_block_config.md
[hybrid-branch]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_simt_hybrid_optimization/compute_optimization/simt_branch_judgment.md
