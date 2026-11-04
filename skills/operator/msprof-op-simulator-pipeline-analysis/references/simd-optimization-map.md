# SIMD 优化技巧如何落到流水图

按源码区分 TPipe/TQue Tensor 编程、静态 Tensor、寄存器 VF 和 AIC/AIV 协作。以下映射是结合官方优化建议形成的诊断方法，表中流水现象是候选入口，不是仅凭一张图即可成立的硬件结论。

## 从现象选择最小实验

| 现象 / 位置 | 定因前的补证 | 官方技巧与适用改动 | 应观察的变化与代价 |
| --- | --- | --- | --- |
| 多数核结束后少数核继续执行，或每核块数差异大 | 完整全核数据、相同核类型、每核有效元素和主尾块分配；核筛选/片段不够 | [核间负载均衡][balance]：调整核间 tiling、主尾块和核数 | 最慢核与其他核的完成差缩小；同步检查单核工作量和新增小块/调度成本，不只看平均值 |
| 第一条有效搬运/计算前 SCALAR 很长，或有效工作结束后仍有尾段 | 源码对应的 tiling 读取、TPipe 初始化、Workspace、Cache 操作；排除漏采 | [头尾开销优化][overhead]：只处理当前路径存在的冗余初始化、过大 TilingData、Kernel 类型与核数配置 | 对应头/尾段及指令次数减少；不将它们计入循环稳态，也不删除语义必需的初始化 |
| MTE2 → 计算 → MTE3 逐块串行，消费者停在等待 | 同步方向、事件 ID、Buffer 地址和复用顺序 | [Double Buffer][double]；MIX 且确实调用 Matmul 高阶 API 时评估[异步 Iterate/IterateAll][iterate]。具体同步口径见[共同参考](pipeline-analysis.md#double-buffer-的适用边界) | 跨块交集增加、内部空泡减少；核算两组 Buffer、workspace 和 API 完成等待，保留真实依赖 |
| MTE 有很多短小搬运、阶段开销大，后续计算反复等供数 | detail 中的长度、stride、地址、调用次数；检查是否因临时 Buffer 挤占而 tile 太小 | [增大连续搬运块][large]、[目标产品的 GM 对齐建议][align]、[选择合适搬运 API][copy]；生命周期不重叠时才[共享临时 Buffer][scratch] | 搬运条数/分段与固定开销减少；有效元素和尾块正确。大块会占用更多 UB，可能减少流水深度；512B 是目标产品的性能建议，不是所有访问的统一合法性要求 |
| 计算间反复出现同一中间结果的 MTE3/MTE2 往返 | 源码数据流确认生产者和消费者，区分必需读写与中间结果落 GM | [UB 融合连续 Vector 计算][fusion]，在同一授权算子内保留可复用数据 | 中间结果的 GM 流量与搬运指令减少，计算段更连续；局部存储/活跃区间可能增加，不要求优化前后物理流量不变 |
| tile 放大后反而更慢，或重复读同一 GM 数据的代价高 | 当前访问复用模式、工作集和 Cache 策略；有计数时核对命中及流量，只有 trace 则保留 Cache 假设 | [L2 CacheMode][cache]、[L2 切分][l2split]：按流式/复用访问分别设计 | 重复搬运代价或实际读流量下降；不能仅凭 MTE 长或短推导命中率，不预设一律旁路或驻留 |
| Vector/RVEC load-store 拖长，重叠越多越慢 | 目标架构 UB bank/group 规则、操作数基址/步长、并发读写；对应冲突证据 | [避免 UB Bank 冲突][bank]：调整地址布局、stride 或 padding | 业务搬运量不变而相应访存/冲突成本减少；padding 增加 UB 占用。2201 与 3510 的规则不可互套，不照搬 SIMT subbank 公式 |
| SCALAR 配置指令夹在大量短计算之间，掩码重复设置 | 当前 Vector API、相同 mask 是否连续复用、主尾块差异和源码/指令次数 | [Counter 模式][counter]、[全局 mask 复用][mask]：仅对支持的 API/产品使用，边界后恢复需要的状态 | 配置/主尾块控制指令减少、发射间隔缩短；不把 A2/A3 全局 mask 方案直接套到 3510 C API 的寄存器 VF |
| 没有大空泡，但归约或其他计算链本身占主要时间 | 对应指令延迟与调用次数、数据依赖、shape；寄存器 VF 还需看内部 RVEC 而非只看 VF 包络 | [低延迟归约指令组合][reduce]，减少源码中可证明的冗余计算/寻址 | 关键依赖链及指令代价下降，不必期待 MTE 重叠变化；归约顺序、舍入和特殊值结果需按原精度标准验证 |
| CUBE/MTE1 或 FIXP/Vector 的反复交换拖延矩阵迭代 | A/B/C 实际数据流、复用范围、L1/L0 容量、矩阵 API 与同步 | [小矩阵驻留 L1][l1]、[L0C 保留矩阵累加][l0c]；已有 Bias/量化操作可评估[BT Buffer][bt]/[FP Buffer][fp]路径 | 重复搬入或 FIXP→GM→UB 往返减少；检查累加初始化、布局与结果可见时机，不仅看 CUBE 覆盖率 |

选择影响当前关键依赖链的少量候选。优化总览中的 dtype、量化或 AtomicAdd 方案只有在现有算子语义与精度契约允许时才适用，不能因为流水忙碌就改数值类型或引入原子写。

对于尾部 DCCI 很长的 Ascend 950 场景，[DCI 编译选项][dci]还要求确认 GM 写入的 Cache 一致性：必须证明不依赖默认 Clean，或已有正确的显式回刷。只有图上存在尾部空白时不建议该开关。

## 配套案例的核对项

msopprof 仓库中 `example/sample_simd_vf_double_buffer` 案例的 `fused_compute_vf` 是 `__simd_vf__`，不是 SIMT；工程位置见 [Skill 案例入口](../SKILL.md#配套实测工程)。

- 从采集的流水中选取完整数据块，检查 MTE2、VF、MTE3 的执行顺序与交集，记录空泡期间其他流水正在执行的指令。
- 对照 UB 地址与 GET_BUF/RLS_BUF 的 PIPE、bufId、计数和顺序，追踪阻挡下一块搬入或计算的生产者；结合源码判断输入与输出的复用约束。
- 分别核对各版本的搬运量、算术工作量、忙碌阶段与头尾成本；短循环、单核数据只能支持其覆盖范围内的判断。
- 只有 RVEC 泳道不能证明寄存器 spill 或 Bank 冲突；没有额外证据时保留候选。也不适用 SIMT warp 发散或 `__launch_bounds__` 建议。

## 来源与版本

用户指定的[官方 SIMD 优化总览][summary]是分类入口。在线页面本次未提供正文，细节依据官方 asc-devkit 提交 `fba3e6925684e20cae884aa86300814e9cc64d05` 的同路径文档核对，下面固定链接便于追溯。实际执行仍按目标 CANN、编译器与芯片版本检查能力，例子中的性能数字不作为当前算子的收益。

[summary]: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/optimization_summary.md
[balance]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/tiling_strategy/inter_core_load_balance.md
[overhead]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/overhead_optimization/overhead_optimization.md
[double]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md
[iterate]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_iterate_avoid_sync.md
[large]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/transfer_larger_data_blocks.md
[align]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/gm_address_512b_alignment.md
[copy]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/efficient_transfer_api.md
[scratch]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/shared_temp_buffer.md
[fusion]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/vector_compute/ub_fusion_vector_compute.md
[cache]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/l2_cache_mode_config.md
[l2split]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/l2_cache_split.md
[bank]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/memory_access/avoid_ub_bank_conflict/overview.md
[counter]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/vector_compute/vector_counter_mode.md
[mask]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/vector_compute/mask_reuse_optimization.md
[reduce]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/vector_compute/low_latency_instruction.md
[l1]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/matrix_compute/small_matrix_l1_buffer.md
[l0c]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/matrix_compute/l0c_buffer_matmul_accumulate.md
[bt]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/matrix_compute/bt_buffer_bias_compute.md
[fp]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/matrix_compute/fp_buffer_quantization.md
[dci]: https://gitcode.com/cann/asc-devkit/blob/fba3e6925684e20cae884aa86300814e9cc64d05/docs/zh/guide/operator_practice/simd_operator_optimization/overhead_optimization/dci_compile_option.md
