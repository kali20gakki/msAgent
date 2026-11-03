# 流水统计、耗时分类与证据边界

## 先分清要优化什么

按“在哪里花时间 → 影响哪个消费者/最后完成者 → 还缺什么证据”建立假设，再按 [SIMD 映射](simd-optimization-map.md) 或 [SIMT 映射](simt-optimization-map.md) 选技巧。

| 耗时形态 | 流水图的作用 | 下一步 |
| --- | --- | --- |
| 阶段之间空泡明显 | 找生产者、消费者、同步与 Buffer 复用顺序 | 检查预取、双缓冲、异步与同步粒度 |
| 阶段持续忙但代价高 | 定位执行范围，关联指令/代码热点 | 检查搬运粒度、重复流量、指令效率、SIMT 地址/线程映射 |
| 开始、结束或各核完成时间差大 | 分离启动、排空、尾块及核间工作量 | 检查 tiling、核数、调度与头尾代码；片段或单核数据不能证明全局尾核 |

对忙碌阶段，不将长指令范围自动归因于某一种硬件瓶颈。覆盖率高只说明区间覆盖多，不证明运算有效、带宽已饱和或访存已合并。优化非关键阶段后若等待增加，需要检查是否只是瓶颈转移。

## 事件和时间口径

- 先分离算子调用、核和子核，再识别实际泳道；常见为 MTE1/2/3、VECTOR、CUBE、FIXP、SCALAR，以及新架构的 PUSHQ、RVECLD、RVECEX、RVECST。
- Chrome trace 的 `X` 事件表示 `[ts, ts + dur)`；`B/E` 按同一线程的嵌套顺序配对，同步事件还应核对 ID。缺失端点的事件不补造结束时间。
- `M` 是元信息，`s/t/f` 是依赖连线，计数、标记、Warp 生命周期和父级范围不计入流水有效执行区间。
- 从本次读取的工具字段说明或生成实现确认时间单位，不能仅凭 `displayTimeUnit` 或数值大小推断 `ts/dur` 的存储单位。
- SET/WAIT 或 mutex 相关区间归入同步；在部分产品上显示为 `set_event/wait_event` 或 `SET_INTRA_BLOCK/WAIT_INTRA_BLOCK`。VF 父级范围与内部 RVEC 指令分别统计，不能把两者相加。
- C API 的 `asc_lock/asc_unlock` 可能在原始 `*.instr_log.dump` 中表现为 GET_BUF/RLS_BUF（含 PIPE、bufId、cnt），而未出现在 trace 中。使用这些记录验证同步方向和重复顺序；只有完成 tick 或顺序记录时，不能反推出完整等待区间。RV_SEND 等内部同步指令也不计作算术执行。
- SIMD/SIMT VF、上板 Warp 生命周期和采样 Pipe 区间是不同层级。Warp 活跃数量不是执行 lane 有效率，也不直接等于 occupancy；上板 Warp 数据可能来自独立重放，不能与仿真 VF 对齐后推算精确 stall。

## 补证的边界

- 源码/反汇编用于确认数据依赖、线程地址、分支、实际指令形式和栈访问；仿真热点 CSV 负责定位，不负责直接证明带宽饱和或 Cache 命中率。
- spill 结合编译器资源报告与栈 load/store 验证；Bank 冲突结合目标架构的地址映射及相应证据；分支发散结合 lane 的分支条件及可用控制流/掩码信息。缺项就保留为候选原因。
- 需要性能计数时，可读取用户已有的带宽、Cache、冲突、PCSampling 或全核数据作为补证；核对其模式、芯片、范围和版本。不能把通用冲突字段直接当成 SIMT subbank 冲突，也不能把样本占比当成精确等待时间。
- 补证数据的获取方式、平台与模式适用范围从 msopprof 仓库当次读取；先确认数据层级，不能把 VF 外层范围当作内部指令明细。

## 空泡与重叠

为每个核选定观察窗口 W，并说明它对应哪些完整数据块。

1. 将一条流水的计算或搬运区间裁剪到 W、按开始时间排序；合并重叠或相邻区间，得到互不相交的忙碌区间。
2. 对相邻合并区间计算 `gap = next.start - current.end`。这才是可观察的内部空泡；直接比较原始相邻事件可能在区间嵌套时造出假空泡。
3. 报告总忙碌时间、内部 gap 总量、最大 gap 和代表性区间；覆盖率为忙碌并集长度除以 W 长度。首尾未覆盖段单列，不自动归为可消除空泡。
4. 对 MTE2 与计算、计算与 MTE3 等做区间交集，报告交集长度；若给比例，明确分母是 W 还是某条流水的忙碌时间。重叠流水占比不能直接相加。
5. 同步等待另做区间并集。等待可能与其他执行区间重叠，等待总量与空泡总量不是可直接相加的互斥分解。

窗口长度或执行速度不同的对照，优先比较相同块序号的稳态阶段，同时给绝对时长和各自分母。只有七个块的示例不足以代表长时间稳态，也不宜对少量 gap 强调高分位数。

## 从现象到可验证原因

| 观察现象 | 需要补齐的证据 | 建议方向 |
| --- | --- | --- |
| MTE2 → V → MTE3 逐块串行 | 下一块 MTE2 等前一块写回；单组 Buffer 或同一同步链 | 检查 Ping/Pong、输入输出同步是否可独立；不是删除必需的数据依赖 |
| 计算间有长空泡，期间 MTE2 持续搬入 | 消费者等待数据就绪、下一块预取位置、搬运长度与地址 | 评估预取或分块；确认瓶颈是否已是搬运通路 |
| 下一块计算等 MTE3，MTE2 也被拖住 | 输出 Buffer 未释放、输入输出被同一 mutex 串联 | 独立管理输出生命周期，评估双缓冲容量 |
| 已有两组 Buffer 仍无跨块重叠 | 每轮全流水 barrier、过早 wait、相同地址或事件复用 | 仅放宽已证实可解耦的依赖，保留正确的复用顺序 |
| 所有主要流水都空闲，SCALAR 或控制流活跃 | 标量地址计算、循环、指令发射/缓存相关证据 | 将其列为发射开销假设，不能仅凭空白认定带宽不足 |
| AIC/AIV 交替等待 | 核间信号、workspace 槽、调用的具体 Matmul API | 检查 MIX 流水编排；符合接口语义时才考虑异步 Iterate/IterateAll |
| 新架构 RVEC 发射呈 burst | VF 发射、寄存器或 dispatch 阻塞日志，以及内部 RVEC 时序 | 区分发射/资源压力和缺数据；不只根据父级 VECTOR 范围判断 |
| 仅首尾或尾块出现空白 | 完整运行、块边界、各核有效工作量 | 启动/排空、短循环或负载不均；不要全部归为稳态同步问题 |

根因至少要连接“空泡区间 → 阻塞条件 → 生产者或资源 → 源码”。只有时序相关性时标为假设。找不到源码映射时引用 PC、指令和 detail，不制造行号。

## Double Buffer 的适用边界

同一块的搬入、计算和搬出有真实依赖。Double Buffer 为相邻块提供不同存储，使 `MTE2(n+1)`、`Compute(n)`、`MTE3(n-1)`有机会重叠，不会取消同一块的数据就绪要求。

- 至少有两个数据块，且 UB 能容纳两组输入/输出、临时数据、VF 栈及接口保留空间。
- TPipe/TQue 的 `InitBuffer(queue, 2, bytes)` 管理队列内存和同步；按真实 tiling 重算容量、块数和尾块，不机械地把所有循环次数翻倍或减半。
- C API 指针编程需显式布置 Ping/Pong 并正确使用 `asc_lock/asc_unlock`。配套样例为每组输入和输出分别使用 mutex，不能只把 `InitBuffer` 示例套到该工程。
- 对齐、尾块、写后读和读后复用的依赖保持正确；额外 Buffer 可能迫使 tile 变小，增加指令与同步成本。
- 已经充分重叠、单次计算完成全部数据、计算或搬运远占主导时，收益可能有限。仿真观察到重叠不等于证明上板加速。

MIX 场景下，异步 `Iterate<false>` / `IterateAll<false>` 的 workspace、结果获取、完成等待和生命周期需按目标版本 API 检查。不要建议用户只改模板参数而省略必要同步；基础 Mmad 或 C API VF 代码不直接套用该建议。

## 资料来源

优先查与实际安装版本匹配的官方资料；在线页面无法读取时可使用对应官方源码仓中的正文，说明来源版本。

- [流水编排：使能 Double Buffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md)
- [异步 Iterate/IterateAll 避免 AIC/AIV 同步依赖](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0034.html)
- [msOpProf 仿真用户指南](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_user_guide.md)
- [msOpProf 仿真产物与字段](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_simulator_performance_data.md)
- [msOpProf 上板能力与 Warp 数据边界](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md)
- [asc-devkit 官方流水编排正文](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/guide/operator_practice/simd_operator_optimization/pipeline_scheduling/enable_double_buffer.md)
