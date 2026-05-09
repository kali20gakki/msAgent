---
name: ascendc-operator-performance-optim
description: 基于 msprof op 工具的端到端 Ascend NPU 算子性能调优技能。当用户提供算子源码目录，并要求使用 msprof op 完成从性能采集、瓶颈分析、代码优化到优化效果验证的完整闭环时，必须使用此 skill。典型触发词包括"端到端调优"、"代码优化"、"msprof op 调优"、"完整调优"、"性能优化闭环"、"端到端"、"调优全流程"。此 skill 自动执行：编译运行 → msprof op 上板/仿真性能采集 → 瓶颈诊断 → 代码优化 → 重新采集 → 优化前后性能对比。注意：此 skill 专门使用 msprof op 工具进行性能采集和分析，覆盖 msot-msopprof-operator-profiler（纯分析），并额外包含编译运行和性能采集的自动化流程。
---

# 基于 msprof op 的端到端算子性能调优技能

对 Ascend NPU 上的单算子，基于 **msprof op** 性能采集工具，skill 不仅排查性能问题，还负责 **修改代码并验证优化效果**，输出优化前后的性能对比报告，完整流程为：

```
Phase 1: 排查 — 审查代码 + 学习设计文档，发现优化点
Phase 2: 基线 — 保存当前性能测试结果
Phase 3: 优化 — 学习知识后修改算子代码，修改备份的算子工程目录
Phase 4: 精度 — 精度验证（确保优化后功能正确）
Phase 5: 性能 — 优化前后性能对比
```

## 适用场景

- 用户提供算子源码目录，包含可编译运行环境
- 用户要求使用 **msprof op** 进行端到端性能调优
- 用户要求"端到端调优"、"完整调优流程"、"性能优化闭环"
- 用户希望自动完成编译、运行、msprof op 性能采集、分析、优化、对比的全流程
- 用户已有算子代码，需要系统性提升算子性能

## 触发词（召回增强）

当用户问题包含以下词或近义表达时，优先触发本技能：

- `ascendc-operator-performance-optim` 
- `端到端优化` / `端到端调优` / `算子代码优化` / `msopprof 端到端调优`


当用户问题包含以下词或近义表达时，优先触发 **`msot-msopprof-operator-profiler`**，不使用 **`ascendc-operator-performance-optim`** ：

- 明确提到 `msprof op`、`msprof op simulator`、`msOpProf`
- 询问如何做 **算子性能调优 / 上板调优 / 仿真调优**
- 询问如何查看或解释：
  - `visualize_data.bin`
  - `trace.json`
  - `OpBasicInfo.csv`
  - `PipeUtilization.csv`
  - `MemoryDetail`
  - `Roofline`
  - `PMSampling`
- 询问如何在 **device vs simulator** 之间选择
- 询问 `application` / `config` / `export` 三种输入形态的差别
- 询问如何生成 **固定报告 / 模板化结论 / Top 5 总结**
- 遇到以下高频故障：
  - `signal 6`
  - `Bad address`
  - `--soc-version` 不生效
  - `--kernel-name` 不生效
  - `--export` 目录如何组织
  - `TimelineDetail` / `PMSampling` / `--core-id` 为什么行为不符合预期

## 不要在这些场景误用本技能

- 用户只是想要输出性能报告，不需要做代码优化
- 用户只想分析性能数据，问一些解释咨询类的
- 用户给出性能数据目录，只要分析场景


---

## Phase 1: 排查 — 发现优化点

### 1.1 学习算子设计文档

**MANDATORY — 排查前必须先理解算子设计**：

1. 读取 用户提供的算子设计文件（若存在），提取：
   - 算子类型（elementwise / 行处理 / Cube）
   - Tiling 策略（核间切分 / 核内切分）
   - UB 空间分配方案
   - 计算逻辑与数据流
2. 读取 `op_host/<op_name>.cpp` 和 `op_kernel/<op_name>.cpp` 全部源码（若存在）
3. 读取用户提供的核函数文件(cpp文件)

### 1.2 逐阶段排查

按以下顺序逐阶段审查算子代码。对每个阶段，加载对应的 reference 文件，逐项
对照代码检查。

```
- [ ] 1. Tiling    — 数据在多核与 L2Cache 间的切分策略
- [ ] 2. 搬运      — DataCopy 的带宽利用率
- [ ] 3. API 使用  — Ascend C API 的高效用法
- [ ] 4. 内存      — 数据在存储层级中的放置策略
- [ ] 5. 流水      — CopyIn / Compute / CopyOut 的重叠执行
- [ ] 6. Scalar    — Scalar标量计算
```

每个阶段有独立的 reference 文件，排查时**仅加载当前阶段**的文件：

- 阶段 1：[references/tiling-prof.md](references/tiling-prof.md)
- 阶段 2：[references/data-copy-prof.md](references/data-copy-prof.md)
- 阶段 3：[references/api-usage-prof.md](references/api-usage-prof.md)
- 阶段 4：[references/memory-prof.md](references/memory-prof.md)
- 阶段 5：[references/pipeline-prof.md](references/pipeline-prof.md)
- 阶段 6：[references/scalar-prof.md](references/scalar-prof.md)

#### 1. Tiling

> 详细示例：[references/tiling-prof.md](references/tiling-prof.md)

排查项：

- [ ] **1.1 多核切分**：`blockDim` 是否设为硬件核数？
  - 耦合架构：`GetCoreNumAiv()` 或 `GetCoreNumAic()`
  - 分离架构 Vector 算子：AIV 核数（如 40）
  - 分离架构 Cube 算子：AIC 核数（如 20）
  - 分离架构 MIX 算子：物理核组数（如 20 = 40 AIV / 2），不可超过物理核数
- [ ] **1.2 L2Cache 切分**：当 `输入 + 输出 > L2Cache 容量` 时，是否将数据
  按 L2Cache 大小分块，所有核协同处理同一块后再切换下一块？
- [ ] **1.3 核间负载均衡**：L2Cache 切分后，尾块是否在各 pass 间交替分配，
  避免固定某些核始终拖尾？

#### 2. 搬运

> 详细示例：[references/data-copy-prof.md](references/data-copy-prof.md)

排查项：

- [ ] **2.1 单次搬运量 >= 16 KB**：每次 `DataCopy` 是否搬运至少 16 KB？
  小于此值带宽利用率显著下降。
- [ ] **2.2 GM 地址 512B 对齐**：GM 起始地址是否 512 字节对齐？
  （Atlas A2 系列上，32B 对齐比 512B 对齐带宽最多低 30%。）
- [ ] **2.3 stride 参数代替 for 循环**：间隔搬运是否使用 `DataCopyParams`
  （blockCount/blockLen/srcStride/dstStride）一次下发，而非用 for 循环逐行搬运？

#### 3. API 使用

> 详细示例：[references/api-usage-prof.md](references/api-usage-prof.md)

排查项：

- [ ] **3.1 TPipe 在 kernel 类外创建**：`TPipe` 是否在 kernel 入口函数中创建
  并以指针传入类？（类内 TPipe 会阻止 Scalar 常量折叠，增加约 17% scalar_time。）
- [ ] **3.2 纯搬运算子使用 TQueBind**：无 Vector 计算的算子是否用
  `TQueBind<VECIN, VECOUT>` 替代了分离的 `TQue<VECIN>` + `TQue<VECOUT>`？
  （消除冗余的 LocalTensor 间 DataCopy，`aiv_vec_time` 降至约 0。）
- [ ] **3.3 Counter 模式（SetMaskCount）**：Vector 指令是否使用 Counter 模式，
  而非 Normal 模式手动计算主块/尾块 mask？
- [ ] **3.4 Matmul AtomicAdd**：Matmul 结果 C 需要与 GM 矩阵 D 相加时，
  是否在 `IterateAll`/`GetTensorC` 中设置 `enAtomic=1` 融合累加？
  （可减少约 12% cycle。）
- [ ] **3.5 归约指令组合**：连续 buffer 归约到标量时，是否使用
  `BlockReduceSum` + `WholeReduceSum` 组合，而非多次相同归约指令？

#### 4. 内存

> 详细示例：[references/memory-prof.md](references/memory-prof.md)

排查项：

- [ ] **4.1 UB Buffer 融合**：连续 Vector 运算（如 Exp → Abs）的中间结果
  是否留在 UB 内，而非经 GM 往返？
- [ ] **4.2 L0C 累加矩阵乘**：`A1*B1 + A2*B2 + ...` 场景下，Mmad 结果是否
  在 CO1（L0C）中原地累加，而非逐次写 GM 再在 UB 求和？
- [ ] **4.3 小矩阵长驻 L1**：当 L1 无法同时容纳左右矩阵时，较小矩阵是否
  一次加载后常驻 L1，仅循环搬运较大矩阵？
- [ ] **4.4 BT Buffer 存放 bias**（分离架构）：bias 是否存入 BT Buffer（C2）
  并通过 `Mmad` 一步融合，而非在 UB 中单独做 Add？
- [ ] **4.5 FP Buffer 存放量化参数**（分离架构）：量化参数是否存入
  FP Buffer（C2PIPE2GM）并通过 `Fixpipe` 随路量化，而非在 UB 中单独计算？

#### 5. 流水

> 详细示例：[references/pipeline-prof.md](references/pipeline-prof.md)

排查项：

- [ ] **5.1 CopyIn/Compute/CopyOut 范式**：算子是否划分为三级流水，
  使用 `TQue` 进行级间同步？
- [ ] **5.2 Double Buffer**：`InitBuffer` 的 buffer 个数是否设为 2，
  使 CopyIn/CopyOut 与 Compute 重叠执行？
  （前提：循环次数 >= 2，且搬运时间相对计算时间不可忽略。）
- [ ] **5.3 异步 Iterate（MIX 模式）**：Matmul MIX 场景下，是否使用
  `Iterate<false>()`/`IterateAll<false>()` 避免每次迭代的 AIC/AIV 同步开销？

#### 6. 标量  
> 详细示例：[references/scalar-prof.md](references/scalar-prof.md)
排查项：
- [ ]**6.1 one-hot**：使用 `for` 循环逐元素操作？


### 1.3 输出排查报告

排查完所有阶段后，按以下格式输出汇总：

```
## 优化排查报告

### 发现的问题（按预期收益排序）
1. [阶段 X.Y] <问题描述> — <预期收益>
2. [阶段 X.Y] <问题描述> — <预期收益>
...

### 已确认无问题
- [阶段 X.Y] <检查项描述>
...

### 优化计划
按预期收益从大到小排列，确定本轮优化的目标项。
```

---

## Phase 2: 基线 — 保存当前性能测试结果

优化前必须保存性能基线，以便优化后精确对比。

### 2.1 性能评估分析报告
**MUST** 调用 **`msot-msopprof-operator-profiler`** skill 完成完整性能评估
1. 读取 `msot-msopprof-operator-profiler` SKILL.md
2. 按照其流程进行性能优化前评估
3. 将当前性能报告备份为基线文件，命名为 `<op_name>_baseline_report.md`
---

## Phase 3: 优化 — 学习知识后修改代码

### 3.1 学习算子开发知识（MANDATORY）

**修改代码前 MUST 加载 reference/ascendc-api下文件**，
确保对 AscendC API、数据搬运、同步控制等有准确理解。

按需加载以下 reference（位于 `ascendc-operator-performance-optim/references/ascendc-api`）：

| Reference 文件 | 用途 |
|---------------|------|
| `GUIDE.md` | 总览：模板选择、代码生成流程 |
| `data-copy-api.md` | DataCopy/DataCopyPad API 详解 |
| `vector-compute-api.md` | Vector 计算 API 详解 |
| `sync-control-api.md` | TQue/Pipe 同步控制 |
| `resource-management-api.md` | TPipe/TBuf 资源管理 |
| `basic-data-structures-api.md` | LocalTensor/GlobalTensor 等基础结构 |
| `kernel-constraints.md` | Kernel 编程约束与常见陷阱 |

根据 Phase 1 发现的优化点，选择性加载相关 reference。例如：
- 优化搬运 → 加载 `data-copy-api.md`
- 优化流水 → 加载 `sync-control-api.md` + `resource-management-api.md`
- 优化计算 → 加载 `vector-compute-api.md`

### 3.2 制定修改方案

针对 Phase 1 排查报告中的每个优化点，制定具体的代码修改方案

### 3.3 执行代码修改

1. 备份原始代码：说明备份代码名称和位置
2. 实施代码修改（使用 `edit_file` 工具）
3. 重新编译运行验证：
   ```bash
   bash {operator_dir}/compile_run.sh {batch_size} {num_class}
   ```
4. 确认输出包含 **"pass"** 或者无 **"error"** 信息，验证优化后功能正确性

按照修改方案逐一修改代码。修改时遵守以下规则：

**MUST 遵守 ascendc-api 反模式清单**：
- **NEVER** 在原工程目录修改代码，必须备份原工程目录，在新文件夹下修改验证
- **NEVER** 让 FP16/BF16 直接参与复杂数学计算，必须先 Cast 到 FP32
- **NEVER** 在 EXEC_KERNEL_CMD 中传右值
- **NEVER** 对 GM↔UB 搬运使用 DataCopy，必须用 DataCopyPad
- **NEVER** 在 ReduceSum/ReduceMax 后直接复用源 tensor
- **NEVER** 在 kernel 中使用 `std::min/max/abs/sqrt/exp` 等标准库函数
- **NEVER** 向高维切分 API 传入 repeatTime > 255
- **NEVER** 修改 `cmake/` 或 `csrc/utils/` 下的文件
- **NEVER** 硬编码核数或 UB 大小


编译失败时进入排错循环（最多 3 次）。

---

## Phase 4: 精度验证 — 确保优化后功能正确

**MANDATORY — 优化后必须先通过精度验证再进行性能对比。**

1. 备份原始代码：说明备份代码名称和位置
2. 实施代码修改（使用 `edit_file` 工具）
3. 重新编译运行验证：
   ```bash
   bash {operator_dir}/compile_run.sh {batch_size} {num_class}
   ```
4. 确认输出包含 **"pass"** 或者无 **"error"** 信息，验证优化后功能正确性


## Phase 5: 性能验证 — 确认优化效果

### 5.1 运行同 case 性能测试
调用 **`msot-msopprof-operator-profiler`** skill 重新执行性能评估。

关键要求：
- **MUST** 使用与基线完全相同的用例shape
- **MUST** 生成新的 `<op_name>_msopprof_report.md`
- **MUST** 在当前对话中展示对比表、汇总与结论

### 阶段五：优化效果验证

#### 步骤 5.1：重新采集上板性能数据
1. 清理旧的 msprof 输出：`rm -rf {operator_dir}/OPPROF_*`
2. 执行上板采集命令：
   ```bash
   cd {operator_dir} && msprof op {算子执行命令}
   ```
   或使用辅助脚本：
   ```bash
   bash {skill_dir}/scripts/e2e_profile_onboard.sh {operator_dir} {batch_size} {num_class}
   ```
3. 找到新生成的 `OPPROF_*` 目录
4. **记录优化后的性能数据目录路径**

#### 步骤 5.2：对比分析

使用辅助脚本生成对比报告：
```bash
bash {skill_dir}/scripts/e2e_compare.sh {优化前_OPPROF_目录} {优化后_OPPROF_目录}
```

##### 5.2.1 总耗时对比
- 优化前总耗时：从优化前的 `OpBasicInfo.csv` 读取 `Task Duration(us)`
- 优化后总耗时：从优化后的 `OpBasicInfo.csv` 读取 `Task Duration(us)`
- 计算加速比：`(优化前耗时 - 优化后耗时) / 优化前耗时 * 100%`

##### 5.2.2 关键指标对比表

| 指标 | 优化前 | 优化后 | 改善幅度 |
|------|--------|--------|---------|
| 总耗时 (us) | | | |
| aiv_vec_ratio | | | |
| aiv_scalar_ratio | | | |
| L2 Cache 命中率 | | | |
| UB 读写带宽 | | | |

##### 5.2.3 流水线利用率对比
- 优化前 vs 优化后的 `PipeUtilization.csv`
- 检查 MTE1/MTE2/Cube/Vector 利用率变化

### 阶段六：输出优化报告

#### 步骤 6.1：生成优化报告

参考按以下格式输出完整优化报告：

```
================================================================
              基于 msprof op 的算子性能优化报告
================================================================

【算子信息】
- 算子名称：CrossEntropy
- 输入参数：batch_size={batch_size}, num_class={num_class}, blockDim=40
- 源码文件：{operator_dir}/cross_entropy.cpp
- 采集工具：msprof op（上板）/ msprof op simulator（仿真）

【优化前性能】
- 总耗时：{value} us
- aiv_vec_ratio：{value}%
- aiv_scalar_ratio：{value}%
- L2 Cache 命中率：{value}%
- 主要瓶颈：
  1. {瓶颈描述}
  2. {瓶颈描述}

【优化内容】
1. {优化描述}
   - 修改前：{代码片段}
   - 修改后：{代码片段}
   - 优化原理：{原理说明}

2. {优化描述}
   ...

【优化后性能】
- 总耗时：{value} us（加速比 {value}%）
- aiv_vec_ratio：{value}%（提升 {value}%）
- aiv_scalar_ratio：{value}%（降低 {value}%）
- L2 Cache 命中率：{value}%（提升 {value}%）

【优化效果总结】
{总结性描述}

================================================================
```

#### 步骤 6.2：保存优化后代码
1. 确保优化后的代码已保存到核函数文件
2. 备份文件夹目录必须告知用户

## 6. 执行规范

### 6.1 安全操作规范
- 修改源码前必须备份（`.bak`）
- 每次修改后必须验证功能正确性（检查 "pass"）
- 性能采集前后必须清理旧的 `OPPROF_*` 目录

### 6.2 数据管理规范
- 优化前的性能数据目录：记录路径，用于对比
- 优化后的性能数据目录：记录路径，用于对比
- 每次优化迭代保存一份性能数据快照

### 6.3 错误处理
| 错误类型           | 处理方法 |
|----------------|----------|
| 编译失败           | 检查编译错误输出，修复后重试 |
| 运行失败           | 检查 ACL 初始化和内存分配 |
| 精度不通过          | 检查算子逻辑是否正确，必要时回退到备份版本 |
| msprof op 采集失败 | 检查环境变量和权限 |
| OPPROF_* 目录未生成 | 确认 msprof op 命令正确执行，检查输出日志 |

### 6.4 msprof op 采集说明
- **上板采集**需要算子已在 NPU 上编译运行通过
- **仿真采集**需要链接 simulator 库（compile_simulator.sh 自动处理）
- 仿真采集的 `--soc-version` 需根据实际硬件型号设置（当前为 Ascend910B1）

## 7. 常见瓶颈模式

### 模式 1：Scalar 指令占比过高
**症状**：`aiv_scalar_ratio > 50%`
**根因**：大量标量操作（`GetValue`、`SetValue` 循环、`for` 循环）
**优化**：用向量化 API（`Duplicate`、`Exp`、`Log`、`Mul`）替代

### 模式 2：冗余同步
**症状**：流水线频繁停顿，`PipeBarrier` 和 `SetFlag`/`WaitFlag` 过多
**根因**：不必要的同步屏障
**优化**：合并连续向量操作，减少中间同步

### 模式 3：重复计算
**症状**：相同计算逻辑出现多次
**根因**：代码中存在冗余逻辑
**优化**：删除重复代码，只保留一份

### 模式 4：单缓冲
**症状**：搬运和计算无法 overlap
**根因**：`BUFFER_NUM = 1`
**优化**：改为 `BUFFER_NUM = 2`，实现 double buffer

### 模式 5：L2 Cache 命中率低
**症状**：`ai*_total_hit_rate(%) < 80%`
**根因**：数据复用性差，数据访问模式不连续
**优化**：优化数据排布和访问模式

### 模式 6：流水线负载不均
**症状**：某条流水线（MTE1/MTE2/Cube/Vector）占比异常高或低
**根因**：计算与搬运不匹配，或计算单元间负载不均
**优化**：调整 double buffer 策略，优化计算分工


## 检查清单（助手自检）

### Phase 1: 排查
- [ ] 已读取完整源码
- [ ] 已逐阶段加载 reference 并逐项排查
- [ ] 已输出排查报告，优化点按预期收益排序

### Phase 2: 基线
- [ ] 已保存基线快照（`_baseline_report.md`）

### Phase 3: 优化
- [ ] 已加载 ascendc-api reference（修改前必读）
- [ ] 代码修改遵守反模式清单
- [ ] 编译安装成功

### Phase 4: 精度
- [ ] 精度验证通过

### Phase 5: 性能
- [ ] 已在对话中展示性能对比数据
- [ ] 已判定是否提升

### 输出
- [ ] **已在当前对话中展示**排查总结、性能对比、≥3 条结论
- [ ] **NEVER** 仅输出文件路径



## 核心参考资源

### 算子源码
- **算子目录**：由用户提供

### 编译运行脚本
- **上板编译运行**：由用户提供
- **仿真编译**：由用户提供
- **算子执行命令**：由用户提供，或基于用户提供的readme文档提取

### 性能采集命令

#### 上板采集
```bash
cd {operator_dir} && rm -rf OPPROF_* && msprof op {算子执行命令}
```

#### 仿真采集
```bash
cd {operator_dir} && rm -rf OPPROF_* && msprof op simulator --soc-version=Ascend910B1 {算子执行命令}
```

### 算子执行参数
- 算子执行命令：用户提供

###  前置技能
- **msot-msopprof-operator-profiler**：用于对采集到的 msprof op 性能数据进行深度分析
- **ascendc-operator-performance-optim**：用于算子性能优化

### 辅助脚本
本 skill 提供以下辅助脚本（位于 `{skill_dir}/scripts/`）：
- `e2e_compile_run.sh` - 编译并运行算子（基准版本）
- `e2e_profile_onboard.sh` - 编译 + 上板运行 + msprof op 性能采集
- `e2e_profile_simulator.sh` - 编译 + 仿真运行 + msprof op simulator 性能采集
- `e2e_compare.sh` - 对比优化前后的 msprof op 性能数据