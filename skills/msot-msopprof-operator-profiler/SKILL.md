---
name: msot-msopprof-operator-profiler
description: 当用户希望使用 msOpProf（`msprof op` / `msprof op simulator`）对昇腾 AI 算子做上板或仿真性能调优、解释 `aic-metrics`/`trace.json`/`visualize_data.bin`、选择 device vs simulator 路径、排查 `--soc-version`/`--export`/`signal 6`/`Bad address`/热点图或流水图相关问题，或要求生成固定分析报告模板（算子基本信息 / 关键数据 TOP5 / 核心瓶颈 TOP5 / 优化建议 TOP5）时，使用本技能。它负责先判定模式、输入形态与芯片/能力边界，再给出正确命令、结果解读、固定报告输出与高频踩坑规避；不要把经验案例当成通用规则。
---

# msOpProf 算子性能调优

## 何时必须使用本技能

当用户问题包含以下任一类需求时，应优先使用本技能：

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

- 用户只是在问通用 CANN 安装，而不是 msOpProf 本身
- 用户只想修一个普通编译错误，且问题与 msOpProf 调优流程无关
- 用户只是要翻译某段文档，而不是要执行或理解调优流程
- 用户要分析的是其他 profiling 工具（如纯 DB 分析、整机 profiler、非 msOpProf 产物）

## 技能目标

帮助算子开发者在 **上板（device）** 或 **仿真（simulator）** 模式下：

1. 选对运行模式和输入形态（`application` / `config` / `export`）。
2. 生成最小可用且可解释的性能数据。
3. 根据产物类型（CSV / `visualize_data.bin` / `trace.json`）选择正确查看方式。
4. 避免被模式差异、芯片限制、参数互斥和目录要求误导。

## 执行协议

每次使用本技能时，按下面顺序工作，不要跳步：

1. **先识别用户当前模式**
   - 用户到底在问 device、simulator，还是还没决定？
2. **再识别输入形态**
   - `application` / `config` / `export`
3. **再识别目标**
   - 采集数据、解释产物、选参数、排故、看热点图还是看流水图
4. **只加载必要 reference**
   - 若用户重点是上板路径，优先读 `references/device-tuning-guide.md`
   - 若用户重点是仿真路径或 dump/trace 解析，优先读 `references/simulator-tuning-guide.md`
   - 若用户给了 `signal 6` / `Bad address` 等仿真拉起错误，再读 `experiences/simulator-needs-sim-build.md`
5. **输出时必须显式带条件**
   - 说明“这条建议适用于 device 还是 simulator”
   - 说明“这条参数是否只对 application / config / export 生效”

## 使用本技能时的硬规则

1. **先分模式，再谈参数。**
   - `msprof op ...` = 上板调优。
   - `msprof op simulator ...` = 仿真调优。
2. **先分输入形态，再谈命令。**
   - `application`：拉起可执行文件。
   - `config`：基于 JSON 配置和 `.o` 文件。
   - `export`：仅在 simulator 模式下，直接解析已有 dump。
3. **不要把条件化事实说成统一结论。**
   - 很多参数只在特定模式、芯片或输入形态下生效。
4. **区分官方事实与经验。**
   - 本技能的主说明以当前仓内 `msopprof` 参考代码和 user guide 为准。
   - `experiences/` 下的内容视为经验案例，不自动提升为通用规则。
5. **遇到文档口径差异时必须明说。**
   - 例如仓内不同章节对通算流水图支持范围有不同表述；若用户追问精确支持范围，优先以当前安装版本帮助信息和对应专章为准。

## 禁止事项

- 不要把 simulator 的 `trace.json` 和 device 的 `trace.json` 混成同一种语义。
- 不要把 `TimelineDetail` 说成 simulator 参数。
- 不要把 `--soc-version` 说成所有 simulator 场景都必选。
- 不要把 `--kernel-name` 说成对 `config` / `export` 也有效。
- 不要把经验文件中的案例说成“所有工程都必须这样”。
- 不要在没有说明前提的情况下，直接给出一个看似通用的命令。
- 没有实际 profiling 数据时，不要强行输出 Top 5 固定报告。
- 数据不足 5 项时按真实数量输出，不要补齐、不造数。
- 不要把命令指导、环境准备或经验猜测伪装成“核心瓶颈”或“优化建议”。
- 不要给出与当前模式明显不兼容的建议（例如把 simulator 参数建议给 device）。

## 快速决策树

1. **你有真实 NPU 卡，且要看真实硬件瓶颈吗？**
   - 是：优先走 **上板调优**。
2. **你没有卡，或者要看指令级流水 / 代码热点吗？**
   - 是：优先走 **仿真调优**。
3. **你的输入是什么？**
   - 可执行文件：`application`
   - JSON + `.o`：`config`
   - 已有 dump：`export`（仅 simulator）
4. **你要的是哪类结论？**
   - 真实耗时 / 内存 / Cache / Roofline：优先上板
   - 指令流水 / 每核热点 / 指令级冲突：优先仿真

## 默认输出契约

默认按两条分支输出，不要混用：

### 分支 A：结果分析 / 调优结论 / 报告化输出

满足以下任一条件时，最终回复默认使用**固定四段报告模板**：

- 用户已经给出 profiling 产物或结果现象，要求解释 / 总结 / 诊断 / 调优建议
- 用户明确要求“报告”“模板化输出”“Top 5 结论”
- 当前任务重点是解释 `csv` / `visualize_data.bin` / `trace.json` / 热点图 / 流水图，而不是教用户先跑命令

### 分支 B：命令指导 / 模式选择 / 采集前排障

满足以下场景时，不强制使用固定报告模板，继续使用 guidance-first 输出：

- 用户还没有 profiling 数据，只是在问怎么采集
- 用户重点是选 device / simulator / application / config / export
- 用户处于启动失败、环境不通、参数不会配的阶段

此时优先输出：

1. **建议路径**
   - 先明确推荐 device 还是 simulator，以及原因
2. **可直接执行的最小命令**
   - 给出与用户当前场景匹配的最小正确命令
3. **关键限制**
   - 只列当前问题真正相关的 2~5 条限制
4. **怎么看结果**
   - 告诉用户跑完后看哪一个文件、用什么工具打开、重点看什么
5. **下一步**
   - 若这是首轮采集：告诉用户下一步该加什么指标
   - 若这是排障：告诉用户下一步该补什么信息或切哪条路径

## 固定报告模板（默认用于结果分析）

### 模板启用规则

- 默认 `TOPx = TOP5`
- 如果用户明确指定别的 `x`，可覆盖默认值
- 数据不足 5 项时按真实数量输出，不补齐
- 每个 Top 条目都应尽量带 `数据来源`
- 如果当前模式或数据集不支持某项，写 `N/A`、`未采集` 或 `不适用当前模式`

### 固定标题

结果分析场景下，最终回复默认使用以下四段标题，不要自行改名：

```markdown
## 1. 算子基本信息
## 2. 关键数据 TOP5
## 3. 核心瓶颈 TOP5
## 4. 优化建议 TOP5
```

### 1. 算子基本信息

使用短表格或键值表，不使用 Top 5。默认字段：

| 字段 | 内容 |
|---|---|
| 模式 | device / simulator |
| 输入形态 | application / config / export |
| 算子名 / 目标对象 | 当前分析对象 |
| 芯片 / 仿真器 | 芯片型号或 simulator 类型 |
| 采集指标 | 当前启用的 `aic-metrics` 或主要分析视图 |
| 主要产物 | 当前引用的 CSV / bin / trace |
| 数据来源 | 文件或产物类型 |

缺字段时写 `未提供` / `未采集`，不要脑补。

### 2. 关键数据 TOP5

使用短表格，固定列：

| 排名 | 指标/对象 | 数值/现象 | 意义 | 数据来源 |
|---|---|---|---|---|

填充规则：

- 只放最值得看的关键数据，不要变成全量指标列表
- device 场景优先来自：`OpBasicInfo.csv`、`PipeUtilization.csv`、`ArithmeticUtilization.csv`、`Memory.csv`、`L2Cache.csv`、`ResourceConflictRatio.csv`
- simulator 场景优先来自：`trace.json`、`core*_code_exe.csv`、`core*_instr_exe.csv`、`PMSampling`

### 3. 核心瓶颈 TOP5

使用短表格，固定列：

| 排名 | 瓶颈结论 | 判断依据 | 影响 | 数据来源 |
|---|---|---|---|---|

填充规则：

- 必须把“结论”和“依据”分开
- 没有足够证据时写 `待确认`，不要把猜测写成确定事实
- 不要直接复述经验案例；只有当前现象匹配时，才能把经验作为辅助说明

### 4. 优化建议 TOP5

使用短表格，固定列：

| 排名 | 建议 | 对应瓶颈 | 预期收益/目的 | 前提/来源 |
|---|---|---|---|---|

填充规则：

- 建议必须与“核心瓶颈 TOP5”逐项关联
- 不要给与当前模式不兼容的建议
- 纯命令指导、环境准备或尚未验证的假设，不应冒充优化建议

## 报告模板的模式差异处理

固定模板只有一套，但字段解释按 mode 变化：

- simulator 的 `trace.json` = 指令流水图依据
- device 的 `trace.json` = 通算 / 通信相关流水图依据
- `PMSampling` 只应出现在支持的 simulator 场景
- `TimelineDetail` 只应出现在支持的 device 场景

统一原则：

- 标题和表格列固定
- 数据来源允许按 mode 变化
- 不适用项明确写 `N/A` 或 `不适用当前模式`

## 高频踩坑（优先提醒用户）

- `--kernel-name` **只支持 application 模式**；对 `--config` / `--export` 无效。
- simulator 的 `--config` 场景应通过 `LD_LIBRARY_PATH` 指定仿真器；`--soc-version` 在该场景**不生效**。
- `--export` 仅用于 simulator，且目录中应包含 dump 数据；如需代码行映射，还应包含 `aicore_binary.o`。
- `TimelineDetail` 是 **device 模式能力**，在 simulator 模式下无效。
- `--replay-mode=range` 必须配合 `--mstx=on`。
- `--replay-mode=range` 不能与 `TimelineDetail` / `Source` / `MemoryDetail` 同时使用。
- simulator 默认指标是 `PipeUtilization + ResourceConflictRatio`；`PMSampling` 默认**不开启**。
- `PMSampling` 解析全部核，`--core-id` 对它**不生效**。
- 输出、配置、导出目录会做权限与软链接检查；权限不对时工具会直接报错。
- device 模式里的 `--dump` / `--core-id` 是与特定能力和芯片绑定的特殊行为，不要当成 simulator 通用参数理解。

## 模式与输入形态矩阵

| 模式 | 输入形态 | 是否支持 | 备注 |
|---|---|---:|---|
| device | `application` | Y | 最常见；支持 `--kernel-name`、`--launch-skip-before-match` |
| device | `config` | Y | 基于 JSON + `.o`，`--kernel-name` 不生效 |
| device | `export` | N | 仅 simulator 支持 |
| simulator | `application` | Y | 可用 `--soc-version` 或 `LD_LIBRARY_PATH` 指定仿真器 |
| simulator | `config` | Y | 应使用 `LD_LIBRARY_PATH`；`--soc-version` 不生效 |
| simulator | `export` | Y | 只解析已有 dump，不重新仿真 |

## 常用命令模板

### 上板调优（application）

```bash
# 单算子默认采集
msprof op --output=./output_npu ./execute_add_op

# 采集全量基础指标 + Roofline
msprof op --aic-metrics=Roofline,Default --output=./output_npu ./execute_add_op

# 多算子：采集前 10 个匹配 Add/Sub 的算子
msprof op --launch-count=10 --kernel-name="Add|Sub" --output=./output_npu ./test
```

### 上板调优（config）

```bash
msprof op --config=./add_test.json --aic-metrics=Default --output=./output_npu
```

### 仿真调优（application）

```bash
# 方式 1：显式指定仿真器
msprof op simulator --soc-version=Ascend910B4 --output=./output_sim ./execute_add_op

# 方式 2：通过环境变量指定仿真器
export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascend910B4/lib:$LD_LIBRARY_PATH
msprof op simulator --output=./output_sim ./execute_add_op
```

### 仿真调优（config）

```bash
export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascend910B4/lib:$LD_LIBRARY_PATH
msprof op simulator --config=./add_test.json --output=./output_sim
```

### 仿真调优（export）

```bash
msprof op simulator --soc-version=Ascend910B4 --export=./dump_dir --output=./output_sim
```

## 参数边界速查

### 适用于两个模式，但要看输入形态

| 参数 | 说明 | 备注 |
|---|---|---|
| `--output` | 输出目录 | 默认当前目录，受权限与软链接限制 |
| `--launch-count` | 最大采集算子数量 | 默认 `1`，范围 `[1,5000]` |
| `--mstx` | 使能 mstx | `on/off` |
| `--mstx-include` | 只使能指定 mstx message | 必须配合 `--mstx=on` |
| `--kernel-name` | 匹配目标算子名 | **仅 application 模式有效** |

### device 模式

| 参数 | 说明 |
|---|---|
| `--aic-metrics` | 选择上板指标能力 |
| `--replay-mode` | `kernel` / `application` / `range` |
| `--launch-skip-before-match` | 跳过前 N 个算子不采集 |
| `--kill` | 采集完成后自动停止程序 |
| `--warm-up` | 预热次数，默认 `5` |

### simulator 模式

| 参数 | 说明 |
|---|---|
| `--soc-version` | `application/export` 场景可用；`config` 下不生效 |
| `--export` | 解析已有 dump |
| `--timeout` | 超时后强制终止仿真并进入解析 |
| `--core-id` | 只解析指定核，范围 `[0,49]` |
| `--dump` | 是否保留 dump；存在芯片和场景限制 |

## 指标与产物速查

### 上板 `--aic-metrics`

| 指标 | 作用 | 常见产物 | 备注 |
|---|---|---|---|
| `Default` | 基础 CSV 指标 | 多个 CSV | 默认基础采集能力 |
| `Roofline` | Roofline 瓶颈分析 | `visualize_data.bin` | 与 `Default` 绑定 |
| `Occupancy` | 核间负载分析 | `visualize_data.bin` | 仅部分芯片支持 |
| `Source` | 代码热点图 | `visualize_data.bin` | 通常需 `-g` 编译 |
| `MemoryDetail` | L2 / 内存细节增强 | CSV + `visualize_data.bin` | 与 `Default` 绑定 |
| `TimelineDetail` | 指令流水 + 上板热点图增强 | `visualize_data.bin` | device-only，且限制较多 |
| `PipeTimeline` | Pipe 流水图 | `trace.json` + `visualize_data.bin` | 仅 Atlas 350 加速卡 |
| `KernelScale` | 指定代码段采集 | CSV / 可视化 | 依赖 Kernel 侧插桩 API |
| `PcSampling` | SIMT stall 信息 | `visualize_data.bin` | 仅 Atlas 350 加速卡 |
| `BasicInfo` | 只采集基础信息 | `OpBasicInfo.csv` | 轻量模式 |

> 说明：
> - 如果用户既要 `TimelineDetail` 又要常规 CSV / 计算内存热力图，通常需要显式带上 `Default`。
> - `TimelineDetail` / `Source` / `MemoryDetail` 与 `range replay` 不能共存。

### 仿真 `--aic-metrics`

| 指标 | 作用 | 默认情况 |
|---|---|---|
| `PipeUtilization` | 指令流水 | 默认开启 |
| `ResourceConflictRatio` | 同步事件 / 冲突细节 | 默认开启 |
| `PMSampling` | 内存通路吞吐率波形图 | 默认关闭 |

## 输出产物结构

### 上板模式（单算子常见结构）

```text
OPPROF_{timestamp}_XXX/
├── dump/
├── OpBasicInfo.csv
├── PipeUtilization.csv
├── ArithmeticUtilization.csv
├── Memory.csv
├── MemoryL0.csv
├── MemoryUB.csv
├── L2Cache.csv
├── ResourceConflictRatio.csv
├── visualize_data.bin
└── trace.json            # 仅在支持的通算/特定视图场景下生成
```

### 仿真模式（单算子常见结构）

```text
OPPROF_{timestamp}_XXX/
├── dump/
└── simulator/
    ├── core0.veccore0/
    │   ├── core0.veccore0_code_exe.csv
    │   ├── core0.veccore0_instr_exe.csv
    │   └── trace.json
    ├── core0.veccore1/
    │   ├── core0.veccore1_code_exe.csv
    │   ├── core0.veccore1_instr_exe.csv
    │   └── trace.json
    ├── ...
    ├── visualize_data.bin
    └── trace.json        # 全核汇总流水图
```

### 多算子补充说明

- device 多算子输出通常会按 `OpName/<order>/...` 组织，单算子时工具可能自动平铺整理。
- simulator 多算子输出会按 `OpName/<order>/dump|simulator` 组织，且 simulator 目录中的 CSV 常带时间戳后缀。

## 如何看结果

- **CSV 文件**：适合快速查总耗时、带宽、利用率、冲突占比。
- **`visualize_data.bin`**：用 MindStudio Insight 查看热力图、Roofline、热点图、流水图等。
- **`trace.json`**：
  - device：主要用于通算/通信相关流水图；
  - simulator：主要用于指令流水图（每核与全核汇总）。
- **如果用户只说“看 trace.json”**：必须先判断该文件来自 device 还是 simulator，再决定解释方式。

## 推荐分析流程

### 上板

1. 用 `Default` 跑通一版，先看 `OpBasicInfo.csv` 和 `PipeUtilization.csv`。
2. 开 `Roofline` 判断 Compute Bound / Memory Bound / Latency Bound。
3. 若偏内存：继续看 `Memory.csv` / `MemoryDetail`。
4. 若偏计算：继续看 `ArithmeticUtilization.csv`、必要时看 `Source`。
5. 若怀疑核间不均衡：开 `Occupancy`。
6. 若是通算融合算子：再看 `trace.json`。

### 仿真

1. 先用默认指标（`PipeUtilization + ResourceConflictRatio`）拿到基本流水。
2. 看 MTE 与 VECTOR/CUBE 是否并行，是否有明显 bubble。
3. 看 `SET_FLAG/WAIT_FLAG` 是否造成不必要等待。
4. 如需内存通路波形，再显式开 `PMSampling`。
5. 数据太大时优先用 `--timeout`，热点集中到少数核时再用 `--core-id` 精细化。

## 结果解读提醒

- `Roofline` 与 pipeline 占比的推断是**分析线索**，不是唯一真相；需要结合 CSV/热点图交叉验证。
- simulator 提供的指令级视角更细，但它不等同于真实硬件最终耗时。
- 上板结果更接近真实运行瓶颈，但指令级细节通常不如 simulator 丰富。

## 深度参考

- [上板调优深度指南](references/device-tuning-guide.md) - 上板调优完整流程、关键视图、参数互斥与调优顺序
- [仿真调优深度指南](references/simulator-tuning-guide.md) - 仿真调优完整流程、真实输出结构、热点图与流水图分析

## 经验沉淀

- [经验案例：部分工程的仿真拉起需要仿真兼容构建产物](experiences/simulator-needs-sim-build.md) - 适用于 `signal 6 / Bad address` 一类仿真拉起故障的经验排查
