# 上板调优深度指南

本文件聚焦 **`msprof op` 上板调优**。如果用户要看指令级流水、每核热点图或 dump 复用解析，请切换到 `simulator-tuning-guide.md`。

## 1. 适用范围与前置检查

### 适用范围

- 有真实昇腾设备。
- 目标是看真实硬件耗时、带宽、Cache、Roofline、核间负载、通算流水等。
- 输入形态可以是：
  - `application`：可执行文件
  - `config`：JSON + `.o`

### 开始前检查清单

1. app 本身能够正常运行。
2. 输出路径、配置路径不含不安全软链接，且父目录权限符合工具要求。
3. 如果要看代码热点图 / 代码调用栈，算子编译时应带 `-g`。
4. 若涉及 `range replay`，确认用户代码里已经有成对的 mstx 标记。

## 2. 编译准备

如果用户需要 **代码热点图 / Cache 热力图跳转 / 更完整的源码映射**，应在算子编译阶段加入调试信息：

```bash
# 示例：在算子编译配置中添加 -g
add_ops_compile_options(ALL OPTIONS -g)
```

然后重新编译、重新部署算子包。

> 注意：
> - `-g` 会把调试信息带入二进制，需控制访问权限。
> - 官方文档明确指出不支持 `-O0` 编译选项。

## 3. 采集路径选择

### 3.1 application 场景

```bash
# 单算子默认采集
msprof op --output=./output ./execute_add_op

# 全量基础指标 + Roofline
msprof op --aic-metrics=Roofline,Default --output=./output ./execute_add_op
```

### 3.2 多算子 application 场景

```bash
# 采集前 10 个匹配 Add/Sub 的算子
msprof op --launch-count=10 --kernel-name="Add|Sub" --output=./output ./test

# 跳过前 3 个算子，从第 4 个开始采集 5 个
msprof op --launch-skip-before-match=3 --launch-count=5 --output=./output ./test
```

说明：

- `--kernel-name` 只在 application 模式有效。
- `--launch-skip-before-match` 计数时不要求先命中 `kernel-name`。

### 3.3 config 场景

```bash
msprof op --config=./add_test.json --aic-metrics=Default --output=./output
```

此模式常用于没有直接 app 拉起路径、但已经有 JSON + `.o` 配置的情况。

## 4. replay 模式选择

| 模式 | 用法 | 适合场景 | 注意事项 |
|---|---|---|---|
| `kernel` | `--replay-mode=kernel` | 默认模式；聚焦单个算子核函数 | 最稳妥 |
| `application` | `--replay-mode=application` | 希望保留应用级上下文 / L2 状态 | 单独使能部分指标时，`visualize_data.bin` 可能缺部分数据 |
| `range` | `--replay-mode=range --mstx=on` | 需要对指定多算子范围整体重放 | 限制最多，务必先检查兼容性 |

### `range` 模式重点限制

- 必须配合 `--mstx=on`。
- 只支持特定芯片系列（仓内文档按 A2/A3 描述）。
- 不支持与 `MemoryDetail`、`TimelineDetail`、`Source` 同时使能。
- 不建议与 `--kill=on` 同时使用。
- 对通算融合算子存在额外限制，具体以当前版本 user guide 和安装版本帮助信息为准。

## 5. 按目标选指标

| 目标 | 推荐指标 | 主要看什么 |
|---|---|---|
| 先确认整体耗时是否异常 | `Default` | `OpBasicInfo.csv`、`PipeUtilization.csv` |
| 判断是算力瓶颈还是带宽瓶颈 | `Roofline,Default` | `visualize_data.bin` 中的 Roofline 视图 |
| 看核间是否负载不均 | `Occupancy,Default` | 各核耗时 / 吞吐 / Cache 命中率差异 |
| 看源码热点 / 代码行热点 | `Source,Default` | `visualize_data.bin`，通常需 `-g` |
| 看 L2 / 内存细节 | `MemoryDetail` | L2 命中率、GM 相关搬运量、MTE1/MTE2 活跃带宽 |
| 看 TimelineDetail 上板指令相关视图 | `TimelineDetail,Default` | 仅 A2/A3 等特定场景支持，限制较多 |
| 看 Pipe 流水图 | `PipeTimeline` | 仅 Atlas 350 加速卡 |
| 只想要最轻量基础信息 | `BasicInfo` | 只生成 `OpBasicInfo.csv` |

## 6. 结果查看

### 6.1 CSV 首轮分析

| 文件 | 典型问题 | 快速判断方法 |
|---|---|---|
| `OpBasicInfo.csv` | 总耗时异常 | 先确认算子名、block dim、总耗时 |
| `PipeUtilization.csv` | 计算/搬运不平衡 | 看各 pipe 耗时占比 |
| `ArithmeticUtilization.csv` | 计算单元利用率低 | 看 Cube / Vector 指令耗时和占比 |
| `Memory.csv` | 主通路带宽不足 | 看 UB/L1/L2/GM 读写带宽 |
| `MemoryUB.csv` | 各 block 差异大 | 看是否存在核间不均衡 |
| `L2Cache.csv` | 命中率低 | 看 L2 Hit/Miss 情况 |
| `ResourceConflictRatio.csv` | 资源冲突高 | 看 bank conflict / 资源冲突占比 |

### 6.2 `visualize_data.bin`

导入 MindStudio Insight 后，通常会看到：

- 计算内存热力图
- Roofline 瓶颈分析图
- Cache 热力图
- 算子代码热点图
- 通算相关可视化

### 6.3 `trace.json`

在上板模式里，`trace.json` 主要用于 **通算/通信相关流水图**。  
它的语义与 simulator 下的 `trace.json` 不同，不要混用解释。

## 7. 关键视图怎么解读

### 7.1 计算内存热力图

重点看三类信息：

1. **核间负载分析（Occupancy）**
   - 若最大值和最小值差异显著（官方文档里给出 10% 量级经验阈值），通常说明负载不均。
2. **计算负载分析**
   - 看 Cube / Vector 利用是否偏低。
3. **内存负载分析**
   - 看 MTE 通路活跃带宽是否成为瓶颈。

### 7.2 Roofline

可把它当成“瓶颈初判器”：

- 性能点靠近算力屋顶：更偏 **Compute Bound**
- 性能点靠近带宽斜线：更偏 **Memory Bound**
- 两边都没贴近：更可能是 **Latency Bound**，需结合 pipeline / memory / compute caused 再细看

### 7.3 Cache 热力图

适合回答：

- 哪些源码位置或指令片段 L2 Hit/Miss 异常？
- 低命中率是否集中在某些热点区域？

前提通常是：

- `Source` 已开启
- 算子带 `-g`
- 当前芯片 / 算子类型支持该视图

### 7.4 算子代码热点图

左侧通常偏源码维度，右侧偏指令维度。  
它适合把“耗时高的代码行”和“具体耗时指令”对应起来。

### 7.5 通算流水图

这里是仓内文档口径最容易混淆的点之一：

- user guide 的摘要位置对支持范围写得更保守；
- 通算流水图专章又给出更展开的 MC2/LCCL/ASC 描述。

因此：

- 若只是泛化说明，可说“适用于支持的通算融合算子场景”；
- 若用户问“我这个算子类型到底支不支持”，应提示以 **当前安装版本帮助信息 + 当前版本专章限制** 为准。

## 8. 常见误区

1. **上来就开最重的指标**
   - 正确做法：先用 `Default` 跑通，再定向加 `Roofline` / `Source` / `MemoryDetail`。
2. **把 `TimelineDetail` 当成所有上板热点分析的默认入口**
   - 实际上它限制较多，`Source` 往往更适合作为常规热点分析入口。
3. **忽略权限问题**
   - 输出目录、配置目录、导出目录权限不符合要求时，工具可能直接失败。
4. **把通算 `trace.json` 当成 simulator 指令流水图**
   - 两者同名，但语义不同。

## 9. 推荐调优顺序

1. `Default`：确认是否真的有性能问题。
2. `Roofline`：定大方向（算力 / 带宽 / 延迟）。
3. `MemoryDetail` 或 `Source`：做定向下钻。
4. `Occupancy`：看核间不均衡。
5. 通算或特定芯片能力：再看 `trace.json` / `PipeTimeline` / `PcSampling`。
