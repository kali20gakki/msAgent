# 仿真调优深度指南

本文件聚焦 **`msprof op simulator`**。它适合没有真实设备、或需要更细粒度指令级分析的场景。

## 1. 适用范围与模式边界

### 更适合 simulator 的问题

- 指令流水是否存在 bubble？
- MTE 与 VECTOR/CUBE 是否充分并行？
- SET_FLAG / WAIT_FLAG 是否造成等待？
- 哪些代码行、哪些指令最耗时？
- 是否需要每核热点图、每核 `trace.json`、吞吐率波形图？

### 不要混淆的边界

- simulator 的 `--aic-metrics` 只有：
  - `PipeUtilization`
  - `ResourceConflictRatio`
  - `PMSampling`
- `TimelineDetail` 是 **device 模式能力**，不是 simulator 参数。
- simulator 下的 `trace.json` 是 **指令流水图**；device 下的 `trace.json` 更多是通算/通信相关流水图。

## 2. 启动前先确认三件事

### 2.1 仿真器类型怎么指定

有两种方式：

1. `--soc-version=Ascendxxxyy`
2. `LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH`

但要注意：

- `application` / `export`：两种方式都可用。
- `config`：应使用 `LD_LIBRARY_PATH`；`--soc-version` 在该场景不生效。

### 2.2 是否需要 `-g`

如果用户要看 **代码行映射 / 调用栈 / 更完整热点图**，建议编译时带 `-g`。

### 2.3 构建产物是否仿真兼容

不是所有工程都能直接拿“设备侧可执行文件”拉起仿真：

- 某些官方示例或工程产物可同时运行在设备和仿真器上；
- 某些模板库或工程构建链则需要显式启用 simulator 构建。

如果用户遇到 `signal 6`、`Bad address`、`std::__ios_failure` 等拉起错误，先看 `experiences/simulator-needs-sim-build.md`。

## 3. 数据采集路径

### 3.1 application 场景

```bash
# 方式 1：显式指定仿真器
msprof op simulator --soc-version=Ascend910B4 --output=./output_sim ./execute_add_op

# 方式 2：通过环境变量指定仿真器
export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascend910B4/lib:$LD_LIBRARY_PATH
msprof op simulator --output=./output_sim ./execute_add_op
```

### 3.2 config 场景

```bash
export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascend910B4/lib:$LD_LIBRARY_PATH
msprof op simulator --config=./add_test.json --output=./output_sim
```

要点：

- `--config` 下不要再带 `--soc-version`。
- `--kernel-name` 对 `--config` 不生效。

### 3.3 export 场景（只解析已有 dump）

```bash
msprof op simulator --soc-version=Ascend910B4 --export=./dump_dir --output=./output_sim
```

`--export` 目录要求：

- 目录中应是 dump 数据和相关核函数文件。
- 若需要代码行映射，应包含名为 `aicore_binary.o` 的算子核函数文件。
- 若只是纯 dump，没有 `aicore_binary.o`，仍可做流水解析，但无法完整做代码行映射。

## 4. 仿真专用参数怎么用

### `--soc-version`

参考 `${INSTALL_DIR}/tools/simulator/` 下的目录名，例如：

- `Ascend910B4`
- `Ascend910_9391`
- `Ascend310B4`
- `Ascend950`

### `--timeout`

适合“大算子、长仿真”场景：

```bash
msprof op simulator --soc-version=Ascend910B4 --timeout=5 --output=./output_sim ./app
```

说明：

- 单位是分钟，范围 `[1,2880]`。
- 超时后工具会杀掉仿真进程并直接进入解析。

### `--core-id`

只解析指定核，适合算子分布均匀、只想深挖个别核时使用：

```bash
msprof op simulator --soc-version=Ascend910B4 --core-id="0|31" --output=./output_sim ./app
```

说明：

- 范围 `[0,49]`
- 只影响部分核解析
- **对 `PMSampling` 不生效**

### `--dump`

```bash
msprof op simulator --soc-version=Ascend910B4 --dump=on --output=./output_sim ./app
```

说明：

- 默认 `off`
- A2/A3 系列下可用于控制是否保留 dump
- 对部分 Atlas 推理系列产品，文档说明该参数不生效，dump 会按正常流程落盘
- 仅适用于单进程场景

## 5. 指标与视图

### 默认指标

simulator 默认会启用：

- `PipeUtilization`
- `ResourceConflictRatio`

这意味着：

- 即使用户没显式传 `--aic-metrics`，通常也能直接看到基础流水图与同步事件细节。

### `PipeUtilization`

- 只显示指令流水。
- 更适合先看整体执行时序。

### `ResourceConflictRatio`

- 在流水之外，提供 SET/WAIT FLAG 等同步事件细节。
- 更适合分析同步等待或冲突问题。

### `PMSampling`

```bash
msprof op simulator --soc-version=Ascend910B4 --aic-metrics=PMSampling --output=./output_sim ./app
```

用途：

- 展示内存通路吞吐率波形图。
- 重点看：
  - `GM <-> L1`
  - `GM <-> UB`
  - `GM <-> other`

注意：

- 默认不开启。
- 解析全部核，`--core-id` 对其无效。

## 6. 真实输出结构

### 单算子常见结构

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
    └── trace.json
```

解释：

- 每个 `core*.veccore*` / `core*.cubecore*` 子目录下都有该核的局部结果。
- `simulator/trace.json` 是全核汇总流水图。
- `visualize_data.bin` 是 Insight 使用的汇总可视化文件。

### 多算子结构

多算子时通常会变成：

```text
OPPROF_{timestamp}_XXX/
└── OpName/
    └── 0/
        ├── dump/
        └── simulator/
```

而 simulator 下的 CSV 常带时间戳后缀。

## 7. 结果怎么看

### 7.1 指令流水图

用 Chrome `chrome://tracing` 或 MindStudio Insight 查看 `trace.json`。

重点观察：

- 各流水线是否长时间空闲
- MTE 和 VECTOR/CUBE 是否并行
- 同步指令是否造成明显停顿

### 7.2 算子代码热点图

simulator 的热点图比上板更“指令级”，常见信息包括：

- 源码与指令映射
- 执行次数
- Cycles 耗时
- UB Conflict
- Process Bytes
- 部分平台支持的寄存器相关信息

### 7.3 `core*_code_exe.csv` / `core*_instr_exe.csv`

- `core*_code_exe.csv`：更适合看“哪段代码最耗时”
- `core*_instr_exe.csv`：更适合看“哪条指令最耗时 / 执行次数最高”

## 8. 仿真特有技巧

### 8.1 先全核，后定核

```bash
# 先快速跑一版
msprof op simulator --soc-version=Ascend910B4 --timeout=1 --output=./quick ./app

# 再深挖目标核
msprof op simulator --soc-version=Ascend910B4 --core-id="0" --output=./detail ./app
```

### 8.2 先截断，再定位

大算子完整仿真可能非常慢，优先用 `--timeout` 拿到“足够诊断”的部分流水。

### 8.3 dump 复用

```bash
# 第一次仿真时保留 dump
msprof op simulator --soc-version=Ascend910B4 --dump=on --output=./output ./app

# 后续仅从 dump 解析
msprof op simulator --soc-version=Ascend910B4 --export=./output/dump --output=./output2
```

## 9. 仿真 vs 上板热点图差异

| 功能 | 上板 | 仿真 |
|---|---|---|
| 指令级 Cycles | 弱 / 不强调 | 强 |
| 每核信息 | 弱 | 强 |
| UB Conflict | 一般不主打 | 强 |
| GPR 相关信息 | 受平台限制 | 某些平台更丰富 |
| L2Cache 命中率按代码行/指令看 | 上板更强 | 通常不作为主特性 |

## 10. 常见问题

### 10.1 仿真时间太长

- 用 `--timeout`
- 用 `--core-id`
- 减少非必要算子
- 检查 block_dim 是否过大

### 10.2 代码行映射缺失

- 确认编译带 `-g`
- `--export` 时确认目录中有 `aicore_binary.o`

### 10.3 `PMSampling` 没数据

- 它默认不开启，需显式加 `--aic-metrics=PMSampling`
- 不要误以为 `--core-id` 会影响它

### 10.4 用户把 `TimelineDetail` 用到 simulator

- 直接指出：这是 device 模式能力，不是 simulator 参数。
