# 编译与范围级重放采集

## 1. 使用条件

读取目标 msOpProf 版本的使用指南和帮助信息，记录工具版本、源码提交、CANN/driver、具体 SoC、设备和运行环境。
范围级重放适用于配套版本支持的 Atlas A2、Atlas A3 和 Ascend 950PR&950DT 产品，不能据此推定所有调用接口均可捕获。

同一范围内目标算子的 stream 不得改变。文档建议每个范围的采集算子数不超过 50，且受 Block Dim 数量限制。长循环按完整逻辑迭代划分范围。

实际使用 CUDA/NCCL 的工程不能作为 Ascend 采集入口，需提供对应 Ascend 分支及可运行调用链。工程名称不作为后端判定依据。
实际识别为 `Is MC2=1` 的目标不纳入实机验收；排除该目标不代表同仓其他算子已完成验收。

## 2. 编译与正确性验证

仅在用户选择编译采集时执行以下流程：

1. 核对 Python/Conda 环境、CANN 路径、依赖和目标设备；多卡通信场景的可用设备数量须满足通信域要求。
2. 按工程现有入口编译 Host 和 kernel，保证 MSTX 改动进入实际执行产物。
3. 检查运行时动态插件、自定义算子包和库搜索路径，不能仅凭 executable 存在判断构建完整。
4. 沿用原工程正确性测试、输入及容差。未通过时停止采集，不放宽阈值或删除校验。

SHMEM AllGather 的构建示例，在仓库根目录执行：

```bash
cmake -S . -B build -DUSE_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target allgather aclshmem_bootstrap_config_store -j4
```

其他工程以其 README 和当前源码为准。需要安装自定义包时记录安装位置及环境配置。

## 3. 基础性能采集

根据实际环境选择 `msopprof` 或 `msprof op` 入口，并核对当前工具版本的参数。

```bash
<msopprof> --mstx=on --replay-mode=range --aic-metrics=Default \
  --launch-count=<采集数量> \
  --output=<新的输出目录> <原有应用或多rank启动命令> \
  > collect.log 2>&1
```

- 根据实际范围、调用次数和辅助 kernel 推导采集预算，并核对当前工具的多卡计数作用域；JSON 的 `launch_count` 不是全任务采集数量。
- 普通算子可直接使用上述命令。仅在需要多卡流水合并或其他自定义功能时，添加 `--custom-input=<绝对路径/custom_input.json>`。
- `--mstx-include` 按精确范围名选择，多名称语法以当前帮助为准。省略时注意全部使能范围均纳入预算。
- 显式使用 `--aic-metrics=Default`，不附加 `--set`、`--kill` 或与 range 不兼容的 Source、TimelineDetail、MemoryDetail 等能力。
- 当前参考版本命令行 `--launch-count` 上限为 5000，使用前核对实际版本。
- 多卡需由同一次 msOpProf 任务拉起，不能将独立采集结果直接视为同一次任务的合并输入。

例如，SHMEM AllGather 每 rank 有 24 个单 kernel 范围时，在样例目录执行：

```bash
<msopprof> --mstx=on --replay-mode=range --aic-metrics=Default \
  --launch-count=24 --custom-input=<绝对路径/custom_input.json> \
  --output=<新的输出目录> \
  bash run.sh -pes 2 -type float16_t -fnpu <空闲连续卡的起始号>
```

应从当前 `run.sh` 和调用循环核对执行次数，不固定沿用示例数值。

## 4. 性能数据检查

Default 基础指标包括以下七类，另需检查 `OpBasicInfo*.csv`：

| 指标 | 常见产物 |
| --- | --- |
| ArithmeticUtilization | `ArithmeticUtilization*.csv` |
| L2Cache | `L2Cache*.csv` |
| Memory | `Memory*.csv` |
| MemoryL0 | `MemoryL0*.csv` |
| MemoryUB | `MemoryUB*.csv` |
| PipeUtilization | `PipeUtilization*.csv` |
| ResourceConflictRatio | `ResourceConflictRatio*.csv` |

按参与设备、kernel 和调用实例检查有效数据行及对应原始数据。文件命名以工具版本为准，不能用 MemoryL0 文件代替 Memory 指标。
检查子进程返回码、golden 结果及 msOpProf 的缺失 PMU/duration、采集失败和零耗时等异常。结束提示或 shell 返回 0 不能单独证明采集成功。
产品确实不支持的字段须附版本依据，单列为不支持，不能据此宣称七类指标全部通过。

## 5. 结果判定

| 环节 | 通过条件 |
| --- | --- |
| 代码和配置生成 | 标记成对、调用边界正确、构建依赖完整、JSON 合法且保留已有配置 |
| 编译 | Host、kernel 及运行依赖构建成功，产物可定位 |
| 正确性 | 目标算子的原有校验通过；多卡场景覆盖所有参与 rank |
| Range 采集 | 日志确认实际进入 range 模式，目标 kernel 和设备完整 |
| Default 指标 | 各设备目标 kernel 有有效基础性能数据，无未解释的缺失或失败 |
| 流水合并 | 存在有效单卡流水，合并结果的设备集合、事件和次数符合预期 |

仅生成代码时，后续环节记为未执行。不需要多卡流水合并时，该项记为不适用；需要合并但没有有效单卡流水时，单列为“无可合并流水”。不将配置生成等同于合并成功。
普通 kernel 重放成功不能代替 range 验证成功，也不能用一个工程的成功替代另一个工程的验收。
记录真实命令、日志、产物和失败原因，不将未验证的预期写成结果。
