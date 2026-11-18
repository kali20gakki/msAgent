---
name: msopprof-range-replay-adapter
description: 为算子适配 msOpProf 范围级重放，在 C++ 或 PyTorch 调用入口添加 MSTX 范围及构建依赖，按需自动生成 custom-input 的 timeline_merge_config。适用于普通算子，也适用于多卡通算融合算子。用于接入 MSTX、生成 range 重放代码或补丁、配置多卡流水合并；默认直接修改原工程并仅生成代码，可选择编译验证及采集 Default 基础指标。不限定工程框架，需核验实际后端、芯片及工具支持范围。
---

# msOpProf 范围级重放适配

## 功能概述

为已有昇腾算子工程添加 Host 侧 MSTX 范围，适配 msOpProf 范围级重放运行方式。适用于普通算子，也适用于多卡通算融合算子。支持 C++ 与 PyTorch 调用入口，不限定工程框架；多卡流水合并场景按需自动生成合并配置。

默认直接修改原工程并仅生成代码。用户可选择仅输出补丁，或继续编译、验证正确性及采集 `Default` 基础性能指标。

## 交付选项

优先采用用户已指定的选项。未指定时展示以下选择，默认按第一项执行：

| 配置项 | 默认选项 | 可选项 |
| --- | --- | --- |
| 修改方式 | 直接修改原工程 | 仅生成 patch |
| 执行范围 | 仅生成代码和配置 | 编译、正确性验证及 range 基础性能采集 |

宿主具备交互选择工具时使用该工具，否则以文字选项供用户选择。用户已明确选择时不重复询问。

- **仅生成代码和配置**：完成源码、构建配置和命令生成，按需生成 custom-input，并执行静态检查，不执行构建或采集。
- **仅生成 patch**：在独立副本中生成包含新增文件的补丁，不修改原工程及其 Git index；选择验证时也在副本中执行。
- **编译及采集**：沿用工程构建和测试入口，先通过正确性检查，再采集性能数据。

## 执行流程

### 1. 核对工具文档并分析工程

开始适配前，先通过可用的在线搜索或网页读取工具访问 [msopprof 仓库](https://gitcode.com/Ascend/msopprof)，读取与目标工具版本对应的范围级重放典型用例、MSTX 扩展功能及命令参考正文，核对接口用法、参数、支持范围和限制。具体文档入口见文末参考资料。

优先匹配用户提供的工具版本、分支或标签；可执行文件可用时结合版本信息及 `--help` 核实。仓库默认分支不一定与安装版本一致，发现差异时以目标版本的文档和实际工具为准，记录依据。在线查询不可用时，读取与目标版本对应的本地工具仓文档；无法确认版本或取得匹配资料时，明确标注未核实的内容，不将其他版本的能力当作已确认事实。

从用户输入、构建文件和源码获取工程路径、构建命令、运行目录、C++/PyTorch 启动链及目标调用位置；确定设备、线程、stream、循环次数和目标 kernel 序列。多卡场景还需确定 rank 数和设备映射。

记录输入 shape/dtype、warmup/repeat 和原有校验方式。仅当缺失信息会改变适配路径时询问用户。
Python 调用扩展时应追踪实际 launch 的进程与线程；异步通信应识别 event、hook 与完成等待，不能仅根据函数返回位置确定范围。

### 2. 插入 MSTX 并适配构建

按 [MSTX 接入](references/mstx-adaptation.md) 修改实际调用入口和 Host 构建目标：

1. 开始与结束标记在同一线程成对调用；多卡场景各参与 rank 在相同逻辑阶段使用相同范围名称。
2. 不交叉标记，优先采用非嵌套范围。对已有嵌套标记，核对目标版本及范围使能后的实际生效行为，不假定内外层均独立重放，也不擅自拉平。检查异常与提前返回路径。
3. 单个范围内目标算子的 stream 不变；多 stream 工程按实际依赖划分范围。
4. 同步等待、输出回读及精度校验置于范围结束之后，保留原有必要同步和通信顺序。
5. 不改变算子算法、输入、rank 数、容差或运行轮次。复用已有 MSTX 配置，避免重复添加。

MSTX Host 范围与设备侧软同步上报、PrintTimeStamp 为不同能力，不得互相替代。

### 3. 按需生成合并配置

单卡普通算子或不需要多卡流水合并时，跳过本步骤，不生成 `timeline_merge_config`。其他功能需要 custom-input 时保留相应配置。

多卡流水合并场景按 [多卡流水合并配置](references/timeline-merge.md)，由 Agent 从工程和已有产物推导 `kernel_name`、`mstx_name`、`rank_count` 和 `launch_count`，直接生成或更新最终的 `custom_input.json`。用户无需手写配置或执行转换命令。

1. 读取已有 custom-input（如有），保留其他字段及不属于本次任务的合并配置。
2. 将推导结果直接写入 `timeline_merge_config` 数组，并检查字段约束、源码范围名称及重复项；冲突时核对来源，不静默覆盖。
3. “仅生成代码和配置”模式也完成 JSON 生成和校验，不涉及编译或运行算子。
4. 仅生成 patch 时，在独立副本或产物目录生成 JSON，并将新增或修改的配置纳入补丁，不写入原工程。

动态范围名称逐项展开，不将 Host API 名称直接作为设备 kernel 名称。字段无法可靠推导时，先补查工程或询问缺失信息，不编造参数。

### 4. 生成命令并按选项验证

根据全部使能范围的 kernel 调用数量确定命令行 `--launch-count`，考虑辅助 kernel 和当前版本的多卡计数作用域，不直接复用 JSON 中每组的合并次数。

```bash
<msopprof> --mstx=on --replay-mode=range --aic-metrics=Default \
  --launch-count=<采集数量> \
  --output=<新的输出目录> <应用或多rank启动命令>
```

生成 custom-input 时添加 `--custom-input=<绝对路径/custom_input.json>`；未使用自定义配置时不添加该参数。
多卡场景使用一个 msOpProf 任务拉起全部 rank。按需添加 `--mstx-include`，确保目标范围已使能。
仅生成模式到此完成；选择编译采集时，按照 [编译与采集](references/collection-guide.md) 执行并检查结果。

## 交付要求

保存源码 diff 或 patch、按需生成的 custom-input 和 `range-adaptation.md`，说明：

- 修改文件、范围名称、调用位置、参与设备及字段推导来源。
- 完整构建、正确性验证和采集命令。
- 代码生成、编译、正确性、range 采集、Default 指标及流水合并的实际状态和产物路径。

未执行的环节明确标注为未验证。基础指标采集与流水合并分别判定；未请求多卡合并时标为不适用，已请求但缺少有效单卡流水时不能报告合并通过。实际识别为 `Is MC2=1` 的目标不纳入本技能的实机验收范围，其他目标仍需依据实际环境和产物验证。

## 参考资料

| 文档 | 内容 |
| --- | --- |
| [MSTX 接入](references/mstx-adaptation.md) | C++、PyTorch 范围标记、构建配置及多卡调用边界 |
| [多卡流水合并配置](references/timeline-merge.md) | 配置字段推导、自动生成及与采集参数的关系 |
| [编译与采集](references/collection-guide.md) | 环境要求、采集命令、性能数据检查及验收标准 |
| [msopprof 范围级重放典型用例](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/best_practices/typical_cases.md) | MSTX 范围级重放说明、限制及采集示例 |
| [msopprof MSTX 扩展功能](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/extended_functions.md) | MSTX 接口及使用说明 |
| [msopprof 用户指南](https://gitcode.com/Ascend/msopprof/blob/master/docs/zh/user_guide/msopprof_user_guide.md) | `--replay-mode`、`--mstx` 等命令参数及版本限制 |

以上在线链接指向默认分支，使用时应按目标版本切换至对应分支或标签。
