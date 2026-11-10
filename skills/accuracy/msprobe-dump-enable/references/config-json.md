# config.json 配置参考

本文是 msProbe 动态图（eager）PyTorch 场景 `PrecisionDebugger` 采集的配置速查，内容来源于
[msProbe 官方文档 config.json 介绍](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/config_json_introduct.md)
与 [PyTorch场景精度数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)。
本节只覆盖本技能范围（PyTorch eager）；MindSpore/静态图相关选项不展开。

## 1. 配置来源与优先级

- 调用 `PrecisionDebugger(config_path=...)` 时加载指定 config.json；未指定 `config_path` 时使用安装目录下的默认 `msprobe/config.json`
  （`pip show mindstudio-probe` 可查安装路径），其默认内容为：`task: statistics`、`dump_path: ./dump_path`、`rank: []`、`step: []`、
  `level: L1`、`extra_info: true`、`async_dump: false`。
- 构造函数参数 `task` / `dump_path` / `level` / `step` 优先级高于 config.json 文件中的同名配置；全部不配置时，
  默认采集所有 rank、所有 step 的 L1 级别统计数据（task=statistics）。
- `dump_enable` 配置在初始化时生效一次，运行中工具会自动重读 `config_path`，实现动态启停（见下文 dump_enable）。

## 2. 通用字段

| 参数 | 可选/必选 | 取值与说明 |
| --- | --- | --- |
| `task` | 可选，默认 `"statistics"` | `"statistics"` 仅统计信息；`"tensor"` 统计+完整真实数据；`"acc_check"` 精度预检（仅 PyTorch）；`"nan_check"` NaN/溢出检测（仅 PyTorch NPU，见 [nan-check.md](nan-check.md)）；`"structure"` 仅模型结构与调用栈。`"exception_dump"` 仅 MindSpore，勿用。代码中另有未文档化的 `diff_check`/`overflow_check`，不建议使用 |
| `dump_path` | **必选** | dump 数据目录，str，推荐绝对路径 |
| `rank` | 可选，默认全采 | `list[Union[int, str]]`，元素为 ≥0 整数、`"0"` 或 `"4-6"` 范围字符串。PyTorch 场景 Rank ID 从 0 计数，最大为所有节点可用卡总数-1；配置实际未运行的 Rank ID 则 dump 数据为空。**单卡训练必须为 `[]`** |
| `step` | 可选，默认全采 | `list[Union[int, str]]`，如 `[0, 1, "2", "4-6"]`；必须配置训练脚本中实际存在的 step。`[]` 表示全部 step |
| `level` | 可选，默认 `"L1"` | `"L0"` 模块级；`"L1"` API 级（L1/mix 仅 PyTorch、MSAdapter、MindSpore**动态图**场景支持）；`"L2"` kernel 级（约束见 [kernel-dump-l2.md](kernel-dump-l2.md)）；`"mix"` L0+L1；`"debug"` 单点保存（配合 `save()`） |
| `async_dump` | 可选，默认 `false` | 异步落盘（step 结束后统一落盘，过程中不触发同步）。**有显存溢出风险**；task=tensor 时**必须配置 list**；该模式下 summary_mode 不支持 md5，不支持复数类型 tensor 统计量 |
| `dump_enable` | 可选 | dump 动态启停开关，`true`/`false`。建议仅在需要动态控制采集时配置，初始值设 `false`。PyTorch 场景：初始化时配置了该字段，则运行中自动重读 config_path 刷新配置（其他字段修改也生效）。推荐流程：常规阶段关闭→定位问题时开启→定位完关闭 |
| `extra_info` | 可选，默认 `true` | 是否输出 `stack.json`（调用栈）与 `construct.json`（分层结构）；`false` 不生成 |
| `precision` | 可选，默认 `"low"` | `"high"` 统计量用 float32 计算，精度更高但**可能显存溢出**；`"low"` 用原始类型计算 |
| `risk_level` | 可选，默认 `"FOCUS"` | 仅 PyTorch 且 level 为 L1/mix 生效。`"ALL"` 全部 API；`"CORE"` 仅核心高风险 API；`"FOCUS"` 核心+关注 API，排除 reshape/transpose/permute/to/view 等形状变换类低风险 API |
| `custom_op_namespaces` | 可选，默认 `["_C_ascend"]` | 自定义算子命名空间列表（代码支持字段，文档未展开） |

## 3. 任务专属段

### statistics

```json
{"scope": [], "list": [], "tensor_list": [], "data_mode": ["all"], "summary_mode": "statistics", "slice": [], "request_id": ""}
```

`tensor_list` 指定某一类 API 或模块时额外采集完整 tensor 数据（仅 level L0/L1/mix）。其余参数见第 4 节。

### tensor

```json
{"scope": [], "list": [], "data_mode": ["all"], "bench_path": "", "summary_mode": "statistics", "diff_nums": 1, "slice": [], "request_id": ""}
```

- `bench_path`：PyTorch 确定性问题的 md5 实时差异分析。传入上一次 dump（summary_mode=md5）的数据路径，本次同样配置 `summary_mode: "md5"` 时，工具识别与预置 md5 有差异的节点后自动 dump 真实数据。
- `diff_nums`：最大差异次数，int，默认 1；`-1` 表示持续检测差异直到训练结束。仅配合 bench_path 生效。

### structure

仅采集模型结构，无专属配置。官方示例 `level: "mix"`。

### nan_check

字段为 `scope`/`list`/`check_mode`/`tensor_list`/`data_mode`，全部约束与构建要求见 [nan-check.md](nan-check.md)。

## 4. 关键参数详解

### list

- API 全称：`"Tensor.permute.1.forward"` 格式（`{api_type}.{api_name}.{API调用次数}.{forward/backward}`）；**level 为 L2 时 list 必填且只能 1 个**。
- 类别名：`"relu"`，dump 该类 API 的输入输出数据。
- level 为 mix 时可用模块名：`"Module.module.language_model.encoder.layers.0.mlp.ParallelMlp.forward.0"`，dump 该模块展开数据（从执行开始到结束的所有数据）。

### scope

列表长度**必须恰好为 2**（闭区间锁定范围），两个元素按工具命名格式配置完整模块名或 API 名。
level=L0 配模块名、L1 配 API 名、mix 两者皆可。示例：`["Tensor.add.0.forward", "Functional.square.2.forward"]`。

### data_mode

取值 `"all"`、`"forward"`、`"backward"`、`"input"`、`"output"`；默认 `["all"]` 且**不能与其他值组合**，其余自由组合。
示例：`["forward", "input"]` 仅保存前向输入数据。nan_check 任务中配置 `"input"` 必须同时配置 `"output"`。

### summary_mode

- `"statistics"`（默认）：dump.json 仅统计量（max/min/mean/L2norm）。
- `"md5"`：额外输出 CRC-32 校验值（字段名 `md5`），用于验证数据完整性/确定性分析。
- `"xor"`（仅 PyTorch）：仅输出 XOR 校验值（字段名 `md5`），不输出 max/min/mean/L2norm；配合编译时
  `--include-mod=xor_checksum` 的工具包可数倍加速（加速算子不可用时自动回退，采集不中断）。

### slice

`list[dict]`，key ∈ {`dim`(默认0), `size`(触发条件，目标维长度等于 size 才生效), `begin`(闭，默认0), `end`(开，默认末尾)}，value 为整数。
示例：`[{"dim": 1, "size": 50, "begin": 10, "end": 20}]`。仅对 statistics/tensor 生效，支持 L0/L1/mix。

### request_id

str，对 tensor 第 0 维按请求切片，优先级高于 slice，需搭配 `start(scheduled_tokens={"<request_id>": n})` 使用。
仅对 statistics/tensor 生效。

## 5. 输出目录结构

```
dump_path/
├── step0/
│   ├── rank0/                    # 分布式按 rank{ID}；进程取不到 rank 信息时用 proc{pid}
│   │   ├── dump_tensor_data/     # tensor 任务采集的 .pt 文件
│   │   ├── dump.json             # 统计量/校验值/is_nan 检测结果
│   │   ├── stack.json            # API/Module 调用栈（extra_info=true）
│   │   ├── construct.json        # 分层分级结构（L1 时内容为空）
│   │   └── dump_error_info.log   # 仅在 dump 报错时生成
│   └── ...
├── step1/ ...
```

- 采集完成打印 `dump.json is at ./dump_path/step*`，打印最后一个 step 编号时表示 dump 结束。
- `.pt` 命名（API 级）：`{api_type}.{api_name}.{API调用次数}.{forward/backward}.{input/output}.{参数序号}`，
  如 `Functional.linear.5.backward.output.pt`；list 参数按 `1.1` 继续排序。模块级命名见官方文档 dump 目录说明。
- `dump.json` 统计字段：max/min/mean/L2norm。**mean/L2norm 仅 fp32/fp16/bf16 计算**，其余 dtype 只有 max/min；
  NZ 格式 tensor 的统计量全为 `null`；`summary_mode=xor` 时仅 `md5` 字段。
- 落盘时序：pt 文件在算子执行后即落盘；**json 文件须正常执行 `stop()` 后才写入完整数据**（异常终止可能丢 json 数据）。
- API 前缀对照：Tensor=torch.Tensor、Torch=torch、Functional=torch.nn.functional、NPU=NPU亲和算子、
  VF=torch._VF、Aten=torch.ops.aten、Distributed=torch.distributed、MindSpeed=mindspeed.ops、Triton=triton。

## 6. 接口参考（msprobe.pytorch）

`PrecisionDebugger` 及 `start`/`stop`/`step`/`set_init_step`/`save`/`register_custom_api`/`seed_all`
等接口的完整签名、参数说明与调用示例见 [api-reference.md](api-reference.md)。

## 7. 安装

- pip 安装：`pip3 install -U mindstudio-probe`（升级时关注[版本配套关系](https://gitcode.com/Ascend/release-management/blob/master/MindStudio/master/release_notes.md)）。
- 源码安装：`python3 build.py [local] [-e include-mod=<module>] [-e no-check=true|false]`。
  `include-mod` 取值：`all`、`tb_graph_ascend`、`trend_analyzer`、`atb_probe`、`aclgraph_dump`、`nan_check`、`xor_checksum`，
  多模块用逗号连接（如 `tb_graph_ascend,trend_analyzer`）。
  **编译 `aclgraph_dump`/`nan_check`/`xor_checksum` 模块前必须先安装 torch 与 torch_npu**（如 `pip3 install torch_npu==2.7.1.post4`）。
  构建产物 wheel 安装：`pip install artifacts/mindstudio_probe*.whl`；也可用 PYTHONPATH 源码方式（装好 pyproject.toml 声明的依赖后 `export PYTHONPATH="$PWD/python:$PYTHONPATH"`）。

## 8. 约束汇总（采集前检查）

- 仅支持 PyTorch；**暂不支持 PyTorch 2.7 及以上版本的 dynamo 场景**；L1/mix 级别仅动态图场景支持。
- 因 PyTorch 自动微分机制限制，dump 数据可能缺少**原地操作**模块/API（如 `nn.ReLU(inplace=True)`）及其上一个模块/API 的反向数据。
- 使用工具后 loss/gnorm 可能变化：工具中的 item 操作引入同步、hook 机制等原因所致。
- 采集之后的分析（msprobe compare、比对、可视化）不属于本技能范围。
