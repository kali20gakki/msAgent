---
name: msprobe-dump-enable
description: "在用户的 PyTorch 动态图（eager）训练/推理代码或 vLLM 推理服务（vllm-ascend、社区原生 vLLM eager）中使能 msProbe 精度调试工具的 PrecisionDebugger 数据采集功能，检查 msProbe 安装与环境、确认纯 eager 场景、与用户逐项确认采集配置、插入打点代码（vllm-ascend 场景免插桩，经 --additional-config 传配置；其余场景先展示 diff 经确认后修改）、验证并交付开启/关闭说明。用于“帮我使能 dump / 给训练代码接入精度数据采集”这类请求；不负责精度问题定位分析。"
keywords: [dump, msprobe, 精度数据采集, PrecisionDebugger, eager, 动态图, 使能]
---

# msProbe 动态图精度数据采集使能

交付物：在用户训练/推理脚本中插入 `PrecisionDebugger` 采集打点 + 经用户确认的 `config.json` +
语法验证（环境允许且用户同意时做一次限步短跑验证）+ 开启/关闭说明。所有事实以
[msProbe 官方文档](https://gitcode.com/Ascend/msprobe)为准，详见文末参考资料。

## 适用范围与重定向

**适用**：用户自有代码中的 PyTorch 动态图（eager）训练/推理流程，以及 vLLM 推理服务场景（vllm-ascend 已内置 msProbe 接入、无需插桩；社区原生 vLLM（GPU）仅支持 eager 采集，需在 `GPUModelRunner` 中插桩），请求使能精度数据 dump 或接入精度数据采集。

**不适用（逐条说明去向，不要硬做）**：

| 情况 | 处理 |
| --- | --- |
| PyTorch ≥ 2.7 的 dynamo 场景 | 工具不支持，明确告知不可行 |
| MindSpore / MSAdapter / aclgraph / torchair 图模式 | 技能不支持这些场景，官方另有对应文档 |
| verl 场景 | 技能不支持强化学习这类场景，另有对应 skill 支持 |
| monitor（`TrainerMon`，`msprobe/pytorch/monitor`） | 与 dump 完全不同的监控子系统，不混淆、不接入 |
| 精度问题本身的定位分析（比对、归因、可视化） | 本技能止步于“使能采集”；分析转向 `precision-debugger`、`loss-gnorm-diff-analyzer`、`nan-overflow-detection` 等技能 |

## 工作流

### 1. 触发判定

- 请求为“帮我使能 dump / 给代码接入 msProbe 采集 / 加精度数据采集打点”等意思时 → 进入本技能。
- 请求为“帮我定位精度问题 / 分析 dump 数据”等意思时 → 不进入本技能。

### 2. 环境检查（第一步，必做）

1. 检查 msProbe 是否已安装：
   ```bash
   pip show mindstudio-probe || python3 -c "import msprobe; print(msprobe.__file__)"
   ```
   未安装 → 报告并给出 `pip3 install -U mindstudio-probe`，**征得用户同意后**再安装。
   nan_check 或 xor 加速等功能需要源码编译安装，见 [nan-check.md](references/nan-check.md)
   与 [config-json.md](references/config-json.md#7-安装)。
2. 确认 PyTorch 版本与执行模式：`python3 -c "import torch; print(torch.__version__)"`。
   ≥ 2.7 且用户代码使用 dynamo（`torch.compile` 等）→ 告知不支持。
3. NPU 能力检查**仅在用户选择 L2 kernel dump 或 nan_check 时需要**：
   `python3 -c "import torch, torch_npu; assert torch.npu.is_available()"`。

### 3. 场景确认

读取用户启动脚本与训练/推理文件，定位训练循环或前向入口，确认是纯 eager PyTorch。
确定插入点文件（多文件时实例化放在导包后、目标进程的入口处；`PrecisionDebugger` 初始化必须与采集目标在同一个进程中）。

vLLM 推理服务场景分两类：**vllm-ascend 无需插桩**——已内置 msProbe 接入能力，启动服务时经 `--additional-config`
传入 dump 配置内容（`dump_config`）或配置文件路径（`dump_config_path`）；**社区原生 vLLM（GPU）仍需插桩**——
在 `GPUModelRunner` 的 `__init__`/`execute_model` 中接入 `PrecisionDebugger`（start/stop/step），且仅支持
eager 采集，启动须加 `--enforce-eager`。详见 [vllm-ascend.md](references/vllm-ascend.md)。

### 4. config.json 逐项确认（必须与用户确认，禁止代填）

以下核心字段逐项询问用户，每项说明取值含义并给出推荐默认；完整配置**写盘前展示给用户确认**：

| 字段 | 询问内容 | 推荐默认 |
| --- | --- | --- |
| `task` | statistics（轻量统计值采集）/ tensor（完整数据采集）/ nan_check（NaN/溢出检测，约束见 [nan-check.md](references/nan-check.md)）/ structure（仅模型结构采集） | `statistics` |
| `level` | L1（API 级）/ L0（模块级）/ mix（L0+L1）/ debug（单点 save）/ L2（kernel 级，约束见 [kernel-dump-l2.md](references/kernel-dump-l2.md)） | `L1` |
| `rank` | `[]` = 全部 rank；单卡必须 `[]`；多卡 `[0,1]` 或 `["4-6"]`；配未运行的 rank 数据为空 | `[]` |
| `step` | `[]` = 全部 step；建议短跑 `[0,1]`；支持 `["4-6"]` 范围 | `[0,1]`（短跑）/ `[]`（正式） |
| `dump_path` | 必选，推荐绝对路径 | 由用户指定 |

无需逐项追问、按默认填写的字段：`async_dump: false`、`extra_info: true`（生成 stack.json 便于回溯代码）。
字段全量说明见 [config-json.md](references/config-json.md)。

### 5. 插装（先展示 diff，用户确认后修改）

> vllm-ascend 场景无需插桩（已内置 msProbe 接入），跳过本步，按 [vllm-ascend.md](references/vllm-ascend.md)
> 给出带 `--additional-config` 的启动命令即可；社区原生 vLLM（GPU）仍需插桩，官方模板见
> [vllm-ascend.md](references/vllm-ascend.md) 的"社区原生 vLLM（GPU）插装"一节。

dump采集基础示例如下，示例代码中的高亮显示为需要在用户脚本中添加的代码。

```diff
+# 导入工具数据采集接口，尽量在迭代训练文件导包后的位置执行seed_all和实例化PrecisionDebugger
+from msprobe.pytorch import seed_all, PrecisionDebugger
+# 在模型训练开始前固定随机性
+seed_all()
+# PrecisionDebugger实例化，加载dump配置文件
+debugger = PrecisionDebugger(config_path="./config.json")

 # 模型、损失函数的定义及初始化等操作
 # ...

 # 数据集迭代的位置一般为模型训练开始的位置
 for data, label in data_loader:
+    debugger.start(model)  # 开启数据dump

     # 如下是模型每个step执行的逻辑
     output = model(data)
     # ...
     # 计算梯度
     loss.backward()
     # 更新参数
     optimizer.step()

+    debugger.stop()  # 关闭数据dump，可继续开启数据dump，采集数据会记录在同一个step中
+    debugger.step()  # 结束数据dump，若继续开启数据dump，采集数据将记录在下一个step中
```

要点：`debugger` 尽可能在导包后的位置实例化，这样才能尽可能的 patch 更多的 API，确保采集不会缺失数据；且不得在循环内实例化；推理无 optimizer / backward 时在输出之后 stop；
`seed_all()` 是固定随机性，尽可能的调用，确保数据方便分析。
各接口的完整签名、参数说明与调用示例见 [api-reference.md](references/api-reference.md)。

### 6. 验证

1. `python3 -m py_compile <修改的文件>` 语法检查。
2. 环境允许且用户同意 → 限步短跑（`step: [0]`），确认打印 `dump.json is at ./dump_path/step*` 且对应 step 目录非空；不满足运行条件 → 明确报告仅完成静态检查。

### 7. 交付与关闭说明

交付：修改的文件与插入点、config.json 路径与最终内容、结果目录、已执行/未执行的验证项。

关闭 dump 的方式：

- 注释掉已经插入的 dump 采集代码，确保不影响业务。

分析下一步（比对、归因）不属于本技能，交付采集结果后按需转精度分析技能。

## 参考资料

- [config-json.md](references/config-json.md) — config.json 全量字段、输出目录结构、安装说明
- [api-reference.md](references/api-reference.md) — PrecisionDebugger 及 start/stop/step/save/register_custom_api/seed_all 等接口的完整签名、参数与调用示例
- [kernel-dump-l2.md](references/kernel-dump-l2.md) — L2 kernel dump 的硬约束与结果排查
- [nan-check.md](references/nan-check.md) — nan_check 两种检测模式的约束与构建要求
- [vllm-ascend.md](references/vllm-ascend.md) — vLLM 推理服务使能方式（vllm-ascend 免插桩 / 社区原生 vLLM 插桩）、字段支持差异与排查
- 官方文档：[PyTorch场景精度数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)、[config.json介绍](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/config_json_introduct.md)、[PyTorch快速入门](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/quick_start/pytorch_quick_start.md)
