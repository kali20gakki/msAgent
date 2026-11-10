# nan_check（NaN/溢出检测）约束与构建要求

`task: "nan_check"` 用 `PrecisionDebugger` 检测 NaN 或溢出。内容依据官方文档
[nan_check 使用示例](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_nan_check_quickstart.md)
与 [config.json 介绍](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/config_json_introduct.md)的 nan_check/check_mode 节。

## 前置约束（硬性）

- 仅支持 **PyTorch + 昇腾 NPU** 场景：`torch_npu` 必须可 import（工具校验报错原文：
  `task nan_check is only supported in NPU environments (torch_npu must be importable).`）。
- **CANN ≥ 8.5.0**。
- **`level` 必须为 `"L1"`**（报错原文：`When the task is set to nan_check, the level must be L1, but got {level}.`）。
- `data_mode` 中配置 `"input"` 时必须同时配置 `"output"`（默认 `["all"]`）。
- 默认检测所有 API，可通过 `scope` 或 `list` 缩小范围。

与常规 dump 的差别：输出的是每个 tensor 的 `is_nan` 检测标志而非统计值，用于定位 NaN/溢出的发生点。

## check_mode 两种检测方式

| | `isnan`（默认） | `slot` |
| --- | --- | --- |
| 检测内容 | 输入、输出 tensor 中的 NaN；**不检测 Inf，也不检测过程溢出** | 读取 NPU 寄存器状态，检测 NaN、Inf 及过程溢出 |
| 构建要求 | 编译 msProbe 时须包含 **`aclgraph_dump`** 模块（`python3 build.py -e include-mod=aclgraph_dump`） | 编译 msProbe 时须包含 **`nan_check`** 模块（`python3 build.py -e include-mod=nan_check`），且**运行前设置 `INF_NAN_MODE_FORCE_DISABLE=1`** |
| 采集范围 | 无需 `tensor_list`；前向输出含 NaN 时自动将该 API 的输入、输出保存为 `dump_tensor_data/` 下的 `.pt`；反向由 `grad_input` 中的 NaN 触发 | 详细采集范围由 `tensor_list`（按 API 名称子串匹配）指定，溢出时触发异常并将详细数据保存至 `extra-info`（供 [Tensor 解析工具](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/maintenref/troubleshooting/troubleshooting_0532.html)解析） |
| 其他 | 采集时不主动触发异常 | 输出 tensor 正常时仍可能检测到过程溢出；int64 运算的检测结果不保证准确 |

`is_nan` 标志语义：`1` 含 NaN、`0` 不含 NaN、`null` 无法检测。`isnan` 模式对空/整数/布尔 tensor 记 `0`，
对 FakeTensor/meta tensor 记 `null`；反向包含 `grad_input`/`grad_output` 检测结果，不记录 API 级标志，也不计算 Max/Min/Mean/Norm。

## 最小配置示例（官方）

```json
{
    "task": "nan_check",
    "dump_path": "./nan_check_dump",
    "level": "L1"
}
```

启用 slot 模式的示例片段：

```json
{
    "nan_check": {
        "check_mode": "slot",
        "tensor_list": ["sqrt"]
    }
}
```

## 插装示例（官方 quickstart 模式）

```python
import torch
import torch_npu
from msprobe.pytorch import PrecisionDebugger

torch.npu.set_device(0)
debugger = PrecisionDebugger(config_path="./config.json")
x = torch.tensor([4.0, -1.0], device="npu", requires_grad=True)

debugger.start()
y = torch.sqrt(x)
z = torch.add(y, 1.0)
loss = z.sum()
loss.backward()
debugger.stop()
debugger.step()
torch.npu.synchronize()
```

## 构建（源码编译）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 按实际 CANN 路径调整
git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe
python3 build.py -e include-mod=aclgraph_dump        # isnan 模式（tensor 自动落盘需要该模块）
# 或 slot 模式：
# python3 build.py -e include-mod=nan_check
```

构建前准备：CANN ≥ 8.5.0、配套 PyTorch 与 TorchNPU、支持 C++17 的编译器、Make、CMake ≥ 3.19.3。
构建后两种使用方式：`pip install artifacts/mindstudio_probe*.whl`；或 PYTHONPATH 源码方式
（安装 pyproject.toml 声明的 Python 依赖后 `export PYTHONPATH="$PWD/python:$PYTHONPATH"`）。

## 输出与初步解读

```
nan_check_dump/
└── step0/
    └── proc<pid>/            # 单进程；有分布式 rank 时为 rank<ID>
        ├── dump.json         # 每个 tensor 的 is_nan 标志（data 下按 API 名分组）
        ├── stack.json        # API 名列表 + 调用栈列表，用于回到调用代码
        ├── construct.json
        └── dump_tensor_data/ # 含 NaN 时自动保存的 .pt 原始 tensor
```

初步定位方法：在 dump.json 中找第一个输出 `is_nan=1` 而输入为 `0` 的 API —— NaN 大概率在那里首次产生，
再结合 stack.json 回到对应调用代码检查输入来源。

## 边界

本技能只负责把 nan_check 使能到用户代码并确认采集落盘；对 NaN 根因的进一步分析（跨 rank 溯源、根因算子定位）
转 `nan-overflow-detection` 或 `precision-debugger` 技能。
