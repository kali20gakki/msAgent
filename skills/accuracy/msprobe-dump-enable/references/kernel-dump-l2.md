# L2 kernel dump 约束与排查

`level: "L2"` 为 kernel 级精度数据采集，仅支持 PyTorch，且**仅支持昇腾 NPU**：
工具基于 TorchNPU 的 `torch_npu.npu` 接口（`init_dump`/`set_dump`/`finalize_dump`）实现，API 运行在其他设备上则不存在 kernel 级数据。
内容依据官方文档 [PyTorch场景kernel级精度数据采集](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_kernel_dump_instruct.md)
及工具代码校验逻辑。

## 硬约束（违反时工具直接报错）

| 约束 | 校验报错原文（代码 `debugger_config.py`） |
| --- | --- |
| `task` 必须为 `"tensor"` | `When level is set to L2, the task must be set to tensor.` |
| `tensor.list` 必须且只能包含 1 个 API 名称 | `When level is set to L2, the list must be configured as a list with one api name.` |
| 禁止配置 `scope` | `When level is set to L2, the scope cannot be configured.` |

- 每个 step 只支持采集 1 个 API 的 kernel 数据。
- API 名称格式：`{api_type}.{api_name}.{API调用次数}.{forward/backward}`，从一次 L1 采集的 `dump.json` 的 data key 中抄写（先跑 L1 拿到合法名称）。
- 代码行为：list 填的是 backward API 名时，工具自动追加对应 forward API 名，即**一次 backward kernel 采集会同时 dump 匹配的 forward kernel**。

## 配置示例（官方）

```json
{
    "task": "tensor",
    "dump_path": "./dump_path",
    "level": "L2",
    "rank": [],
    "step": [],
    "tensor": {
        "scope": [],
        "list": ["Functional.linear.0.backward"]
    }
}
```

## 运行提示

- 成功：`The kernel data of {api_name} is dumped successfully.`（打印后仍无数据 → 见下文排查）
- 不支持的 API：`The kernel dump does not support the {api_name} API.`

## 输出结构

```
dump_path/
├── step0/
│   ├── 20241201103000/              # 日期时间格式
│   │   ├── 0/                       # device id
│   │   │   └── {op_type}.{op_name}.{task_id}.{stream_id}.{timestamp}   # kernel 层算子数据
│   ├── kernel_config_{device_id}.json   # 工具生成的中间文件，一般无需关注
│   └── ...
```

## 采集结果为空排查（官方 FAQ）

1. 确认工具使用方式、config.json 内容、list 填写的 API 名称格式是否正确。
2. 确认该 API 是否运行在昇腾 NPU 上；运行在其他设备上不会存在 kernel 级数据。
3. 排除前两点仍无数据：使用 [TorchNPU](https://gitcode.com/Ascend/pytorch) 的 `torch_npu.npu` 接口自行采集对照
   （参见 [torch_npu.npu API 概述](https://gitcode.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu-npu/torch_npu-npu_list.md)）。
