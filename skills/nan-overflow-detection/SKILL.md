---
name: nan-overflow-detection
description: |
  多卡分布式训练中的 loss/gnorm 精度溢出检测与根因追溯。基于 MSProbe dump 数据，先跨 rank 定位首次出现 NaN 的源卡，再在源卡上追溯具体的溢出根因算子。
  当用户需要：(1) 多卡分布式训练场景下的 NaN/Inf 溢出检测 (2) 找出首先出现 NaN 的源卡 (3) 追溯根因计算算子 (4) loss/gnorm NaN 问题定位
  时使用此 skill。
---

# NaN 溢出检测与根因追溯

此 skill 分两步完成完整的 NaN 溢出分析：

1. **跨 rank 源卡检测** — 分析所有 rank 的通信算子，定位源卡
2. **单卡根因追溯** — 在已知源卡后，追溯该卡上产生 NaN 的计算算子

## 重要说明：执行顺序

**dump.json 中算子的出现顺序即为执行顺序**，无需根据算子名称中的数字来排序。

## 数据文件结构

```
input_path/
├── rank0/
│   ├── dump.json      # 算子输入输出数据
│   ├── construct.json # 算子调用链
│   └── stack.json     # 算子堆栈信息
├── rank1/
│   ├── dump.json
│   ├── construct.json
│   └── stack.json
└── ...
```

### 文件格式

**dump.json** — 算子数据：
```json
{
  "framework": "pytorch",
  "data": {
    "Distributed.all_reduce_0_0": {
      "input": [...],
      "input_args": [...],
      "input_kwargs": {...},
      "output": {"0": {"Max": "inf", "Min": "inf"}}
    }
  }
}
```

**construct.json** — 调用链：
```json
{
  "Torch.ops.aten.linear_0_0": "Torch.nn.functional.linear_0_0"
}
```

**stack.json** — 代码位置：
```json
{
  "Torch.ops.aten.linear_0_0": ["linear", "forward.py", 123]
}
```

## 溢出判定规则

基于 `output` 字段中的 Max/Min 值：
- **NaN**: `"Max": "nan"` 或 `"Min": "nan"`
- **Inf**: `"Max": "inf"`, `"Min": "inf"`, `"Max": "-inf"`, `"Min": "-inf"`

## 非计算算子过滤

以下算子产生 NaN/Inf 是正常现象，应被过滤：
- `torch.empty`, `torch.full`, `torch.zeros`, `torch.ones` — 内存初始化
- `Tensor.to`, `Tensor.clone`, `Tensor.detach` — 类型转换/复制
- `NPU.*_empty`, `NPU.*_full` — NPU 初始化算子

## 合法 -Infinity 过滤

以下算子合法产生 -Infinity，不算溢出：
- `Tensor.masked_fill`, `Torch.masked_fill` — MoE routing、attention mask
- `Tensor.where`, `Torch.where` — 条件操作
- `Tensor.triu`, `Torch.triu`, `Tensor.tril`, `Torch.tril` — 三角矩阵

## 分析流程

### 步骤 1：跨 rank 源卡检测

```bash
python3 "<skill_root>/scripts/cross_rank_analyzer.py" <input_path> [output_path]
```

1. 遍历所有 rank 的 dump.json，识别通信算子（`Distributed.` 开头）
2. 检查每个通信算子的输入/输出 NaN 状态
3. 分类：
   - **源卡**: 通信算子输入包含 NaN
   - **传播卡**: 输入无 NaN，输出有 NaN（通过通信传播）
   - **正常卡**: 输入输出均无 NaN
4. 输出每个通信算子的源卡列表

### 步骤 2：单卡根因追溯

获取源卡后，追溯该卡上的根因算子：

1. 保持 dump.json 中的执行顺序
2. 过滤非计算算子和合法 -Infinity
3. 遍历算子，找到第一个可能的异常节点
4. 利用 construct.json 追溯完整调用链
5. 利用 stack.json 获取代码位置
6. 如果找到的第一个可能异常节点，被排除异常可能，则继续找后续可能是异常的节点，重复进行定位

### 异常节点判定逻辑

```python
def is_anomaly(op_data):
    # 输入有异常或者输出有异常
    is_input_anomaly = check_anomaly(input_args)
    is_output_anomaly = check_anomaly(outputs)
    return is_input_anomaly or is_output_anomaly
```

## 输出

### 跨 rank 分析结果
每个通信算子的源卡、传播卡、正常卡列表。

### 单卡追溯结果
```json
{
  "rank": 168,
  "first_anomaly": {
    "op_name": "Tensor.matmul.42.forward",
    "exec_order": 0,
    "output": {...},
    "input_info": [
      {"shape": [4096, 4096], "Max": "0.001", "dtype": "float16"}
    ],
    "construct_chain": ["Tensor.matmul.42.forward", "Torch.nn.functional.linear.12.forward", ...],
    "stack_info": ["linear", "forward.py", 123]
  }
}
```
