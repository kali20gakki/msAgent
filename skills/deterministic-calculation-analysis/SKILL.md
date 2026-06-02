---
name: deterministic-calculation-analysis
description: 执行msProbe数据比对并分析比对结果，定位确定性计算问题首个输入一致输出不一致的API。
keywords: [确定性计算, msprobe, md5比对, CRC-32, 精度问题定位]
---

# 确定性计算比对分析

## 技能目标

1. **数据比对** — 调用 msProbe 工具比对两份 dump 数据，生成比对结果文件
2. **结果分析** — 分析比对结果，寻找首个输入一致输出不一致的计算 API
3. **排除误检** — 支持排除特定 API 后重新分析，直至定位根因

## 分析流程

### ⚠️ 硬性规则

1. **第 1 步（用户输入数据校验）适用于所有路径类型，不可跳过。** 即使是 db/csv_xlsx 路径，也必须先跑 `md5_dump_files_checker.py` 校验数据完整性。
2. 仅"第 2 步依赖检查（pip install）"和"第 3 步数据比对（msprobe compare/graph_visualize）"可根据路径类型跳过。

### 1. 用户输入数据校验

用户提供路径，支持三种输入类型：

| 输入类型 | 参数个数 | 说明 |
|---------|---------|------|
| dump 路径 | 2 个 | target（调试侧）和 golden（标杆侧）两份 msProbe 工具 dump 数据目录。示例：`dump_path/step0`。 |
| db 路径 | 1 个 | mix 级别比对结果文件（`.vis.db`）或包含该文件的目录。示例：`db_path` 或 `db_path/compare_*.vis.db`。 |
| csv/xlsx 路径 | 1 个 | L1 级别比对结果文件（`.csv/.xlsx`）或包含该文件的目录。示例：`csv_path` 或 `csv_path/compare_*.csv`。 |

```shell
# dump 路径（2 个参数）
python3 "<skill_root>/scripts/md5_dump_files_checker.py" <target_path> <golden_path>

# db 或 csv/xlsx 路径（1 个参数）
python3 "<skill_root>/scripts/md5_dump_files_checker.py" <db_or_csv_path>
```

- 校验通过 → 输出 `level="L1"` 或 `level="mix"`，继续下一步
- 校验不通过 → 输出异常原因，终止流程
- 打印内容包含“没有包含tensor的CRC-32校验值，无法分析确定性问题”、“缺失md5”相关内容 → 终止流程，提示用户必须使用包含 tensor 的 CRC-32 校验值的文件，采集数据的config.json配置必须指定task="statistics"和summary_mode="md5"

### 2. 依赖检查（仅 dump 路径需要）

db 和 csv/xlsx 路径已包含比对结果，无需依赖 msProbe 工具。

```shell
pip3 show mindstudio-probe || pip3 install mindstudio-probe --pre
```

### 3. 数据比对（仅 dump 路径需要）

根据第 1 步输出的 level 选择对应分支。比对等待时间最长为 600 秒。

##### 3.1 level="L1"

```shell
msprobe compare -tp <target_path> -gp <golden_path> -o <output_path>
```

##### 3.2 level="mix"

```shell
msprobe graph_visualize -tp <target_path> -gp <golden_path> -o <output_path>
```

### 4. 数据分析

根据第 1 步输出的 level 和第 1 步输入的路径类型选择对应分支。所有分支都支持 `--exclude-api` 排除后重分析。

#### 4.1 输入为 dump 路径（比对结果已生成）

##### 4.1.1 level="L1"

```shell
python3 "<skill_root>/scripts/find_first_diff_api_L1.py" <output_path>
```

分析特点：
- 无 Module 层级信息，仅有 API 级 md5 比对
- 找不到问题 API 时自动检测"状态跳变边界"，提示可能漏采的位置
- 支持 `--exclude-api "API名称"` 排除误检后重新分析

##### 4.1.2 level="mix"

```shell
python3 "<skill_root>/scripts/find_first_diff_api_mix.py" <output_path>/compare_*.vis.db
```

分析特点：
- 包含 API 和 Module 两级分析，展示父子层级链路
- 当 API 级找不到结果、但 Module 级发现异常时，提示可能漏采的 API
- 支持 `--exclude-api "API名称"` 排除误检后重新分析

#### 4.2 输入为 db 或 csv/xlsx 路径（跳过比对，直接分析）

已有比对结果文件，直接进行分析。

##### 4.2.1 level="mix"（db 路径）

```shell
python3 "<skill_root>/scripts/find_first_diff_api_mix.py" <db_path>
```

##### 4.2.2 level="L1"（csv/xlsx 路径）

```shell
python3 "<skill_root>/scripts/find_first_diff_api_L1.py" <csv_xlsx_path>
```

### 5. 输出

展示比对分析的结果并进行解读。

#### 5.1 找到问题 API

展示首个输入一致输出不一致的 API，包括其 Module 层级链路（仅 mix）和 md5 比对详情：

```
+----------------------+------------------------------------------+
|                     Rank 0                              |
+----------------------+------------------------------------------+
| 首个问题API           | NPU.npu_rms_norm.0.backward              |
+----------------------+------------------------------------------+
| API所在Module层级     | DefaultModel [Module]                    |
|                      |   → input_layernorm.RMSNorm [Module]     |
|                      |   → NPU.npu_rms_norm.0.backward [API]    |
+----------------------+------------------------------------------+
| API分析依据           | Input MD5 (全部一致)                     |
|                      | Output MD5 (不一致)                      |
|                      |   output.1: NPU=xxx vs Bench=yyy         |
+----------------------+------------------------------------------+
```

找到首个问题 API 即排查结束，后续只让用户选择是否排除该 API 重新分析。如果用户选择排除，则让用户输入要排除的 API 名称（支持前缀匹配，多个以空格分隔），内部重新执行分析流程。

#### 5.2 未找到问题 API（状态跳变）

```
+----------------------+------------------------------------------+
| 首个问题API           | 无                                       |
+----------------------+------------------------------------------+
| API分析依据           | 最后一个正常API: xxx                     |
|                      | 第一个输入不匹配API: yyy                 |
|                      | 两者之间的API可能被msprobe漏采           |
+----------------------+------------------------------------------+
```

**解读**: 未找到输入完全一致但输出不一致的 API，说明根因 API 可能被 msProbe 漏采。两个边界 API 之间的区域即为可疑范围，可调整 msProbe 采集配置后重新 dump 分析。

## 关键字段说明

### db（mix 级别）

| 字段 | 说明 |
|------|------|
| `node_name` | API/Module 名称 |
| `node_order` | 执行顺序，越小越先执行 |
| `node_type` | 0=Module, 1=API |
| `data_source` | NPU=调试侧, Bench=标杆侧 |
| `precision_index` | 0=pass, 1=error |
| `up_node / sub_nodes` | 父子层级关系 |
| `input_data / output_data` | 输入/输出数据，JSON 格式，包含 md5 值 |
| `is_distributed` | 是否为通信算子 |

### csv/xlsx（L1 级别）

| 字段 | 说明 |
|------|------|
| `NPU Name` | 调试侧 API 名称 |
| `Bench Name` | 标杆侧 API 名称 |
| `NPU MD5` | 调试侧 tensor CRC-32 值 |
| `BENCH MD5` | 标杆侧 tensor CRC-32 值 |
| `Result` | 比对结果（pass/fail） |
| `NPU Tensor Shape` | 调试侧张量形状 |
| `Bench Tensor Shape` | 标杆侧张量形状 |

## dump 路径说明

### 多卡场景

```
dump_path
└── step0
    ├── rank0
    │   ├── construct.json
    │   ├── dump.json
    │   └── stack.json
    ├── rank1
    │   ├── construct.json
    │   ├── dump.json
    │   └── stack.json
    ...
    └── rank7
        ├── construct.json
        ├── dump.json
        └── stack.json
└── step1
│   ├── ...
└── step2
...
```

### 单卡场景

```
dump_path
└── step0
    └── proc{pid}
        ├── construct.json
        ├── dump.json
        └── stack.json
└── step1
│   ├── ...
└── step2
...
```

### 比对说明
- level="mix"，调用`msprobe graph_visualize`，用户可以传入路径`dump_path`、`dump_path/step{n}`、`dump_path/step{n}/rank{n}`或者`dump_path/step{n}/proc{pid}`。
- level="L1"，调用`msprobe compare`，用户可以传入路径`dump_path/step{n}`、`dump_path/step{n}/rank{n}`或者`dump_path/step{n}/proc{pid}`，不支持比对`dump_path`。如果用户传入`dump_path`，默认比对`dump_path/step0`。
