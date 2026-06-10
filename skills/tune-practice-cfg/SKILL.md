---
name: tune-practice-cfg
description: Use when 量化调优闭环中需要生成或修改一轮调优所需的 Practice YAML，包括敏感层分析、策略决策、写出 YAML 文件和校验。
license: Apache-2.0
metadata:
  version: 0.1.0
  domain: quantization
  framework: msmodelslim
  protocol: cli
  skill_class: tool
  aliases:
    - practice-cfg
    - tune-practice
  trigger_intents:
    - 生成量化配置
    - 修改 practice
    - 敏感层分析并生成配置
  keywords:
    - msmodelslim analyze
    - yaml_validation_validate
    - exclude
    - 敏感层
    - practice yaml
---

# 量化配置生成

## Overview

在量化调优闭环中，根据敏感层分析和上轮评测结果，**生成或修改**一轮调优所需的 Practice YAML，确保其通过校验后交付给后续 model-quantize 执行。

| 负责 | 不负责（编排层） |
|------|------------------|
| 敏感层分析（`msmodelslim analyze`） | 缓存查询（`accuracy_lookup`） |
| 策略生成/修改 Practice YAML | 量化执行（`msmodelslim quant`） |
| YAML 校验（`validate_practice_yaml.py`） | 评测执行（`run_evaluation.py`） |
| | 缓存写入（`accuracy_append`） |
| | 历史记录（`history_append`） |
| | 策略终止决策 |

## 接口

**输入**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `model_type` | `str` | 模型类型名 |
| `model_path` | `str` | 模型路径 |
| `save_path` | `str` | 工作目录，Practice YAML 写入此目录下 |
| `device` | `str` | 分析设备，如 `"npu"`、`"npu:0"`、`"gpu:0,1"` |
| `strategy` | `str` | 调优策略：`"standing_high"` 或 `"standing_high_with_experience"` |
| `max_iterations` | `int` | 最大迭代轮次，由用户指定 |
| `prev_result` | `dict \| None` | 上轮评测结果（EvaluateResult 结构），首轮为 `None` |
| `anchor_practice` | `str \| None` | 当前已知最优且达标的 Practice YAML 路径（锚点） |

**产出**：`practice_path`（合法的 Practice YAML 文件路径）

**工具**：`msmodelslim analyze`（敏感层分析）、`scripts/validate_practice_yaml.py`（校验）

## 执行步骤

### 步骤总览

```
        ┌─────────────────────┐
        │   ① 敏感层分析       │  ← 调优任务开始前执行一次
        │ (msmodelslim analyze)│
        └──────────┬──────────┘
                   │ 敏感度得分文件（各轮复用）
                   ▼
     (>>> 每轮循环 <<<)  ◄──────────────────┐
                   ▼                        │
        ┌─────────────────────────────────┐  │
        │ ② 根据策略选择回退层              │  │
        │   + 生成/修改 Practice YAML      │  │
        └──────────┬──────────────────────┘  │
                   │                          │
                   ▼                          │
        ┌─────────────────────┐              │
        │ ③ 校验 Practice YAML │              │
        │ (validate_practice_  │              │
        │  yaml)               │              │
        └──────────┬──────────┘              │
                   │ practice_path            │
                   ▼                          │
         后续：model-quantize ──→ 下一轮 ─────┘
```

### ① 敏感层分析

通过 `execute` 调用 **msmodelslim CLI** 获取当前模型各线性层的量化敏感度得分（score 越高越敏感）。**每个调优任务调用一次**，后续各轮复用该得分结果。注意默认优先使用 **mse_layer_wise** 指标。

若 `{save_path}/analysis_result.yaml` 已存在，跳过本步骤，直接复用已有得分。

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_layer_wise \
    --calib_dataset ${calib_dataset} \
    --topk 999 \
    --device npu:0 \
  2>&1 | tee "${SAVE_PATH}/analysis_console.log"
```

**成功判定**：命令 exit code 为 0。从控制台输出解析各层 `Score`，写入 `{save_path}/analysis_result.yaml`（格式见 [敏感层分析](references/sensitive_layer_analysis.md)）。

若命令失败或超时，可用经验规则占位，仍需产出相同格式的敏感度得分文件供步骤 ② 使用。仅作占位，**弱于**精确分析。

> 完整参数说明、metrics 选项与分析结果结构见 [敏感层分析](references/sensitive_layer_analysis.md)。

---

### ② 策略生成/修改 Practice 并写出 YAML 文件

**目的**：根据预计算的敏感度得分和当前轮次的策略需要，选择本轮回退层并确定离群值抑制策略，构造完整的 Practice YAML 内容，并**写入磁盘文件**。

**输入**：
- 敏感度得分文件 `{save_path}/analysis_result.yaml`（步骤 ① 产出，各轮复用）
- 上轮评测结果 `prev_result`（首轮为 `None`）
- 当前已知最优且达标的配置（锚点）

**具体动作**：

1. **确定本轮改动**（一次只改一两处字段，从预计算的敏感度得分中选择回退层，遵守同分同退约束）
2. **构造完整的 Practice YAML 内容**（对齐 `modelslim_v1` 格式，详见 [量化配置格式](references/practice_yaml_format.md)）
3. **写出文件**：将 YAML 内容写入 `{save_path}/practice_round_{N}.yaml`（N 为当前轮次），得到 `practice_path`

| 改动项 | 说明 | 对应 YAML 位置 |
|--------|------|----------------|
| 调整 `exclude` | 增减回退层 | `spec.process[].exclude` |
| 替换离群值抑制 | `iter_smooth` ↔ `flex_smooth_quant` ↔ `flex_awq_ssz` | `spec.process[].type` |
| 调整抑制强度 | 如 `flex_awq_ssz` 的 `step`、`enable_subgraph_type` | `spec.process[].qconfig.ext` |

**修改粒度**：
- **一次只改一两处字段**，避免多因素同时变化导致无法归因

**exclude 设计原则**：
- 优先覆盖敏感层排序中 **score 最高的层**
- **同分同退**：敏感度分数相同的层必须作为一个整体同时回退或同时保留
- 回退位置经验优先级：靠近输入的前若干层 > 靠近输出的后若干层 > 语义敏感子模块（部分 MLP / attention 层）
- 回退级别按层组离散化（如前 2 层、前 4 层、前 4 + 后 4 层……），便于二分搜索

**离群值抑制叠加原则**：
- 先上单一、简单的抑制（如仅 `iter_smooth`）
- 确认瓶颈后再考虑更强或组合策略
- **二分阶段抑制组合固定，只动回退刻度**；摸高阶段才允许切换抑制

> 调优策略由入参 `strategy` 决定。`"standing_high"` 详见 [standing_high 策略](references/strategy_standing_high.md)；`"standing_high_with_experience"` 详见 [standing_high_with_experience 策略](references/strategy_standing_high_with_experience.md)。

**始终保留锚点**：掉精度时可回滚到上一已知达标配置。

---

### ③ 校验 Practice YAML

**脚本调用**：

```bash
python skills/tune-practice-cfg/scripts/validate_practice_yaml.py --practice-path /path/to/practice.yaml
```

**返回**：

```json
{
    "ok": true,
    "valid": true,
    "errors": []
}
```

**校验内容**：
1. **YAML 语法**：能否正常解析
2. **Schema 校验**：是否可被 `PracticeConfig.model_validate` 通过（字段名、类型、必填项）
3. **业务规则**：如 `label` 必须是 dict 而非字符串、`type` 与字段是否匹配

**错误处理**：

| 错误类型 | 说明 | 动作 |
|----------|------|------|
| `parse_error` | YAML 语法错误 | 修正语法后重试 |
| `schema_error` | 字段缺失/类型不对 | 修正字段后重试 |
| `business_rule_error` | 业务逻辑违规 | 按提示修正后重试 |

`valid=false` 时**不可继续后续步骤**，必须修正后重新校验。

> YAML 字段名、类型、必填项等 schema 细节见 [量化配置格式](references/practice_yaml_format.md)。

---

## 产出

`practice_path`（合法的 Practice YAML 文件路径），交付给编排层进行缓存查询后，传递给 model-quantize 执行量化。

## 约束汇总

| 约束 | 说明 |
|------|------|
| ① 在首轮前调用一次 | 敏感度得分每个调优任务计算一次，各轮复用 |
| 一次只改一两处 | exclude 或离群值抑制，避免多因素同时变化 |
| 保留锚点 | 始终保留一份当前已知最优且达标的配置，掉精度可回滚 |
| 校验必过 | `valid=false` 时不可继续，必须修正后重新校验 |

## 常见错误

- 回退层选择时拆分同分同退组（应整体回退或整体保留）
- 一次同时改 exclude + 抑制策略 + 校准集，无法归因
- `metadata.label` 写成字符串而非 dict
- `type` 与字段不匹配（如 `flex_awq_ssz` 缺少 `qconfig`），参见 [量化配置格式](references/practice_yaml_format.md)
- `valid=false` 仍继续后续步骤
- 命令行参数 `--device` 未使用 `npu:0` 这种格式，错误地使用了 `DeviceType.NPU`
