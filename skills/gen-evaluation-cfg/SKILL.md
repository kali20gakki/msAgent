---
name: gen-evaluation-cfg
description: Generate msmodelslim evaluation YAML configuration (service_oriented + aisbench + vllm-ascend). Use when user asks for evaluation config generation.
license: Apache-2.0
metadata:
  version: 0.9.4
  domain: quantization
  framework: msmodelslim
  aliases:
    - msmodelslim-evaluation-config
    - evaluation-yaml
  trigger_intents:
    - 生成评测配置
    - 写测评yaml
    - 评测配置怎么写
  keywords:
    - evaluation config
    - aisbench
    - vllm-ascend
    - service_oriented
---

# 生成评测配置 YAML

## Overview

本 Skill 负责生成 `quant-tuning-evaluate` / `run_evaluation.py` 所需的单文件评测 YAML 配置。本 Skill 仅生成 Evaluation YAML。不要据此推断 Practice YAML 的 spec.save 应使用 `compressed_tensors`；Practice 的 save 以 tune-practice-cfg 为准，默认情况下使用 `ascendv1_saver`。

**核心功能**：
- 生成包含 `demand`（目标精度）、`evaluation`（AISBench）、`inference_engine`（vLLM-Ascend）三个模块的完整 YAML
- 确保三个模块之间的字段保持一致（模型名、服务地址、端口）

**不适用**：
- 生成其他类型的配置（如量化策略配置）
- 执行评测或分析评测结果等配置生成以外的任务

**模板参考**：[evaluation_config.example.yaml](assets/evaluation_config.example.yaml)

## 输入

执行时从上下文中提取以下信息：

| 参数 | 说明 | 缺省时默认值 |
|------|------|--------|
| 模型名称 | 量化后的模型标识符 | 从上下文获取 |
| 服务地址 | 推理服务 host | `localhost` |
| 服务端口 | 推理服务 port | `8000` |
| 设备类型 | 推理后端设备 | `ascend` |
| 设备数量 | 并行推理的卡数 | `1` |
| 目标数据集 | 要评测的数据集列表 | 从上下文获取 |
| 精度目标 | 每个数据集的目标精度百分比 | 从上下文获取 |
| 精度容差 | 允许的精度波动范围 | 从上下文获取 |

## 文件生成规则

### 文件生成步骤

1. 在工作目录生成一个 YAML 文件，包含以下结构：

```yaml
type: service_oriented

demand:
  expectations:
    - dataset: <数据集名称>
      target: <目标精度>
      tolerance: <容差>

evaluation:
  type: aisbench
  precheck: [...]  # 可选
  aisbench: { ... }
  datasets:
    <数据集名称>:
      config_name: <ais_bench 注册名>
      # ...
  host: <服务地址>
  port: <服务端口>
  served_model_name: <模型名称>

inference_engine:
  type: vllm-ascend
  served_model_name: <模型名称>
  host: <服务地址>
  port: <服务端口>
  args: { ... }
```

2. 文件生成后，执行文件检查。如果未通过检查，需要修正后重新生成，直到生成的文件满足所有要求。
3. 如果用户提供了参考的测评配置，则尽可能地按照用户的配置。如果进行了修改，则需要向用户回显该修改，给出简要的原因解释，但不必中断流程向用户确认
4. 文件生成并验证通过后，返回文件路径。

### 关键必填字段填写要求：

| 路径 | 类型 | 说明 |
|------|------|------|
| `type` | string | 必须为 `service_oriented` |
| `demand.expectations` | list | 至少包含一项 |
| `demand.expectations[].dataset` | string | 必须存在于 `evaluation.datasets` |
| `demand.expectations[].target` | float | 必须 > 0 |
| `demand.expectations[].tolerance` | float | 必须 ≥ 0 |
| `evaluation.type` | string | 必须为 `aisbench` |
| `evaluation.datasets` | dict | 必须非空 |
| `evaluation.datasets.*.config_name` | string | AISBench 注册名（查找方法见 [how_to_find_aisbench_config_name.md](references/how_to_find_aisbench_config_name.md)） |
| `evaluation.host` | string | 与 `inference_engine.host` 保持一致 |
| `evaluation.port` | int | 与 `inference_engine.port` 保持一致 |
| `evaluation.served_model_name` | string | 与 `inference_engine.served_model_name` 保持一致 |
| `inference_engine.type` | string | 必须为 `vllm-ascend` |
| `inference_engine.args.served-model-name` | string | 与 `served_model_name` 保持一致 |
| `inference_engine.args.tensor-parallel-size` | int | 与设备数量保持一致 |

### 文件检查步骤（直接检查即可，无需写检查脚本）

1. 确保所有必填字段存在且符合格式要求
2. 确保生成的 YAML 文件语法正确，可以被 YAML 解析器成功解析
3. 如果你在测浮点模型精度基线，则 `demand.expectations[].target` 和 `demand.expectations[].tolerance` **必须**都设置为 100 进行占位。
4. 确保测评配置一致性，你应确保测评浮点权重和量化权重的配置的通用参数一致，尤其是 `evaluation.aisbench`、`inference_engine.args.max-model-len`**必须**保持一致。在不一致的情况下，你应该修改当前生成的配置文件。例如先前生成了浮点的测评配置且已经测评过了，则你应该修改当前生成的量化测评配置。

## 执行约束

**绝对禁止**：
- 不得阅读任何源码文件
- 不得使用 `SemanticSearch` 进行检索

**允许**：
- 使用 `assets/evaluation_config.example.yaml` 作为模板
- 参考 `quant-tuning-evaluate` Skill 中 `device` / `device_indices` 与 `inference_engine.args.tensor-parallel-size` 的对齐要求
- 仅通过本 SKILL.md 文档和模板文件完成配置生成

## 常见错误

| 错误类型 | 描述 | 修复方法 |
|----------|------|----------|
| 数据集未统一 | `expectations` 中的 dataset 不在 `datasets` 中 | 同步添加或删除 |
| 服务地址不一致 | `evaluation` 与 `inference_engine` 的 host/port 不统一 | 统一设置 |
| 模型名不一致 | `served_model_name` 在三处不统一 | 统一设置 |
| 命名规则错误 | `args` 内使用了 snake_case 而非 kebab-case | 转换为 kebab-case（如 `served_model_name` → `served-model-name`） |
| 配置名错误 | `config_name` 与 ais_bench 注册名不匹配 | 查询正确的注册名 |
