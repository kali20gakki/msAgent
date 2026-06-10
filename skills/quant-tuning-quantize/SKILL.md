---
name: quant-tuning-quantize
description: 执行模型量化。通过 msmodelslim quant 依据 Practice YAML 对模型进行量化。
license: Apache-2.0
metadata:
  version: 0.3.0
  domain: quantization
  framework: msmodelslim
  protocol: cli
  skill_class: tool
  aliases:
    - quantizer
    - quantization-run
  trigger_intents:
    - 执行量化
    - 运行 quantization_run
    - 量化模型
  keywords:
    - msmodelslim quant
    - quantize
    - practice.yaml
---

# Skill: Quant Tuning Quantize

## Overview

**解决什么**：依据 Practice YAML 配置，调用 `msmodelslim quant` 执行模型量化。

**不解决什么**：
- 不生成/修改 Practice YAML → 见 `quant-tuning-practice-generator` Agent
- 不执行评测 → 见 `quant-tuning-evaluator` Agent
- 不做策略决策 → 见 `quantization-accuracy-tuning-orchestrator` Skill

**执行主体**：`msmodelslim quant`（`execute` 调用，以 exit code 判定成败）

---

## 协作关系

```
quantization-accuracy-tuning-orchestrator (workflow)
        │
        ▼ 调用
quant-tuning-quantize (tool)
        │
        ▼ CLI
  msmodelslim quant
        │
        ▼ 输出
  量化后的模型权重
```

---

## 执行步骤

```
┌─────────────────┐
│ 输入检查        │
│ - practice_path │
│ - model_path    │
│ - save_path     │
└────────┬────────┘
         ▼
┌─────────────────┐
│ 参数校验        │
│ 路径存在且可读   │
└────────┬────────┘
         ▼
┌─────────────────┐
│ execute:        │
│ msmodelslim quant│
└────────┬────────┘
         ▼
┌─────────────────┐
│ 结果处理        │
│ - 检查 exit code │
│ - 记录产物路径   │
│ - 错误上报       │
└─────────────────┘
```

---

## 输入参数

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `config_path` | string | ✅ | Practice YAML 路径 |
| `model_path` | string | ✅ | 原始模型路径 |
| `save_path` | string | ✅ | 量化产物保存路径；须体现调优轮次，推荐 `{workdir}/round_{N}/quantized`（如 `round_1/quantized`） |
| `model_type` | string | ✅ | 模型类型名 |
| `device` | string | ✅ | 设备类型，如 `npu:0` |
| `trust_remote_code` | bool | ❌ | 是否信任远程代码 |

---

## CLI 调用

```bash
msmodelslim quant \
  --model_path "${MODEL_PATH}" \
  --save_path "${WORKDIR}/round_1/quantized" \
  --device npu:0 \
  --model_type "${MODEL_TYPE}" \
  --config_path "${CONFIG_PATH}" \
  --trust_remote_code False
```

`save_path` 必须包含轮次与产物目录层级，便于 orchestrator 区分各轮权重，例如 `{workdir}/round_1/quantized`、`{workdir}/round_2/quantized`。

**成功判定**：exit code 为 0，且 `${SAVE_PATH}` 下出现量化权重产物。

### 错误处理

| 错误类型 | 处理 |
|----------|------|
| msmodelslim 未安装 | 按 prepare_environment.md 安装后重试 |
| 路径不存在 | 检查路径后重试或中止 |
| 量化失败 | 报 stderr 摘要，等待 orchestrator 决策 |
| 超时 | 按 Agent execution_timeout 处理，不上层续跑 |

---

## 输出结果

### 成功

向 orchestrator 回显：

```
量化完成:
- 产物路径: /workspace/output/round_1/quantized  （与 `--save_path` 一致，须含 round_N/quantized）
- 耗时: （可选）
```

### 失败

立即中止，回显命令名与 stderr 关键摘要，不续跑其它命令。

---

## 磁盘管理

- 量化产物写入 `save_path`
- 由 orchestrator 管理磁盘空间（最多保留 2 份权重）
- 本 skill 不主动清理历史产物

---

## 约束

- **错误即停**：命令失败后立即中止，不兜底续跑
- **单轮单次**：每次调用只执行一次量化
- **config_path 模式**：调优闭环使用 `--config_path`，与 `--quant_type` 互斥
- **save_path 命名**：每轮使用 `{workdir}/round_{N}/quantized`，N 为当前调优轮次（如 `round_1/quantized`）
- **device**：优先使用单卡，即以 `--device npu:0`/`--device npu:3` 这种入参形式

---

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| `practice.yaml not found` | 配置文件不存在 | 检查 `config_path` |
| `out of memory` | 设备内存不足 | 换设备 |

若错误不在上述常见错误中或者多次解决后依然未解决，依据[错误上报](references/error_handling.md)，按照错误上报格式返回至 orchestrator。

---

## 检查清单

- [ ] `config_path` 指向的 Practice YAML 已通过校验
- [ ] `device` 格式正确（如 `npu:0`, `npu:0,1,2,3`），优先使用单卡
- [ ] `save_path` 为 `{workdir}/round_{N}/quantized` 形式且磁盘空间充足
- [ ] `msmodelslim quant --help` 可正常执行
