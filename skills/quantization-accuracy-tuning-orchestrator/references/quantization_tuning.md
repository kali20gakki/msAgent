# 量化配置调优（阶段）

## 阶段说明

**量化配置调优阶段**是**端到端自动量化与调优流程**编排的第4阶段。这个阶段将循环迭代生成多组量化配置，进行测评，并从中选择最优策略用作结果。

## 执行依赖项

### Agent

| Plugin | Agent name | 功能用途 |
| --- | --- | --- |
| `modelslim-agent` | `quant-tuning-evaluation-generator` | 生成测评配置 |
| `modelslim-agent` | `quant-tuning-practice-generator` | 生成量化配置 |
| `modelslim-agent` | `quant-tuning-quantizer` | 依据量化配置进行量化 |
| `modelslim-agent` | `quant-tuning-evaluator` | 对量化后的模型进行测评 |

### Scripts（编排层）

| Skill | 脚本 | 功能用途 |
| --- | --- | --- |
| `quantization-accuracy-tuning-orchestrator` | `scripts/history_clear.py` | 清空当前调优历史，每次调优任务开始时用于初始化 |
| `quantization-accuracy-tuning-orchestrator` | `scripts/history_append.py` | 记录一条调优历史 |
| `quantization-accuracy-tuning-orchestrator` | `scripts/accuracy_append.py` | 将 practice + evaluate 评测结果写入精度缓存 |
| `quantization-accuracy-tuning-orchestrator` | `scripts/accuracy_lookup.py` | 查询精度缓存，避免重复计算 |

### CLI / 脚本（子 Skill，由 subagent 调用）

| Skill | 命令 / 脚本 | 功能用途 |
| --- | --- | --- |
| `tune-practice-cfg` | `msmodelslim analyze linear ...` | 敏感层分析 |
| `tune-practice-cfg` | `scripts/validate_practice_yaml.py` | Practice YAML 校验 |
| `quant-tuning-quantize` | `msmodelslim quant --config_path ...` | 执行量化 |
| `quant-tuning-evaluate` | `scripts/run_evaluation.py` | 执行评测 |

## 详细步骤

完整的调优循环流程图如下。在执行该流程前和流程中，须遵守后续小节列出的约束规则。

```plaintext
             ┌───────────────┐
             │ Agent:        │
             │ evaluation-   │
             │ generator     │
             │ (生成测评配置) │
             └───────┬───────┘
                     ▼
             (>>> 循环开始 <<<) ◄────────────┐
                     ▼                      │
            ┌────────────────────┐          │
            │ 输出:              │          │
            │ "第X次调优循环"     │          │
            └────────┬───────────┘          │
                     ▼                      │
            ┌─────────────────┐             │
            │ Script:         │             │
            │ history_clear   │             │
            │ (初始化历史)     │             │
            └────────┬────────┘             │
                     ▼                      │
            ┌─────────────────┐             │
            │ Agent:          │             │
            │ practice-       │             │
            │ generator       │             │
            │ (生成量化配置)   │             │
            └────────┬────────┘             │
                     ▼                      │
            ┌─────────────────┐             │
            │ Script:         │             │
            │ accuracy_lookup │             │
            │ (查询精度缓存)   │             │
            └────────┬────────┘             │
               ┌─ 缓存命中? ─┐               │
               │            │               │
         Yes   │        No  ▼               │
(跳过量化/评估) │    ┌───────────────┐       │
               │    │ Agent:        │       │
               │    │ quant-tuning- │       │
               │    │ quantizer     │       │
               │    │ (执行量化)     │       │
               │    └────────┬──────┘       │
               │             ▼              │
               │    ┌───────────────┐       │
               │    │ Agent:        │       │
               │    │ quant-tuning- │       │
               │    │ evaluator     │       │
               │    │ (模型测评)     │       │
               │    └────────┬──────┘       │
               │             ▼              │
               │    ┌─────────────────┐     │
               │    │ Script:       │     │
               │    │ accuracy_append │     │
               │    │ (写入精度缓存)   │     │
               │    └────────┬────────┘     │
               └─────┬───────┘              │
                     ▼                      │
            ┌────────────────┐              │
            │ Script:        │              │
            │ history_append │              │
            │ (记录调优历史)  │              │
            └────────┬───────┘              │
                     ▼                      │
            ┌─────────────────┐             │
            │ 检查退出条件:    │             │
            │ 1. 精度达标?     │             │
            │ 2. 达到最大次数? │             │
            └────────┬────────┘             │
         ┌───── 满足退出条件? ──────┐        │
         │                         │        │
      No ▼                     Yes ▼        │
(继续循环)                      [ 任务结束 ] │
         └──────────────────────────────────┘
```

## 调优约束规则
 	 
以下规则对上方流程图中的各个环节施加额外约束，执行调优时必须一并遵守。

### FP baseline 获取（循环前强制前置）

若用户未提供 FP baseline：
1. 生成浮点模型的评测配置（不含 quantization 相关参数）
2. 对浮点模型执行评测，记录 baseline 精度
3. 将 baseline 值写入 evaluate_config.yaml 的 target 字段（target = FP baseline - tolerance, 其中 tolerance 为用户容差）
4. 然后才进入循环

注意：target 不要随意计算。参照以下示例进行计算：
（1）如果用户描述"不低于浮点精度超过 1%"，则 target = FP baseline - 1%。
（2）如果描述"某数据集相比浮点模型多错一道题"，则需要通过计算该数据集一道题目对应的精度数值，然后 target = FP baseline - 一题对应的数值。

### standing_high 摸高二分搜索

采用 standing_high 策略时，调优循环不是"达标即停"，而是二分搜索最少回退层数：

1. **Round 1**：0层回退（下界）
2. **Round 2**：全部敏感层回退（上界）
3. **后续轮**：取上下界中位数，保持 gate_proj/up_proj 配对完整性（见下方约束）
4. **终止条件**：上界与下界不可再二分（如差距 ≤ 配对粒度）
5. **最终结果**：上界（最少回退且达标的配置）为最优

退出条件（判断优先级）：
- 二分收敛（上界与下界不可再分）→ 输出上界为最优
- 达到最大迭代次数 → 输出历史最优达标配置
- **"某轮达标"不是退出条件**，达标只标记上界

### 二分搜索约束（exclude 列表截断规则）

生成 Practice YAML 的 exclude 列表时，必须遵守以下规则：
- gate_proj 和 up_proj 在 vllm 中融合为 gate_up_proj，**必须同退同量化**
- 截断 exclude 列表时，不能在 gate_proj/up_proj 配对中间截断
- 若中位数截断点落在配对中间，向上取整到配对末尾

## 拉起 subagent 的格式（MSAGENT_IO v1）

协议总则见 [subagent_io_protocol.md](./subagent_io_protocol.md)。本文档面向**主 Agent**，重点定义委派 `input`；回传 `output` 一行简述主 Agent 需读取的业务字段。

调用 `task` 时，`description` **必须**包含一个 ` ```msagent-io v1 ` JSON 块；`input` 按下表填写。收到回传后从 `output` 读取下表字段驱动流程，完整 output 示例见各 subagent prompt。

编排脚本（`accuracy_lookup`、`history_clear`、`history_append`、`accuracy_append`）由主 Agent `execute`，**不得** `task` 委派。

### Agent: quant-tuning-practice-generator

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | 模型类型名 |
| `model_path` | string | ✓ | 模型路径 |
| `save_path` | string | ✓ | 工作目录，Practice YAML 写入此目录 |
| `device` | string | ✓ | 如 `npu:2,3` |
| `strategy` | string | ✓ | `standing_high` 或 `standing_high_with_experience` |
| `max_iterations` | int | ✓ | 最大迭代轮次 |
| `round` | int | ✓ | 当前调优轮次 |
| `prev_result` | object\|null | | 上轮评测结果，首轮 `null` |
| `anchor_practice` | string\|null | | 已知最优且达标的 Practice 路径 |

回传 `output` 必填：`practice_path`，`validation: { ok, valid, errors }`，`commands`（须含 `sensitive_layer_analysis` 与 `validate_practice_yaml`；跳过敏感层分析时前者 `skipped: true`）

委派模板：

````markdown
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-practice-generator",
  "input": {
    "model_type": "",
    "model_path": "",
    "save_path": "",
    "device": "",
    "strategy": "standing_high",
    "max_iterations": 10,
    "round": 1,
    "prev_result": null,
    "anchor_practice": null
  }
}
```
````

### Agent: quant-tuning-evaluation-generator

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_name` | string | ✓ | 量化后的模型标识符 |
| `save_path` | string | ✓ | 工作目录 |
| `datasets` | object[] | ✓ | 评测数据集列表，每项见下表 |
| `service_host` | string | | 默认 `localhost` |
| `service_port` | int | | 默认 `8000` |
| `device_type` | string | | 默认 `ascend` |
| `device_count` | int | | 默认 `1` |

`datasets[]` 每项：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `name` | string | ✓ | 数据集名称（如 `gpqa`） |
| `config_name` | string | ✓ | AISBench 注册配置名 |
| `target` | number | ✓ | 目标精度（百分比） |
| `tolerance` | number | | 容差，默认 `0` |

回传 `output` 必填：`evaluate_config_path`；若执行了 YAML 校验，建议填 `commands`（`name: validate_yaml`）

委派模板：

````markdown
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-evaluation-generator",
  "input": {
    "model_name": "",
    "save_path": "",
    "datasets": [
      {
        "name": "gpqa",
        "config_name": "gpqa_gen_0_shot_cot_str",
        "target": 79.0,
        "tolerance": 1.0
      }
    ],
    "service_host": "localhost",
    "service_port": 8000,
    "device_type": "ascend",
    "device_count": 1
  }
}
```
````

### Agent: quant-tuning-quantizer

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config_path` | string | ✓ | Practice YAML 路径 |
| `model_path` | string | ✓ | 原始模型路径 |
| `save_path` | string | ✓ | 量化产物目录，如 `.../round_N/quantized` |
| `device` | string | ✓ | 如 `npu:2,3` |
| `trust_remote_code` | bool | | 默认 `true` |
| `round` | int | | 建议填写当前轮次 |

回传 `output` 必填：`success`，`quantized_path`，`exit_code`，`commands`（含 `name: quantize` 的量化命令）

委派模板：

````markdown
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-quantizer",
  "input": {
    "config_path": "",
    "model_path": "",
    "save_path": "",
    "device": "",
    "trust_remote_code": true,
    "round": 1
  }
}
```
````

### Agent: quant-tuning-evaluator

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `config_path` | string | ✓ | Evaluation YAML 路径 |
| `quant_model_path` | string | ✓ | 量化模型路径 |
| `save_path` | string | ✓ | 工作目录 |
| `device` | string | ✓ | 如 `npu` |
| `device_indices` | int[] | ✓ | 如 `[0, 1]` |
| `evaluate_id` | string | | 评测标识 |
| `round` | int | | 建议填写当前轮次 |

回传 `output` 必填：`overall_passed`，`datasets: [{ name, score, target, passed }]`，`commands`（须含 `inference_service` 与 `evaluation`）

委派模板：

````markdown
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "quant-tuning-evaluator",
  "input": {
    "config_path": "",
    "quant_model_path": "",
    "save_path": "",
    "device": "npu",
    "device_indices": [0, 1],
    "evaluate_id": "",
    "round": 1
  }
}
```
````
