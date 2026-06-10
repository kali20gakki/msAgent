# 用户输入

**Load when:** 尚无合法 practice YAML，需从自然语言抽取或确认必填项、精度表述、数据集名；进入主循环前的输入对齐。

## Overview

执行全自动量化调优前，Agent 从自然语言中**自动提取**必要参数；用户无需手写完整 YAML。落地后的 practice/评测仍须走本 Skill 的 **YAML 校验与 MCP 编排**（与主 `SKILL.md` 一致）。

**示例：**

> 帮我把 `./models/Llama-3-8B-Instruct` 量化到 NPU 设备 0，精度损失控制在 2% 以内，结果保存到 `./output/llama3-8b-quant`

> Qwen2-7B 在 HuggingFace 上，**W8A8** 量化，NPU 0，`gsm8k` 不低于 81%

## 参数提取

### 必填（须提取或确认）

| 参数 | 说明 |
| --- | --- |
| 模型路径 | 本地路径或 HuggingFace 仓库名 |
| 保存路径 | 产物与过程输出目录 |
| **量化方案 / 需求** | 用户应说明或确认，如 **W8A8**、W4A8、是否**动态**激活量化、是否量化 KV 等；口语「8bit 权重 + 8bit 激活」等须映射成方案并在回显中写清。**未说明时**由 agent 提议默认（常见 W8A8）并**经用户确认**后再生成 practice |
| 量化设备类型 | NPU / CUDA / CPU |
| 设备索引 | 卡号或多卡 |
| 精度需求 | 数据集、目标、容差等 |
| `trust_remote_code` | 用户知悉风险后确认 |

### 可推导（默认须回显确认）

| 参数 | 规则 |
|------|------|
| 调优策略类型 | 用户指定优先；**不在此写死某一算法为全局默认**；由对话、`msmodelslim/skills` 命中的策略 Skill 或项目约定决定 |
| 调优历史记录路径 | 用户指定优先，否则 `{save_path}/tuning_history` 等可审计路径；**新任务**与**续跑**须与用户确认是否清空或沿用 |
| **权重量化位数**（w_bit） | "8bit"/"int8"→8，"4bit"/"int4"→4；**未提及时常用 8**（与 `auto_config_generation.md` 位数表一致） |
| **激活量化位数**（a_bit） | 常与 w_bit 一致，除非用户指定；组合见 `auto_config_generation.md` |
| **是否量化 KV 缓存** | 用户提到则开启；默认 false |
| **校准数据集** | 未指定可用项目默认混合校准集；须符合 `lab_calib` 等治理约束 |

**与必填「量化方案」对齐**：用户已说明 **W8A8**、**W4A8** 等时，上表 **w_bit / a_bit**、KV、动态量化相关推导须与之**一致**，不得用另一套默认悄悄覆盖。

## 精度需求提取

### 表达方式 1：相对精度损失限制（常用）

> 「精度损失不超过 2%」 → 容差与基准精度结合；若无基准精度须引导评估或放宽假设（见 `auto_config_generation.md`）。

### 表达方式 2：绝对精度目标

> 「在 gsm8k 上精度要达到 83%」 → dataset=gsm8k, target=83 等，映射到评测配置。

### 表达方式 3：简洁描述

> 「尽量保持精度」 / 「精度要求不高」 → 采用明确默认容差并在回显中写明。

### 支持的数据集（示例）

| 数据集名称 | 默认 AISbench 配置名 | 说明 |
|------------|---------------------|------|
| `gsm8k` | `gsm8k_gen_0_shot_cot_str` | 数学推理 |
| `aime25` | `aime2025_gen_0_shot_chat_prompt` | 高阶数学 |
| `bfcl-simple` | `BFCL_gen_simple` | 工具调用 |
| `mmlu` | `mmlu_gen_0_shot` | 知识问答 |

其它数据集名称由 agent 按项目内测评配置映射。

## 调优策略类型

**策略须由用户指定、或由命中的策略 Skill / 项目文档给出**；本参考文件不列出「唯一合法默认算法」。未确定策略前，agent 应在 `msmodelslim/skills` 下用各 **`SKILL.md` 的 frontmatter（`description`、`skill_class` 等）做匹配**，再**按需深读**候选条目并与用户对齐（不必通读所有 Skill）。

## 处理流程

1. **提取参数**：从自然语言中提取可确定项  
2. **补全缺失**：必填缺失时渐进式提问  
3. **推导配置**：按 `auto_config_generation.md` 形成可校验的 practice / 评测意图  
4. **确认参数**：**回显**全部关键项（含默认值）并经用户确认后再进入调优循环  
5. **执行调优**：与主 SKILL 一致：MCP 校验 → 缓存 → 量化 → 评测 → history  

## 自然语言输入示例（节选）

### 示例 1：最简输入

```
帮我量化 ./models/Qwen2-7B-Chat，结果保存到 ./output/qwen2-7b-npu，要求精度损失不超过 2%
```

**自动提取示例**（具体默认值以回显确认为准）：模型路径、保存路径、设备、精度需求、w_bit/a_bit 等。**策略类型**须由会话与 `msmodelslim/skills` 中**按需选中的 `SKILL.md`** 确定，**不**在提取阶段写死摸高或其它单一算法。

### 示例 2：更详细的需求

```
量化 meta-llama/Llama-3-8B-Instruct 到 NPU 设备 0，做 8bit 量化，在 gsm8k 精度至少 83%，aime25 至少 50%，结果保存到 ./output/llama3-8b-int8
```

多数据集目标分别解析；HuggingFace 模型 `trust_remote_code` 须经用户确认。

## 评测配置约束规则
 	 
以下规则在生成或修改评测配置时必须遵守。

### baseline 获取规则（强制）

若用户未提供 FP baseline 精度：
→ **必须**先对浮点模型执行评测获取 baseline，禁止猜测或使用默认值
→ baseline 值必须写入 evaluate_config.yaml 的 target 字段后才能开始调优循环

### 评测参数一致性规则（强制）

生成或修改评测配置时，以下参数必须保持一致，修改任一项须同步修改其余项：
- `device_indices`（实际使用的卡号列表）→ 决定卡数 N
- `tensor-parallel-size`（YAML 中）→ 必须等于 N
- `data-parallel-size`（YAML 中）→ 默认为 1

不可只改 device_indices 而不改 tensor-parallel-size，反之亦然。
