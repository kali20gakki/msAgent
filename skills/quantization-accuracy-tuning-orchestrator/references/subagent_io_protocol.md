# 主 Agent ↔ Subagent 交互协议（MSAGENT_IO v1）

编排层通过 deepagents `task` 工具委派 subagent。主 Agent 在 `task.description`、Subagent 在最终回复中，须使用统一的 `msagent-io` 机器可读块。

各 subagent 的 `input` / `output` 字段定义见对应 reference；本文只规定**围栏格式、信封字段、职责边界**。

## 适用 subagent

自动调优流程中，以下 subagent **均须**遵守本协议：

| subagent | 职责 | 字段定义 |
|----------|------|----------|
| `msmodelslim-model-analysis` | 适配前模型分析 | [prepare_model.md](./prepare_model.md) |
| `msmodelslim-model-adapt` | 模型适配与验证 | [prepare_model.md](./prepare_model.md) |
| `quant-tuning-evaluation-generator` | 生成测评配置 | [quantization_tuning.md](./quantization_tuning.md) |
| `quant-tuning-practice-generator` | 生成 Practice 配置 | [quantization_tuning.md](./quantization_tuning.md) |
| `quant-tuning-quantizer` | 执行量化 | [quantization_tuning.md](./quantization_tuning.md) |
| `quant-tuning-evaluator` | 执行精度评测 | [quantization_tuning.md](./quantization_tuning.md) |

## 职责边界

| 角色 | 写什么 | 读什么 |
|------|--------|--------|
| **主 Agent** | `task.description` 中的 msagent-io 块（含 `input`） | Subagent 回传 msagent-io 块中的 `output` / `error` |
| **Subagent** | 最终回复中的 msagent-io 块（含 `status` + `output` 或 `error`） | 主 Agent 委派 msagent-io 块中的 `input` |

主 Agent **不得**伪造 Subagent 的 `output`；汇总结论须来自 Subagent 回传的 msagent-io 块。

## 消息结构（委派与回传统一）

每条消息由两部分组成：

- **块外**（可选）：≤3 行纯文本摘要
- **块内**（必选）：有且仅有一个 ` ```msagent-io v1 ` 围栏块

完整形态参考如下：

````markdown
<可选摘要，≤3 行>

```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "<subagent 名称>",
  ...
}
```
````

约束：

1. 禁止第二个 msagent-io 块或重复 JSON
2. 块外**禁止**：长参数列表、SKILL 全文、完整 YAML/日志正文、重复 `input` 已有字段的执行细节
3. JSON 须可解析；`protocol` 固定为 `msagent.subagent_io`
4. 委派块**不含** `status` / `output` / `error`；回传块**不含** `input`

### 委派信封（主 Agent → task.description）

块外摘要规则见上文。以下**仅展示块内** msagent-io 围栏内容：

```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "<与 task 参数 subagent_type 一致>",
  "input": { }
}
```

`input` 字段见上表对应 reference 中的 subagent 字段表。

### 回传信封（Subagent → 最终回复）

块外摘要规则见上文。以下**仅展示块内** msagent-io 围栏内容：

成功时（`status: "ok"` 时填 `output`，**不填** `error`）：

```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "<本 subagent 名称>",
  "status": "ok",
  "output": { }
}
```

失败时（`status: "failed"` 时填 `error`，**不填** `output`）：

```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "<本 subagent 名称>",
  "status": "failed",
  "error": {
    "code": "UNKNOWN_ERROR",
    "message": "简短错误描述"
  }
}
```

`output` / `error` 内具体字段见对应 reference，不在此重复。

### `commands` 字段（回传 `output` 内，涉及 CLI/脚本时必填）

当 subagent 通过 `execute` 运行 shell 命令或脚本时，须在 `output.commands` 中列出**实际执行**（或等价、可复现）的命令，供审计日志追溯。

每项结构：

| 字段 | 类型 | 说明 |
|------|------|------|
| `name` | string | 步骤标识，如 `quantize`、`sensitive_layer_analysis` |
| `command` | string | 完整 shell 命令；未执行时可省略 |
| `skipped` | bool | 未执行时为 `true` |
| `reason` | string | 跳过原因（可选） |

各 subagent 要求的 `name` 见 `quantization_tuning.md` 对应小节。

## 反例

| 反例 | 问题 |
|------|------|
| 整段自然语言参数列表、无 msagent-io 块 | 无法解析 |
| 块内缺少必填字段 | 委派不合规，须修正后重委派 |
| 块外重复 `input` 路径/设备说明或写执行步骤 | 违反「块外 ≤3 行摘要」 |
| 回传只有 Markdown 表格或纯自然语言（如「全部任务完成。」） | 无 msagent-io 块，不得作为有效结论 |
| 在 `output` 中粘贴完整 YAML / 日志正文 | 应只回传路径等结构化字段 |
| 回传同时含 `output` 与 `error`，或 `status` 与内容不匹配 | 信封字段冲突 |

quant-tuning 四类完整示例见 `quantization_tuning.md` 各 subagent 小节；msmodelslim 两类见 `prepare_model.md`。

## 回传检查（主 Agent）

`task` 返回 Subagent 原文，不附带校验标志。须从回传中解析 msagent-io 块：

- `status: "ok"` → 读 `output`
- `status: "failed"` → 读 `error`
- 无块或无法解析 → 重试或判该步失败，**不得**用自然语言摘要代替
