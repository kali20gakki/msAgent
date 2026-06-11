# 模型准备

## 阶段说明

**模型准备阶段**是端到端自动量化与调优流程编排的第 3 阶段。在本阶段，你需要确保目标模型已被 msModelSlim 支持并完成适配，使后续量化配置调优阶段可以正常调用。

## 执行依赖项

### 子代理

由两个专用子代理承载模型分析与适配工作；主代理 **不要**在本会话中代替 subagent 完成分析或适配。

| 子代理 | 功能用途 |
|--------|----------|
| `msmodelslim-model-analysis` | 适配前分析：实现来源解析、结构/MoE/逐层加载等风险评估 |
| `msmodelslim-model-adapt` | 分析通过后：适配模板、注册、`config.ini` 与四步验证 |

## 执行流程

### 1. 检查模型是否已支持

查询用户提供的 `model_type` 是否已在 `msmodelslim/config/config.ini` 的 `[ModelAdapter]` 中注册。注意 `model_type` 不是模型权重路径中 `config.json` 里的 `model_type`，一般形如 `Qwen3-32B`、`DeepSeek-V3`。如果已注册且适配器存在，则跳过本文档的后续 subagent 委派。

### 2. 委派模型分析

若模型未注册，委派 `msmodelslim-model-analysis` subagent。调用 `task` 时 `description` **必须**包含 MSAGENT_IO 块，字段见下文。

### 3. 委派模型适配

仅当分析回传 `next_step: "model-adapt"` 时，委派 `msmodelslim-model-adapt` subagent。`next_step: "dequant"` 时先走反量化 skill；`blocked` / `need_user_input` 时停止并向用户说明（细节见 `summary` 与 `report_path`）。`description` **必须**包含 MSAGENT_IO 块，字段见下文。

### 4. 最终验证

确认以下条件均已满足后，方可进入下一阶段（量化配置调优）：

- [ ] 模型适配已完成，适配器已注册
- [ ] 模型权重文件完整可加载
- [ ] 模型可在目标设备（NPU）上正常执行前向推理

若上述任何步骤失败，须向用户明确报告原因并停止流程。

## 注意事项

- 禁止在本会话中代替 subagent 完成分析或适配代码编写
- 分析阶段判定阻塞时，不得强行进入适配或调优
- 适配完成后，按 `config.ini` 注册格式确认 `model_type` 已正确添加

## 拉起 subagent 的格式（MSAGENT_IO v1）

协议总则见 [subagent_io_protocol.md](./subagent_io_protocol.md)。本文档面向**主 Agent**：定义委派 `input` 与回传 `output` 业务字段；`commands` 见协议。完整 output 示例见各 subagent prompt。

调用 `task` 时，`description` **必须**包含一个 ` ```msagent-io v1 ` JSON 块。

### Agent: msmodelslim-model-analysis

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册名，如 `Qwen3-8B` |
| `model_path` | string | ✓ | 模型权重目录 |
| `trust_remote_code` | bool | | 默认 `true` |
| `save_path` | string | | 工作目录；分析报告写入 `{save_path}/model_analysis_report.md` |

回传 `output` 必填：`next_step`，`implementation_source`，`summary`，`report_path`；有 shell 执行时填 `commands`

委派模板：

````markdown
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-analysis",
  "input": {
    "model_type": "Qwen3-8B",
    "model_path": "/data/models/Qwen3-8B/",
    "trust_remote_code": true,
    "save_path": "/path/to/workdir/"
  }
}
```
````

### Agent: msmodelslim-model-adapt

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_type` | string | ✓ | msModelSlim 注册名 |
| `model_path` | string | ✓ | 模型权重目录 |
| `trust_remote_code` | bool | | 默认 `true` |
| `analysis_report_path` | string | ✓ | 步骤 2 产出的分析报告路径 |
| `save_path` | string | | 适配工作目录 |

回传 `output` 必填：`adapter_registered`，`verification_steps`（四步全 `passed: true` 即通过），`artifact_paths`（可选），`commands`（须含 `install` 与 `verification_step1`～`verification_step4`）

委派模板：

````markdown
```msagent-io v1
{
  "protocol": "msagent.subagent_io",
  "subagent_type": "msmodelslim-model-adapt",
  "input": {
    "model_type": "Qwen3-8B",
    "model_path": "/data/models/Qwen3-8B/",
    "trust_remote_code": true,
    "analysis_report_path": "/path/to/workdir/model_analysis_report.md",
    "save_path": "/path/to/workdir/"
  }
}
```
````
