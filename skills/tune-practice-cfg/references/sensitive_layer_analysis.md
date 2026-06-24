# 敏感层分析

## 分析工具

通过 `execute` 调用 **msmodelslim CLI**：

```bash
msmodelslim analyze layer \
    --model_type Qwen3-32B \
    --model_path ${model_path} \
    --metrics mse_layer_wise \
    --calib_dataset ${calib_dataset} \
    --topk 999 \
    --device npu \
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `model_type` | 模型类型（模型名） | 必填 |
| `model_path` | 模型路径 | 必填 |
| `metrics` | 分析指标（linear scope） | `"mse_layer_wise"` |
| `calib_dataset` | 校准数据集 | `"mix_calib.jsonl"` |
| `pattern` | 层匹配模式列表 | `"*"`（全量） |
| `topk` | 返回 Top-K 敏感层 | `999` |
| `device` | 执行设备 | `npu` |

- `model_type` 与模型 `config.json` 中的 `model_type` 并非同一概念，你应该参考 `msmodelslim/config/config.ini`，如 `Qwen3-32B` `DeepSeek-V3` 才是正确合法的 `model_type`。
- 官方文档：[敏感层分析使用指南](https://gitcode.com/Ascend/msmodelslim/blob/master/docs/zh/feature_guide/sensitive_layer_analysis/usage.md)

**注意事项**
- 敏感层分析运行时长可能较长，超过 timeout 3600s，将结果输出到 stdout 时，由于 tee 因超时而中断，造成输出丢失。你**应当**使用 nohup 或将输出重定向到文件。**务必避免**在上一个敏感层分析进程未结束时再次拉起一个敏感层分析进程。
- 当使用 mse_layer_wise 时，topk 按照模型实际层数（config.json 中 num_hidden_layers）填写。

## 支持的分析指标

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| **kurtosis** | 峰度，衡量激活分布的尾部厚度 | 通用，默认选项 |
| **std** | 标准差，衡量激活值的离散程度 | 通用 |
| **quantile** | 分位数分析，基于激活分布的分位范围 | 通用 |
| **attention_mse** | 注意力层量化前后 MSE | 需适配器实现 `AttentionMSEAnalysisInterface`；使用 `msmodelslim analyze attn --metrics mse` |
| **mse_layer_wise** | 整层量化前后 MSE | 使用 `msmodelslim analyze layer --metrics mse_layer_wise` |
| **mse_model_wise** | 整模型量化 MSE | 使用 `msmodelslim analyze layer --metrics mse_model_wise` |

## 分析结果结构

CLI 在控制台输出各层 Score。解析后写入 `{save_path}/analysis_result.yaml`：

```yaml
layer_scores:
  - name: "model.layers.0.mlp.*"
    score: 12.5
  - name: "model.layers.15.*"
    score: 8.3
  # ... 按 score 降序排列
method: "mse_layer_wise"
patterns:
  - "*"
```

- **score 越高，层越敏感**，量化时越容易造成精度损失。
- 分析结果直接用于调优策略中的 `exclude` 决策和回退层排序。

## 分析结果在调优中的使用

敏感度得分在调优任务开始时计算一次，写入 `{save_path}/analysis_result.yaml`，各轮复用。每轮根据策略从预计算的得分排序中选择回退层，无需重新调用分析命令。

选择回退层时需遵守**同分同退约束**：分析结果按 score 降序排列，分数相同的层作为一个整体（同分组），`topk` 参数选取的是前 K 个**同分组**而非前 K 个单独层。在调优过程中，同分组内的层必须同时回退或同时保留，不可拆分。

## 工具不可用时的经验规则

当 `msmodelslim analyze` 失败（非 0 exit code）或超时时，按以下步骤占位：

1. **获取总层数 N**：从 `<model_path>/config.json` 读取 `num_hidden_layers`。嵌套 config 依次查顶层、`text_config`、`language_config`、`thinker_config.text_config` 等同名字段
2. **构造经验排序**：层序上前 2-4 层 + 后 2-4 层视为更敏感，中间段相对低敏感
3. **写出结果文件**：将经验排序按上方"分析结果结构"的格式写入 `{save_path}/analysis_result.yaml`，确保后续步骤无需区分数据来源

仅作占位，**弱于**精确分析。
