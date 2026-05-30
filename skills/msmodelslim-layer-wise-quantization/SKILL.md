---
name: msmodelslim-layer-wise-quantization
description: 为 msModelSlim 适配器实现逐层量化（按层加载/懒加载）能力。仅在用户明确要求逐层量化或基础适配因 CPU 内存不足无法全量加载权重时使用。该特性为高阶可选项，不是基础适配必需项。
---

# msModelSlim 逐层量化 Skill

用于在基础适配完成后，为超大模型增加“逐层量化（按层加载）”能力，以降低 CPU 内存峰值。

## 触发条件

- 用户明确要求实现逐层量化/逐层加载/懒加载/按层加载。
- 或基础适配已完成，但 CPU 内存无法全量加载权重。

## 必须前置条件

- 已完成基础适配器开发（5 个基础接口可用）。
- 已完成四步验证流程（生成测试模型 -> 全回退量化 -> 权重一致性 -> 实际量化验证）。
- 模型目录存在 `model.safetensors.index.json`，且可按层定位权重。

## 约束声明（必须告知用户）

- 逐层量化是高阶可选特性，不是基础适配必需项。
- 该方案用于“内存受限时继续开发”，不保证一定适配成功。
- 常见失败点：层构造参数不一致、权重键名不匹配、自定义模块副作用、MTP/MoE 特殊路径未覆盖。

## 最小实现步骤

1. 增加权重索引与按需读取能力：
   - `get_weight_map`（缓存 `model.safetensors.index.json`）
   - `get_state_dict(module, prefix)`（只读取当前层所需权重）
2. 实现按层实例化与加载：
   - `load_decoder_if_not_exist(model, name, idx)`
   - 缺层时按模板层构造并加载该层 state_dict。
3. 实现逐层遍历器：
   - `generate_decoder_layer(model)` 按 `num_hidden_layers` 逐层 `yield`。
4. 在 `generate_model_visit` 与 `generate_model_forward` 统一使用逐层遍历器，保持严格同序。

## 最小验证

- `generate_decoder_layer` 能完整遍历全部层。
- `generate_model_visit` 与 `generate_model_forward` 无层序错位。
- 抽样 1-2 层前向结果 shape 连续、无异常。

## 参考资料

- [逐层量化工作流](references/workflow.md)
- [逐层量化实现指南](references/implementation_guide.md)
- [逐层适配示例代码](references/layerwise_adapter_example.py)

## 结束输出要求

- 明确告知用户是否满足触发条件。
- 若满足，给出接入点、最小改造范围与验证结果。
- 若不满足，建议先完成基础适配与四步验证，不提前进入逐层量化。
