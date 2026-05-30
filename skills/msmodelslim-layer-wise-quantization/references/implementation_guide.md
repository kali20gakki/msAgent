# 逐层量化实现指南

## 关键实现点

### 1) 权重索引缓存

- 从 `model.safetensors.index.json` 读取 `weight_map`。
- 对 `weight_map` 做缓存，避免重复解析。

### 2) 单层按需加载

- 实现 `get_state_dict(module, prefix)`。
- 按 `prefix` 过滤当前层参数，只读取当前层所需 safetensors 切片。

### 3) 缺层按需构造

- 实现 `load_decoder_if_not_exist(model, name, idx)`。
- 当目标层不存在时，用模板层构造 `idx` 层并加载该层 state_dict。
- 不要硬编码层构造参数，按目标模型真实 block 构造签名适配。

### 4) 统一层遍历器

- 实现 `generate_decoder_layer(model)`。
- 在 `generate_model_visit` 与 `generate_model_forward` 统一调用该遍历器，保持严格同序。

## 典型风险点

- 层构造参数不兼容（例如无 `layer_idx`）。
- state_dict 键名前缀与模型层路径不一致。
- 远程代码中的自定义模块在懒加载时有副作用。
- MTP/MoE 特殊路径未接入逐层逻辑。

## 完成判定

- 全层可遍历、visit/forward 同序、抽样前向正常。
- 不因逐层改造破坏基础四步验证能力。
